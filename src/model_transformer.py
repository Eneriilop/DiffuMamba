# ============================================================
# model_transformer.py — DiffuTransformer (ablation di DiffuMamba)
# ============================================================
# Stessa pipeline di DiffuMamba (embedding, timestep, LM head,
# masked diffusion), ma i blocchi Bi-Mamba sono sostituiti da
# blocchi Transformer encoder (attenzione bidirezionale).
#
# Architettura:
#   TokenEmbedding + PositionEmbedding + TimestepEmbedding
#       ↓
#   N × TransformerBlock (pre-LN, come BiMambaBlock)
#       ↓
#   LayerNorm → LM Head
#
# Note:
#  - Mamba codifica l'ordine con la scan, l'attenzione no: serve
#    un positional embedding (appreso, come BERT)
#  - FFN_SIZE è scelto in modo che ogni blocco abbia circa gli
#    stessi parametri di un BiMambaBlock (vedi config_transformer.py)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import config_transformer as config


# ------------------------------------------------------------
# Blocco Transformer
# ------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Blocco encoder pre-LN: x + Attn(LN(x)), poi x + FFN(LN(x)).
    Stessa posizione della LayerNorm di BiMambaBlock.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model deve essere divisibile per n_heads"
        self.n_heads = n_heads
        self.dropout = dropout

        self.norm_attn = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attn_out = nn.Linear(d_model, d_model)

        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, L, D), attention_mask: (B, L) con 1 = token, 0 = padding
        B, L, D = x.shape

        h = self.norm_attn(x)
        q, k, v = self.qkv(h).view(B, L, 3, self.n_heads, D // self.n_heads).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))   # (B, H, L, D/H)

        # nessuna maschera causale: ogni token vede tutta la sequenza (come BERT).
        # Le chiavi di padding vengono escluse; con i blocchi 512 non c'è padding.
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.bool()[:, None, None, :]   # (B, 1, 1, L)

        a = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        a = a.transpose(1, 2).reshape(B, L, D)
        x = x + self.attn_out(a)                    # residual connection

        x = x + self.ffn(self.norm_ffn(x))          # residual connection
        return x


# ------------------------------------------------------------
# Modello completo
# ------------------------------------------------------------

class DiffuTransformer(nn.Module):

    def __init__(self, cfg=config):
        # cfg: config_transformer (blocchi 512) o config_transformer_sentences (frasi 128)
        super().__init__()

        # Embedding token
        self.token_emb = nn.Embedding(
            num_embeddings=31102,
            embedding_dim=cfg.HIDDEN_SIZE,
            padding_idx=cfg.PAD_TOKEN_ID,
        )

        # Positional embedding appreso (assente in DiffuMamba)
        self.pos_emb = nn.Embedding(cfg.MAX_SEQ_LEN, cfg.HIDDEN_SIZE)

        # Timestep embedding
        self.time_emb = nn.Linear(1, cfg.HIDDEN_SIZE)

        # Stack di blocchi Transformer
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=cfg.HIDDEN_SIZE,
                n_heads=cfg.NUM_HEADS,
                d_ff=cfg.FFN_SIZE,
                dropout=cfg.DROPOUT,
            )
            for _ in range(cfg.NUM_LAYERS)
        ])

        # Normalizzazione finale
        self.norm_out = nn.LayerNorm(cfg.HIDDEN_SIZE)

        # LM head con weight tying
        self.lm_head = nn.Linear(cfg.HIDDEN_SIZE, 31102, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # inizializzazione BERT-style per stabilità
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def embed(self, input_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Input del primo blocco: token + posizione + timestep.
        Separato da forward() così il probing può iterare sui layer.
        """
        L = input_ids.size(1)
        pos = torch.arange(L, device=input_ids.device)
        x = self.token_emb(input_ids) + self.pos_emb(pos)   # (B, L, D)
        t_vec = self.time_emb(t.unsqueeze(1))               # (B, D)
        return x + t_vec.unsqueeze(1)                       # broadcast su L

    def forward(self, input_ids: torch.Tensor, t: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        input_ids      : (B, L)
        t              : (B,)
        attention_mask : (B, L) opzionale, 1 = token, 0 = padding
        returns        : (B, L, vocab_size)
        """
        x = self.embed(input_ids, t)

        for layer in self.layers:
            x = layer(x, attention_mask)

        x = self.norm_out(x)
        return self.lm_head(x)                 # (B, L, vocab_size)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
