# ============================================================
# model.py — DiffuMamba
# ============================================================
# Dipendenze:
#   pip install transformers torch
#   pip install torch mamba-ssm causal-conv1d
#
# Architettura:
#   TokenEmbedding + TimestepEmbedding
#       ↓
#   N × BidirectionalMambaBlock   
#       ↓
#   LayerNorm → LM Head
#
# Note: 
#  - Originalmente implementato con MambaMixer di HuggingFace Transformers (backend mamba.py)
#  - Migrato a mamba_ssm (versione originale di Mamba con kernel CUDA ottimizzati)
#  - Richiede CUDA >= 11.6: pip install mamba-ssm causal-conv1d
# ============================================================

import torch
import torch.nn as nn
# from transformers import MambaConfig
# from transformers.models.mamba.modeling_mamba import MambaMixer
from mamba_ssm import Mamba
import config


# ------------------------------------------------------------
# Blocco Mamba bidirezionale
# ------------------------------------------------------------

class BiMambaBlock(nn.Module):
    """
    Esegue Mamba in entrambe le direzioni e somma i risultati.
    Usa mamba_ssm.
    """

    def __init__(self, d_model, d_state, d_conv, expand, layer_idx: int):
        super().__init__()
        # Due Mamba separati: uno per la direzione forward (left → right) e uno per la direzione backward (right → left) -> pesi separati
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        #Code MambaMixer di HuggingFace
        # self.norm = nn.LayerNorm(mamba_cfg.hidden_size)
        # self.mamba_fwd = MambaMixer(mamba_cfg, layer_idx=layer_idx)
        # self.mamba_bwd = MambaMixer(mamba_cfg, layer_idx=layer_idx)

    # Normalizza, processa in entrambe le direzioni e somma i risultati con residual connection
    # Ogni layer prende in input l'output del layer precedente (o l'embedding iniziale) e produce un output della stessa dimensione (con l'aggiunta dei risultati delle scan)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        residual = x
        x = self.norm(x)
        fwd = self.mamba_fwd(x)                     # left → right
        """
        Per la direzione backward, invertiamo la sequenza in input (flip su dimensione 1), applichiamo Mamba e poi invertiamo di nuovo l'output per riportarlo nell'ordine originale.
        Implemetazione usata in Vision Mamba - Efficient Visual Representation Learning with Bidirectional State Space Model
        """
        bwd = self.mamba_bwd(x.flip(1)).flip(1)     # right → left
        return residual + fwd + bwd                 # residual connection


# ------------------------------------------------------------
# Modello completo
# ------------------------------------------------------------

class DiffuMamba(nn.Module):

    def __init__(self):
        super().__init__()

        # Embedding token
        self.token_emb = nn.Embedding(
            num_embeddings=31102,
            embedding_dim=config.HIDDEN_SIZE,
            padding_idx=config.PAD_TOKEN_ID,
        )

        # Timestep embedding
        self.time_emb = nn.Linear(1, config.HIDDEN_SIZE)

        # Stack di blocchi Bi-Mamba
        self.layers = nn.ModuleList([
            BiMambaBlock(
                d_model=config.HIDDEN_SIZE,
                d_state=config.STATE_SIZE,
                d_conv=config.CONV_KERNEL,
                expand=config.EXPAND,
                layer_idx=i,
            )
            for i in range(config.NUM_LAYERS)
        ])

        # Normalizzazione finale
        self.norm_out = nn.LayerNorm(config.HIDDEN_SIZE)

        # LM head con weight tying
        self.lm_head = nn.Linear(config.HIDDEN_SIZE, 31102, bias=False)
        self.lm_head.weight = self.token_emb.weight

        # inizializzazione BERT-style per stabilità
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.time_emb.weight, std=0.02)
        nn.init.zeros_(self.time_emb.bias)


    def forward(self, input_ids: torch.Tensor, t: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        input_ids      : (B, L)
        t              : (B,)
        attention_mask : ignorata, serve solo per avere la stessa firma di DiffuTransformer
        returns        : (B, L, vocab_size)
        """
        x = self.token_emb(input_ids)          # (B, L, D)
        t_vec = self.time_emb(t.unsqueeze(1))  # (B, D)
        x = x + t_vec.unsqueeze(1)             # broadcast su L

        for layer in self.layers:
            x = layer(x)

        x = self.norm_out(x)
        return self.lm_head(x)                 # (B, L, vocab_size)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Mamba AR — ablation dell'obiettivo
# ============================================================
# Mamba autoregressivo classico (next-token prediction), unidirezionale:
#   TokenEmbedding
#       ↓
#   N × MambaBlock (un solo Mamba, sinistra → destra)
#       ↓
#   LayerNorm → LM Head
#
# Rispetto a DiffuMamba: niente direzione backward, niente timestep
# embedding (non c'è rumore). Per avere gli stessi parametri ogni blocco
# usa expand = 4 invece di 2 (vedi config_ar_sentences.py / config_ar_blocks.py).
# ------------------------------------------------------------

class MambaBlock(nn.Module):
    """Blocco pre-LN unidirezionale: x + Mamba(LN(x)). Causale: il token i vede solo i token ≤ i."""

    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mamba(self.norm(x))         # left → right + residual connection


class MambaAR(nn.Module):

    def __init__(self, cfg):
        # cfg: config_ar_sentences (frasi 128) o config_ar_blocks (blocchi 512)
        super().__init__()

        self.token_emb = nn.Embedding(
            num_embeddings=31102,
            embedding_dim=cfg.HIDDEN_SIZE,
            padding_idx=cfg.PAD_TOKEN_ID,
        )

        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=cfg.HIDDEN_SIZE,
                d_state=cfg.STATE_SIZE,
                d_conv=cfg.CONV_KERNEL,
                expand=cfg.EXPAND,
            )
            for _ in range(cfg.NUM_LAYERS)
        ])

        self.norm_out = nn.LayerNorm(cfg.HIDDEN_SIZE)

        # LM head con weight tying, come DiffuMamba
        self.lm_head = nn.Linear(cfg.HIDDEN_SIZE, 31102, bias=False)
        self.lm_head.weight = self.token_emb.weight

        nn.init.normal_(self.token_emb.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor, t: torch.Tensor = None,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        input_ids      : (B, L)
        t, attention_mask : ignorati (stessa firma degli altri modelli);
                         il padding è a destra e, essendo il modello causale,
                         non influenza i token reali
        returns        : (B, L, vocab_size); la posizione i predice il token i+1
        """
        x = self.token_emb(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)
        return self.lm_head(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)