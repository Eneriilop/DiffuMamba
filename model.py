# ============================================================
# model.py — DiffuMamba
# ============================================================
# Dipendenze:
#   pip install transformers torch
#
# Architettura:
#   TokenEmbedding + TimestepEmbedding
#       ↓
#   N × BidirectionalMambaBlock   (MambaMixer di HuggingFace)
#       ↓
#   LayerNorm → LM Head
#
# Note: 
#  - MambaMixer è un modello mantenuto da HuggingFace Transformers, utilizza la stessa logica di Mamba originale ma è ottimizzato per PyTorch e non ha kernel custom
#  - Utilizzo MambaMixer per implementare un blocco bidirezionale minimal: in futuro magari si potrà sostituire con mamba_ssm (versione originale di Mamba)
#  - Esiste bimamba-template (implementazione di mamba bidirectional minimale: https://github.com/yair-schiff/bimamba e https://huggingface.co/yairschiff/bimamba-template) ma sembra un progetto poco affidabile (poche stelle) e poco calcolato dalla community. Inoltre per inserire il timestep t (per il diffusion) andrebbe riadattato.
# ============================================================

import torch
import torch.nn as nn
from transformers import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaMixer

import config


# ------------------------------------------------------------
# Blocco Mamba bidirezionale
# ------------------------------------------------------------

class BiMambaBlock(nn.Module):
    """
    Esegue MambaMixer in entrambe le direzioni e somma i risultati.
    Usa MambaMixer di HuggingFace Transformers.
    """

    def __init__(self, mamba_cfg: MambaConfig, layer_idx: int):
        super().__init__()
        self.norm = nn.LayerNorm(mamba_cfg.hidden_size)
        # Due MambaMixer separati: uno per la direzione forward (left → right) e uno per la direzione backward (right → left) -> pesi separati
        self.mamba_fwd = MambaMixer(mamba_cfg, layer_idx=layer_idx)
        self.mamba_bwd = MambaMixer(mamba_cfg, layer_idx=layer_idx)

    # Normalizza, processa in entrambe le direzioni e somma i risultati con residual connection
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        residual = x
        x = self.norm(x)
        fwd = self.mamba_fwd(x)                     # left → right
        bwd = self.mamba_bwd(x.flip(1)).flip(1)     # right → left
        return residual + fwd + bwd                 # residual connection


# ------------------------------------------------------------
# Modello completo
# ------------------------------------------------------------

class DiffuMamba(nn.Module):

    def __init__(self):
        super().__init__()

        # MambaConfig di HuggingFace solo campi base - configurazione del modello
        mamba_cfg = MambaConfig(
            vocab_size=config.HIDDEN_SIZE,   # non usato direttamente, serve per init
            hidden_size=config.HIDDEN_SIZE,
            state_size=config.STATE_SIZE,
            num_hidden_layers=config.NUM_LAYERS,
            expand=config.EXPAND,
            conv_kernel=config.CONV_KERNEL,
            use_mambapy=True                # usa mamba.py (Python vettorizzato, più veloce della fallback sequenziale)
                                            # Per i kernel CUDA nativi: pip install mamba-ssm causal-conv1d
        )

        # Embedding token
        self.token_emb = nn.Embedding(
            num_embeddings=31102,            # vocabolario bert-base-italian-uncased
            embedding_dim=config.HIDDEN_SIZE,
            padding_idx=config.PAD_TOKEN_ID,
        )

        # Timestep embedding: t ∈ [0,1] → vettore D
        # Il Timestep t è scalare: viene trasformato in vettore di dim HIDDEN_SIZE con un singolo layer lineare
        self.time_emb = nn.Linear(1, config.HIDDEN_SIZE)

        # Stack di blocchi Bi-Mamba
        self.layers = nn.ModuleList([
            BiMambaBlock(mamba_cfg, layer_idx=i)
            for i in range(config.NUM_LAYERS)
        ])

        # Normalizzazione finale prima della predizione 
        self.norm_out = nn.LayerNorm(config.HIDDEN_SIZE)

        # LM head: proietta ogni vettore hidden (128) in un punteggio per ciascun token del vocabolario (31102)
        # Weight tying: condivide la stessa matrice di pesi con token_emb
        # → il vettore che rappresenta un token in input è lo stesso usato per predirlo in output
        # → risparmia ~4M parametri e migliora la generalizzazione (standard in BERT, GPT-2)
        self.lm_head = nn.Linear(config.HIDDEN_SIZE, 31102, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        input_ids : (B, L)  — sequenza con [MASK] inseriti
        t         : (B,)    — timestep in [0, 1]
        returns   : (B, L, vocab_size) -> contiene per ogni token la distribuzione predetta dal modello su tutto il vocabolario
        """
        x = self.token_emb(input_ids)                    # (B, L, D)

        # aggiungi timestep come bias sull'intera sequenza
        t_vec = self.time_emb(t.unsqueeze(1))            # (B, D)
        x = x + t_vec.unsqueeze(1)                       # broadcast su L

        for layer in self.layers:
            x = layer(x)

        x = self.norm_out(x)
        return self.lm_head(x)                           # (B, L, vocab_size)

    # Conta il numero di parametri addestrabili (utile per debug e confronto con BERT)
    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
