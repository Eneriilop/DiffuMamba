# ============================================================
# seeds.py — seed usati per tutti i modelli
# ============================================================
# Il seed controlla l'inizializzazione dei pesi e il rumore della
# diffusion (t e posizioni mascherate). L'ordine dei dati NON dipende
# dal seed: è fisso e identico per tutti i modelli (confronti appaiati).
# Il probe (MinMaxScaler + Ridge) è deterministico e non usa seed.

import random
import numpy as np
import torch

SEEDS = [0, 1, 2]


def set_seed(seed: int):
    """Fissa tutti i generatori casuali. Va chiamata prima di creare il modello."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)            # CPU (inizializzazione dei pesi)
    torch.cuda.manual_seed_all(seed)   # GPU (t e posizioni mascherate)
