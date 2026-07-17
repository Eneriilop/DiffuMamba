import matplotlib.pyplot as plt
import pandas as pd

# ── Dati ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("run_mamba_sentences/all_metrics_mamba_sent.csv")
steps = df["step"].tolist()
rho   = df["avg_rho"].tolist()
steps_k = [s / 1000 for s in steps]

BERT_RHO    = 0.70
EPOCH1_END  = 39.062   # 10M / 256 ≈ 39062 step → ~39k
EPOCH2_END  = 78.125
MAX_STEP_K  = 120

# ── Figura ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#F8F9FA')

ax.grid(axis='y', color='#DDDDDD', linewidth=0.8, zorder=0)
ax.grid(axis='x', color='#EEEEEE', linewidth=0.5, zorder=0)
ax.set_axisbelow(True)

# Sfondo epoche
ax.axvspan(0,          EPOCH1_END, alpha=0.06, color='#1B3A6B', zorder=1)
ax.axvspan(EPOCH1_END, EPOCH2_END, alpha=0.03, color='#1B3A6B', zorder=1)
ax.text(EPOCH1_END / 2,            min(rho) - 0.005, 'Epoca 1', ha='center', fontsize=8, color='#1B3A6B', alpha=0.7)
ax.text((EPOCH1_END + EPOCH2_END) / 2, min(rho) - 0.005, 'Epoca 2', ha='center', fontsize=8, color='#555555', alpha=0.7)
ax.text((EPOCH2_END + MAX_STEP_K) / 2, min(rho) - 0.005, 'Epoca 3', ha='center', fontsize=8, color='#555555', alpha=0.7)

# Linea BERT
ax.axhline(y=BERT_RHO, color='#E05A2B', linewidth=1.8, linestyle='--', zorder=2,
           label=f'BERT-medium (Dini et al. 2026) — ρ ≈ {BERT_RHO}')

# Curva DiffuMamba
ax.plot(steps_k, rho,
        color='#1B3A6B', linewidth=2.5,
        marker='o', markersize=5,
        markerfacecolor='white', markeredgecolor='#1B3A6B', markeredgewidth=1.5,
        zorder=3, label='DiffuMamba (BiMamba, Run 3)')

# Etichette su punti chiave
idx_max  = rho.index(max(rho))
idx_last = len(rho) - 1
# primo checkpoint, fine epoca 1, fine epoca 2, max, ultimo
label_indices = {0, 10, 19, idx_max, idx_last}
for i, (x, y) in enumerate(zip(steps_k, rho)):
    if i in label_indices:
        ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10),
                    textcoords='offset points', ha='center',
                    fontsize=8, color='#1B3A6B', fontweight='bold')

# Assi e titolo
ax.set_xlabel('Step di training (×1000)', fontsize=11, color='#333333', labelpad=8)
ax.set_ylabel('Media ρ (Spearman)', fontsize=11, color='#333333', labelpad=8)
ax.set_title('Curva di apprendimento — DiffuMamba (Run 3, 39 checkpoint)',
             fontsize=13, fontweight='bold', color='#1B3A6B', pad=14)

ax.set_xlim(0, MAX_STEP_K)
ax.set_ylim(min(rho) - 0.01, max(max(rho), BERT_RHO) + 0.02)
ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
ax.tick_params(colors='#555555', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')

ax.legend(fontsize=9, loc='lower right', framealpha=0.9,
          edgecolor='#CCCCCC', facecolor='white')

plt.tight_layout()
plt.savefig('learning_curve_mamba_sent.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()