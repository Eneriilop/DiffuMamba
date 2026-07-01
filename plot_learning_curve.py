import matplotlib.pyplot as plt

# ── Dati ──────────────────────────────────────────────────────────────────────
import pandas as pd

df = pd.read_csv("learning_curve.csv")
steps = df["step"].tolist()
rho = df["avg_rho"].tolist()
steps_k = [s / 1000 for s in steps]

BERT_RHO   = 0.70
RUN1_STEP  = 208        # in migliaia
RUN1_RHO   = 0.68
EPOCH1_END = 208        # step fine prima epoca (in migliaia)

# ── Figura ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#F8F9FA')

# Griglia
ax.grid(axis='y', color='#DDDDDD', linewidth=0.8, zorder=0)
ax.grid(axis='x', color='#EEEEEE', linewidth=0.5, zorder=0)
ax.set_axisbelow(True)

# Sfondo prima epoca
ax.axvspan(0, EPOCH1_END, alpha=0.06, color='#1B3A6B', zorder=1)
ax.text(EPOCH1_END / 2, 0.610, 'Epoca 1',
        ha='center', fontsize=8, color='#1B3A6B', alpha=0.7)
ax.text((EPOCH1_END + 624) / 2, 0.610, 'Epoche 2–3',
        ha='center', fontsize=8, color='#555555', alpha=0.7)

# Linea BERT di riferimento
ax.axhline(y=BERT_RHO, color='#E05A2B', linewidth=1.8, linestyle='--', zorder=2,
           label=f'BERT-medium (Dini et al. 2026) — ρ = {BERT_RHO}')

# Linea verticale fine Run 1
ax.axvline(x=RUN1_STEP, color='#5A7FC0', linewidth=1.2, linestyle=':', zorder=2, alpha=0.8)
ax.annotate(f'Fine Run 1\n({RUN1_STEP}k step, ρ={RUN1_RHO})',
            xy=(RUN1_STEP, RUN1_RHO),
            xytext=(RUN1_STEP + 25, 0.655),
            fontsize=8.5, color='#5A7FC0',
            arrowprops=dict(arrowstyle='->', color='#5A7FC0', lw=1.2),
            zorder=5)

# Curva DiffuMamba
ax.plot(steps_k, rho,
        color='#1B3A6B', linewidth=2.5,
        marker='o', markersize=7,
        markerfacecolor='white', markeredgecolor='#1B3A6B', markeredgewidth=2,
        zorder=3, label='DiffuMamba (BiMamba)')

# Etichette sui punti
idx_max = rho.index(max(rho))
label_indices = [0, 1, 3, 7, 12, idx_max, len(rho) - 1]

for i, (x, y) in enumerate(zip(steps_k, rho)):
    if i in label_indices:
        ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10),
                    textcoords='offset points', ha='center',
                    fontsize=8, color='#1B3A6B', fontweight='bold')
        
# Assi e titolo
ax.set_xlabel('Step di training (×1000)', fontsize=11, color='#333333', labelpad=8)
ax.set_ylabel('Media ρ (Spearman)', fontsize=11, color='#333333', labelpad=8)
ax.set_title('Curva di apprendimento — DiffuMamba (Run 2, 25 checkpoint)',
             fontsize=13, fontweight='bold', color='#1B3A6B', pad=14)

ax.set_xlim(0, 660)
ax.set_ylim(0.58, 0.72)
ax.set_xticks([0, 100, 200, 300, 400, 500, 600])
ax.tick_params(colors='#555555', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')

ax.legend(fontsize=9, loc='lower right', framealpha=0.9,
          edgecolor='#CCCCCC', facecolor='white')

plt.tight_layout()
plt.savefig('curva_apprendimento_DiffuMamba.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()