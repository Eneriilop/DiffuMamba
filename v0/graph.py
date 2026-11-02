import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np

df = pd.read_csv("run4/learning_curve_multilayer.csv")
steps_k = df["step"] / 1000
layer_cols = [f"layer_{i}" for i in range(8)]

BERT_RHO = 0.70

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#F8F9FA')
ax.grid(axis='y', color='#DDDDDD', linewidth=0.8, zorder=0)
ax.grid(axis='x', color='#EEEEEE', linewidth=0.5, zorder=0)
ax.set_axisbelow(True)

# palette: dal blu chiaro al blu scuro
colors = cm.Blues(np.linspace(0.35, 0.95, 8))

for i, col in enumerate(layer_cols):
    ax.plot(steps_k, df[col],
            color=colors[i], linewidth=1.8,
            marker='o', markersize=3,
            markerfacecolor=colors[i], markeredgewidth=0,
            zorder=3, label=f'Layer {i}')

# etichetta valore finale per ogni layer
for i, col in enumerate(layer_cols):
    final_val = df[col].iloc[-1]
    ax.annotate(f'{final_val:.3f}',
                xy=(steps_k.iloc[-1], final_val),
                xytext=(5, 0), textcoords='offset points',
                fontsize=7.5, color=colors[i], va='center')

# linea BERT
ax.axhline(y=BERT_RHO, color='#E05A2B', linewidth=1.8,
           linestyle='--', zorder=2, label='BERT ρ ≈ 0.70')

ax.set_xlabel('Step di training (×1000)', fontsize=11, color='#333333', labelpad=8)
ax.set_ylabel('Media ρ (Spearman)', fontsize=11, color='#333333', labelpad=8)
ax.set_title('Curva di apprendimento per layer — DiffuMamba (Run 4)',
             fontsize=13, fontweight='bold', color='#1B3A6B', pad=14)

ax.set_xlim(-1, 125)
ax.set_ylim(0.38, 0.74)
ax.set_xticks([0, 20, 40, 60, 80, 100, 120])
ax.tick_params(colors='#555555', labelsize=9)
for spine in ax.spines.values():
    spine.set_edgecolor('#CCCCCC')

ax.legend(fontsize=8.5, loc='lower right', framealpha=0.9,
          edgecolor='#CCCCCC', facecolor='white', ncol=2)

plt.tight_layout()
plt.savefig('run_mamba_sentences/learning_curve_multilayer.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()