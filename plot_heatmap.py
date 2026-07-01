import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

# ── Configurazione ────────────────────────────────────────────────────────────
CSV_PATH = 'all_metrics.csv'     
dir = 'C:/Users/irene/Desktop/DiffuMamba/'      
OUTPUT   = 'heatmap_probing.png'

# Checkpoint da mostrare come colonne
SELECTED_STEPS = [25000, 50000, 100000, 200000, 375000, 624000]
COL_LABELS     = ['25k', '50k', '100k', '200k', '375k', '624k']

# ── Caricamento dati ──────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df_sel = df[df['step'].isin(SELECTED_STEPS)].set_index('step').loc[SELECTED_STEPS]

features = [c for c in df.columns if c not in ('step', 'avg_rho')]

# ── Categorizzazione feature ──────────────────────────────────────────────────
def categorize(f):
    if f.startswith('upos_dist') or f.startswith('xpos_dist'):
        return 'POS'
    if f.startswith('dep_dist'):
        return 'Dipendenze'
    if f.startswith('verbs_'):
        return 'Morfologia verbale'
    if f in ('n_tokens', 'char_per_tok', 'lexical_density',
             'max_links_len', 'n_prepositional_chains'):
        return 'Superficie'
    return 'Sintassi'

CAT_ORDER = ['Superficie', 'POS', 'Dipendenze', 'Morfologia verbale', 'Sintassi']
CAT_COLORS = {
    'Superficie':         '#1565C0',
    'POS':                '#2E7D32',
    'Dipendenze':         '#E65100',
    'Morfologia verbale': '#6A1B9A',
    'Sintassi':           '#B71C1C',
}

# Ordina: per categoria, poi per ρ finale decrescente
rho_final = df_sel.loc[624000]
#per ordinamento per categorie di feature e poi ordine di p
#sort_key  = [(CAT_ORDER.index(categorize(f)), -rho_final[f]) for f in features]
order     = sorted(range(len(features)), key=lambda i: [features[i]])
features_sorted = [features[i] for i in order]
cats_sorted     = [categorize(f) for f in features_sorted]

# ── Matrice ───────────────────────────────────────────────────────────────────
matrix   = df_sel[features_sorted].values.T   # (n_feat, n_steps)
avg_row  = df_sel['avg_rho'].values            # (n_steps,)
full_matrix = np.vstack([matrix, avg_row])     # AVG come ultima riga

# ── Etichette riga ────────────────────────────────────────────────────────────
# def short_label(f):
#     f = f.replace('upos_dist_', '').replace('xpos_dist_', '')
#     f = f.replace('dep_dist_', 'dep:').replace('verbs_', '')
#     f = f.replace('mood+tense_dist_', '').replace('mood_dist_', '')
#     f = f.replace('tense_dist_', '').replace('num_pers_dist_', '')
#     f = f.replace('voice_dist_', '').replace('avg_', '')
#     f = f.replace('_dist', '').replace('dist_', '')
#     return f

row_labels = [f for f in features_sorted] + ['AVG']
row_cats   = cats_sorted + ['AVG']

# ── Figura ────────────────────────────────────────────────────────────────────
n_rows = len(row_labels)
n_cols = len(COL_LABELS)

cell_w = 0.72
cell_h = 0.22
fig_w  = 3.5 + n_cols * cell_w
fig_h  = n_rows * cell_h + 1.2

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.patch.set_facecolor('#FFFFFF')

cmap = mcolors.LinearSegmentedColormap.from_list(
    'wb', ['#FFFFFF', '#AED6F1', '#2980B9', '#154360'], N=256)

vmin, vmax = 0.28, 0.97
im = ax.imshow(full_matrix, aspect='auto', cmap=cmap,
               vmin=vmin, vmax=vmax, origin='upper')

# Valori numerici nelle celle
for r in range(n_rows):
    for c in range(n_cols):
        val = full_matrix[r, c]
        brightness = (val - vmin) / (vmax - vmin)
        txt_color  = 'white' if brightness > 0.55 else '#1a1a1a'
        fw = 'bold' if r == n_rows - 1 else 'normal'
        ax.text(c, r, f'{val:.2f}', ha='center', va='center',
                fontsize=7.2, color=txt_color, fontweight=fw)

# Asse X (in alto)
ax.set_xticks(range(n_cols))
ax.set_xticklabels(COL_LABELS, fontsize=9, fontweight='bold', color='#333')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_xlabel('Step di training', fontsize=10, labelpad=6)

# Asse Y con colori per categoria
ax.set_yticks(range(n_rows))
ax.set_yticklabels(row_labels, fontsize=7.5)
for tick, cat in zip(ax.get_yticklabels(), row_cats):
    color = CAT_COLORS.get(cat, '#222222')
    tick.set_color(color)
    if cat == 'AVG':
        tick.set_fontweight('bold')
        tick.set_color('#1B3A6B')

# Separatore sopra AVG
ax.axhline(n_rows - 1.5, color='#555', linewidth=1.2)

# # Legenda categorie
# from matplotlib.patches import Patch
# legend_elements = [Patch(facecolor=CAT_COLORS[c], label=c) for c in CAT_ORDER]
# ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
#           framealpha=0.95, edgecolor='#ccc', bbox_to_anchor=(1.0, -0.01))

# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.01, aspect=20)
cbar.set_label('Spearman ρ', fontsize=9)
cbar.ax.tick_params(labelsize=8)

ax.set_title('Probing linguistico — DiffuMamba (Run 2)',
             fontsize=12, fontweight='bold', color='#1B3A6B', pad=14)

for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig(dir+OUTPUT, dpi=160, bbox_inches='tight', facecolor='white')
print(f"Salvato: {OUTPUT}")