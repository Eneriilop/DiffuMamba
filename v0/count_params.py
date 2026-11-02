from mamba_ssm import Mamba
import torch.nn as nn

def count_params(d_model, d_state, d_conv, expand, num_layers):
    layers = nn.ModuleList([
        nn.ModuleList([
            nn.LayerNorm(d_model),
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand),
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand),
        ]) for _ in range(num_layers)
    ])
    emb = nn.Embedding(31102, d_model)
    time_emb = nn.Linear(1, d_model)
    norm_out = nn.LayerNorm(d_model)
    
    total = sum(p.numel() for m in [layers, emb, time_emb, norm_out] 
                for p in m.parameters())
    return total

for d_model in [480, 496, 512, 528, 544]:
    n = count_params(d_model=d_model, d_state=16, d_conv=4, expand=2, num_layers=8)
    print(f"d_model={d_model}: {n/1e6:.1f}M parametri")