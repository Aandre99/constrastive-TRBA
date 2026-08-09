"""
tps.py – Visualiza a saída do módulo TPS (Thin Plate Spline) em N imagens.

Uso:
    python visualize/tps.py \
        --model saved_models/NOME/best_accuracy.pth \
        --image_folder demo_image/ \
        --N 8 \
        --output tps_output.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

# ── repositório raiz no path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules.transformation import TPS_SpatialTransformerNetwork

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_tps(model_path: str, F: int, imgH: int, imgW: int, channels: int) -> TPS_SpatialTransformerNetwork:
    tps = TPS_SpatialTransformerNetwork(
        F=F,
        I_size=(imgH, imgW),
        I_r_size=(imgH, imgW),
        I_channel_num=channels,
    ).to(DEVICE)

    state = torch.load(model_path, map_location=DEVICE)
    # suporta checkpoint completo (model.Transformation.*) ou apenas o submódulo
    if any(k.startswith('module.Transformation.') for k in state):
        prefix = 'module.Transformation.'
    elif any(k.startswith('Transformation.') for k in state):
        prefix = 'Transformation.'
    else:
        prefix = ''

    if prefix:
        tps_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        tps.load_state_dict(tps_state, strict=False)
    else:
        tps.load_state_dict(state, strict=False)

    tps.eval()
    return tps


def collect_images(folder: str, n: int, rgb: bool):
    """Retorna lista de (original_np, rectified_np) para as primeiras N imagens."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    paths = sorted(p for p in Path(folder).rglob('*') if p.suffix.lower() in exts)
    if not paths:
        sys.exit(f'[erro] Nenhuma imagem encontrada em: {folder}')
    paths = paths[:n]

    mode = 'RGB' if rgb else 'L'
    channels = 3 if rgb else 1

    transform = T.Compose([
        T.Resize((opt.imgH, opt.imgW)),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * channels, std=[0.5] * channels),
    ])

    tps = load_tps(opt.model, opt.num_fiducial, opt.imgH, opt.imgW, channels)

    results = []
    for p in paths:
        img_pil = Image.open(p).convert(mode)
        tensor = transform(img_pil).unsqueeze(0).to(DEVICE)  # [1, C, H, W]

        with torch.no_grad():
            rectified = tps(tensor)

        def to_np(t):
            arr = t.squeeze(0).cpu().numpy()          # [C, H, W]
            arr = np.clip((arr * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
            return arr.transpose(1, 2, 0)             # [H, W, C]

        results.append((to_np(tensor), to_np(rectified)))

    return results


def render(pairs, output: str):
    n = len(pairs)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5.5), facecolor='#12121e')
    fig.subplots_adjust(hspace=0.06, wspace=0.04, top=0.88, bottom=0.02, left=0.02, right=0.98)

    row_labels = ['Original', 'TPS']
    label_colors = ['#7eb8f7', '#a5f791']

    for col, (orig, rect) in enumerate(pairs):
        for row, (img, color) in enumerate(zip([orig, rect], label_colors)):
            ax = axes[row, col] if n > 1 else axes[row]
            cmap = None if img.shape[2] == 3 else 'gray'
            ax.imshow(img.squeeze() if img.shape[2] == 1 else img, cmap=cmap, aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(color)
                sp.set_linewidth(1.2)
            if col == 0:
                ax.set_ylabel(row_labels[row], color=color, fontsize=9, fontweight='bold')

#    fig.suptitle('TPS Spatial Transformer — Original vs Rectified',
#                 color='#e0e0ff', fontsize=13, fontweight='bold')

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Salvo em: {output}')


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model',        required=True,  help='Checkpoint .pth do modelo')
    parser.add_argument('--image_folder', required=True,  help='Pasta com imagens de entrada')
    parser.add_argument('--N',            type=int, default=8, help='Número de imagens')
    parser.add_argument('--output',       default='tps_output.png', help='PNG de saída')
    parser.add_argument('--imgH',         type=int, default=32)
    parser.add_argument('--imgW',         type=int, default=100)
    parser.add_argument('--num_fiducial', type=int, default=20)
    parser.add_argument('--rgb',          action='store_true')
    opt = parser.parse_args()

    pairs = collect_images(opt.image_folder, opt.N, opt.rgb)
    render(pairs, opt.output)
