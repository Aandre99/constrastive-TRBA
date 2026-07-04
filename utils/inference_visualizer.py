"""
infer_visualize.py  –  Inference com visualização via Matplotlib
================================================================
Aceita uma imagem única ou uma pasta com imagens (.jpg/.jpeg/.png).
Roda inferência em batch (--bsize) e salva uma imagem de saída para
cada entrada, com o texto reconhecido como título da figura.

Exemplo de uso:
    python infer_visualize.py \
        --input demo_image/ \
        --saved_model saved_models/TPS-ResNet-BiLSTM-Attn.pth \
        --Transformation TPS --FeatureExtraction ResNet \
        --SequenceModeling BiLSTM --Prediction Attn \
        --output_dir result_visualize/

    # Imagem única:
    python infer_visualize.py \
        --input demo_image/img_000002.jpg \
        --saved_model saved_models/TPS-ResNet-BiLSTM-Attn.pth \
        --Transformation TPS --FeatureExtraction ResNet \
        --SequenceModeling BiLSTM --Prediction Attn
"""

import os
import sys
import math
import string
import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')          # backend sem display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from natsort import natsorted

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import AlignCollate, RawDataset
from model import Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────────────────────
# Dataset auxiliar: aceita lista de paths (arquivo único ou pasta)
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


class ImageListDataset(torch.utils.data.Dataset):
    """Dataset que aceita um único arquivo ou uma pasta de imagens."""

    def __init__(self, input_path: str, opt):
        self.opt = opt
        p = Path(input_path)

        if p.is_file():
            if p.suffix.lower() not in SUPPORTED_EXT:
                raise ValueError(f"Formato não suportado: {p.suffix}")
            self.image_path_list = [str(p.resolve())]
        elif p.is_dir():
            paths = [
                str(f.resolve())
                for f in p.rglob('*')
                if f.suffix.lower() in SUPPORTED_EXT
            ]
            self.image_path_list = natsorted(paths)
            if not self.image_path_list:
                raise FileNotFoundError(f"Nenhuma imagem encontrada em: {input_path}")
        else:
            raise FileNotFoundError(f"Caminho inválido: {input_path}")

        self.nSamples = len(self.image_path_list)
        print(f"[dataset] {self.nSamples} imagem(ns) encontrada(s).")

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        path = self.image_path_list[index]
        try:
            if self.opt.rgb:
                img = Image.open(path).convert('RGB')
            else:
                img = Image.open(path).convert('L')
        except (IOError, OSError):
            print(f"[aviso] Imagem corrompida: {path}  — usando imagem preta.")
            mode = 'RGB' if self.opt.rgb else 'L'
            img = Image.new(mode, (self.opt.imgW, self.opt.imgH))
        return img, path


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento do modelo
# ─────────────────────────────────────────────────────────────────────────────

def load_model(opt):
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt)
    print(
        f"[model] {opt.Transformation}-{opt.FeatureExtraction}-"
        f"{opt.SequenceModeling}-{opt.Prediction}  |  "
        f"classes={opt.num_class}  device={device}"
    )
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    model.eval()
    print(f"[model] Pesos carregados de: {opt.saved_model}")
    return model, converter


# ─────────────────────────────────────────────────────────────────────────────
# Geração das imagens de saída via Matplotlib
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_bar_color(score: float) -> str:
    """Verde → amarelo → vermelho conforme a confiança."""
    if score >= 0.75:
        return '#2ecc71'
    elif score >= 0.40:
        return '#f39c12'
    return '#e74c3c'


def save_result_image(
    img_path: str,
    pred_text: str,
    confidence: float,
    output_path: str,
    dpi: int = 150,
):
    """Salva uma figura Matplotlib com a imagem de entrada e o texto predito."""

    # ── leitura da imagem original (para exibição) ──────────────────────────
    try:
        orig = Image.open(img_path).convert('RGB')
    except Exception:
        orig = Image.new('RGB', (100, 32), color=(40, 40, 40))
    orig_np = np.asarray(orig)

    # ── layout ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(7, 3.2), facecolor='#1a1a2e')
    gs = gridspec.GridSpec(
        2, 1,
        height_ratios=[5, 1],
        hspace=0.08,
        left=0.04, right=0.96, top=0.82, bottom=0.04,
    )

    # — painel da imagem —
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(orig_np, aspect='auto')
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_edgecolor('#4a4a6a')
        spine.set_linewidth(1.5)

    # — barra de confiança —
    ax_bar = fig.add_subplot(gs[1])
    bar_color = _confidence_bar_color(confidence)
    ax_bar.barh(0, confidence, height=0.6, color=bar_color, alpha=0.85)
    ax_bar.barh(0, 1.0,        height=0.6, color='#2c2c4a', alpha=0.6, zorder=0)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.5, 0.5)
    ax_bar.axis('off')
    ax_bar.text(
        confidence + 0.02, 0,
        f'{confidence:.2%}',
        va='center', ha='left',
        color=bar_color, fontsize=8, fontweight='bold',
    )
    ax_bar.text(
        -0.01, 0,
        'conf',
        va='center', ha='right',
        color='#888888', fontsize=7,
    )

    # — título com o texto reconhecido —
    display_pred = pred_text if pred_text else '(vazio)'
    fig.suptitle(
        display_pred,
        fontsize=16,
        fontweight='bold',
        color='#e0e0ff',
        y=0.96,
        fontfamily='monospace',
    )

    # — nome do arquivo (rodapé discreto) —
    fig.text(
        0.96, 0.01,
        Path(img_path).name,
        ha='right', va='bottom',
        fontsize=6, color='#555577',
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Loop de inferência
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(opt):
    model, converter = load_model(opt)

    dataset = ImageListDataset(opt.input, opt)
    collate_fn = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.bsize,
        shuffle=False,
        num_workers=opt.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
    )

    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(dataset)
    processed = 0

    print(f"\n{'─'*60}")
    print(f"{'Arquivo':<35}  {'Predição':<20}  {'Conf':>6}")
    print(f"{'─'*60}")

    with torch.no_grad():
        for image_tensors, image_paths in loader:
            batch_size = image_tensors.size(0)
            images = image_tensors.to(device)

            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred  = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            # ── forward ────────────────────────────────────────────────────
            if 'CTC' in opt.Prediction:
                preds = model(images, text_for_pred)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, preds_size)
            else:
                preds = model(images, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            # ── confiança ──────────────────────────────────────────────────
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for img_path, pred, pred_max_prob in zip(image_paths, preds_str, preds_max_prob):
                # Pós-processamento Attn: cortar após token [s]
                if 'Attn' in opt.Prediction:
                    eos_idx = pred.find('[s]')
                    pred = pred[:eos_idx]
                    pred_max_prob = pred_max_prob[:eos_idx]

                try:
                    confidence = float(pred_max_prob.cumprod(dim=0)[-1])
                except Exception:
                    confidence = 0.0

                # ── nome do arquivo de saída ───────────────────────────────
                stem = Path(img_path).stem
                out_name = f"{stem}_pred.png"
                out_path = str(output_dir / out_name)

                save_result_image(img_path, pred, confidence, out_path, dpi=opt.dpi)

                short_name = Path(img_path).name
                print(f"{short_name:<35}  {pred:<20}  {confidence:>5.2%}  → {out_name}")
                processed += 1

    print(f"{'─'*60}")
    print(f"✓ {processed}/{total} imagem(ns) processada(s).  Resultados em: {output_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Inferência com visualização Matplotlib (deep-text-recognition-benchmark)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── entrada / saída ──────────────────────────────────────────────────────
    parser.add_argument('--input', required=True,
                        help='Imagem única (.jpg/.png/…) ou pasta contendo imagens')
    parser.add_argument('--output_dir', default='result_visualize/',
                        help='Pasta onde as imagens de saída serão salvas')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI das imagens de saída')

    # ── modelo ───────────────────────────────────────────────────────────────
    parser.add_argument('--saved_model', required=True,
                        help='Caminho para o arquivo .pth do modelo treinado')

    # ── DataLoader ───────────────────────────────────────────────────────────
    parser.add_argument('--bsize', type=int, default=16,
                        help='Tamanho do batch para inferência')
    parser.add_argument('--workers', type=int, default=4,
                        help='Número de workers do DataLoader')

    # ── pré-processamento ────────────────────────────────────────────────────
    parser.add_argument('--batch_max_length', type=int, default=25,
                        help='Comprimento máximo do label')
    parser.add_argument('--imgH', type=int, default=32,
                        help='Altura da imagem de entrada')
    parser.add_argument('--imgW', type=int, default=100,
                        help='Largura da imagem de entrada')
    parser.add_argument('--rgb', action='store_true',
                        help='Usar entrada colorida (3 canais)')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz',
                        help='Conjunto de caracteres')
    parser.add_argument('--sensitive', action='store_true',
                        help='Modo case-sensitive (94 caracteres imprimíveis)')
    parser.add_argument('--PAD', action='store_true',
                        help='Manter proporção e preencher com padding')

    # ── arquitetura ──────────────────────────────────────────────────────────
    parser.add_argument('--Transformation', type=str, required=True,
                        help='Estágio de transformação: None | TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='Extrator de features: VGG | RCNN | ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True,
                        help='Modelagem sequencial: None | BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True,
                        help='Predição: CTC | Attn')
    parser.add_argument('--num_fiducial', type=int, default=20,
                        help='Pontos fiduciais do TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='Canais de entrada do extrator')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='Canais de saída do extrator')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Tamanho do estado oculto do LSTM')

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    opt = parse_args()

    if opt.sensitive:
        opt.character = string.printable[:-6]   # 94 caracteres (mesmo que ASTER)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    run_inference(opt)
