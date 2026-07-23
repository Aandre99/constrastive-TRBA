"""
infer_visualize.py  –  Inference com visualização via Matplotlib
================================================================
Aceita uma imagem única ou uma pasta com imagens (.jpg/.jpeg/.png).
Roda inferência em batch (--bsize) e salva uma imagem de saída para
cada entrada, com o texto reconhecido como título da figura.

Suporta dois tipos de modelos:
  • Base     : modelo padrão (sem perda contrastiva)
  • Contrastivo : modelo treinado com --use_contrastive (Triplet Loss
                  nos hidden states do decoder de atenção)

Exemplo de uso (modelo base):
    python evaluate.py \
        --input demo_image/ \
        --saved_model saved_models/TPS-ResNet-BiLSTM-Attn.pth \
        --Transformation TPS --FeatureExtraction ResNet \
        --SequenceModeling BiLSTM --Prediction Attn \
        --output_dir result_visualize/

Exemplo de uso (modelo contrastivo):
    python evaluate.py \
        --input demo_image/ \
        --saved_model saved_models/TPS-ResNet-BiLSTM-Attn-Contrastive.pth \
        --Transformation TPS --FeatureExtraction ResNet \
        --SequenceModeling BiLSTM --Prediction Attn \
        --use_contrastive \
        --contrastive_embedding_dim 128 \
        --output_dir result_visualize/

    # Imagem única:
    python evaluate.py \
        --input demo_image/img_000002.jpg \
        --saved_model saved_models/TPS-ResNet-BiLSTM-Attn.pth \
        --Transformation TPS --FeatureExtraction ResNet \
        --SequenceModeling BiLSTM --Prediction Attn
"""

import os
import csv
import sys
import math
import string
import argparse
from pathlib import Path
from typing import Optional

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
from tqdm import tqdm

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import AlignCollate, RawDataset
from model import Model

# device é resolvido em __main__ a partir de --device e passado via opt.device

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
        #print(f"[dataset] {self.nSamples} imagem(ns) encontrada(s).")

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
# Leitura do ground truth
# ─────────────────────────────────────────────────────────────────────────────

def load_gt(input_path: str, sensitive: bool = False) -> dict:
    """Lê gt.txt da pasta de entrada e retorna {filename: label}.

    Formato esperado (tab-separado, uma linha por imagem):
        img_000002.jpg\tPPC5431

    Retorna dict vazio se gt.txt não for encontrado.
    """
    gt_path = Path(input_path) / 'gt.txt'
    if not gt_path.is_file():
        return {}

    gt = {}
    with open(gt_path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            fname, label = parts
            if not sensitive:
                label = label.lower()
            gt[fname.strip()] = label.strip()
    #print(f"[gt] {len(gt)} labels carregados de: {gt_path}")
    return gt


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento do modelo
# ─────────────────────────────────────────────────────────────────────────────

def load_model(opt):
    """Carrega modelo base ou contrastivo conforme opt.use_contrastive."""
    device = opt.device

    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt).to(device)

    model_type = 'Contrastivo' if getattr(opt, 'use_contrastive', False) else 'Base'
    #print(
    #    f"[model] Tipo={model_type}  "
    #    f"{opt.Transformation}-{opt.FeatureExtraction}-"
    #    f"{opt.SequenceModeling}-{opt.Prediction}  |  "
    #    f"classes={opt.num_class}  device={device}"
    #)
    #if getattr(opt, 'use_contrastive', False):
    #    print(
    #        f"[model] contrastive_embedding_dim={opt.contrastive_embedding_dim}  "
    #        f"margin={opt.contrastive_margin}  "
    #        f"lambda={opt.contrastive_lambda}  "
    #        f"mining={opt.contrastive_mining}"
    #    )

    # Os pesos foram salvos com DataParallel (prefixo "module.").
    # Para inferência em device único removemos o DataParallel e stripamos o prefixo.
    raw_sd = torch.load(opt.saved_model, map_location=device)
    if all(k.startswith('module.') for k in raw_sd):
        raw_sd = {k[len('module.'):]: v for k, v in raw_sd.items()}
    model.load_state_dict(raw_sd)
    model.eval()
    #print(f"[model] Pesos carregados de: {opt.saved_model}")
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
    model_type: str = 'Base',
    gt_text: Optional[str] = None,
    dpi: int = 150,
):
    """Salva uma figura Matplotlib com a imagem de entrada e o texto predito.

    Quando gt_text é fornecido, exibe ambos (GT e predição) e destaca o erro.
    """

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

    # — título: predição e GT (quando disponível) —
    display_pred = pred_text if pred_text else '(vazio)'
    if gt_text is not None:
        display_gt = gt_text if gt_text else '(vazio)'
        title = f'GT:   {display_gt}\nPred: {display_pred}'
        title_color = '#ff6b6b'   # vermelho → erro
        title_size = 13
    else:
        title = display_pred
        title_color = '#e0e0ff'
        title_size = 16

    fig.suptitle(
        title,
        fontsize=title_size,
        fontweight='bold',
        color=title_color,
        y=0.97,
        fontfamily='monospace',
        linespacing=1.5,
    )

    # — tipo de modelo (canto superior esquerdo, discreto) —
    tag_color = '#a855f7' if model_type == 'Contrastivo' else '#3b82f6'
    fig.text(
        0.04, 0.99,
        f'[{model_type}]',
        ha='left', va='top',
        fontsize=6, color=tag_color, fontweight='bold',
    )

    # — nome do arquivo (rodapé discreto) —
    fig.text(
        0.96, 0.01,
        Path(img_path).name,
        ha='right', va='bottom',
        fontsize=6, color='#555577',
    )

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Loop de inferência
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(opt):
    device = opt.device
    model, converter = load_model(opt)

    is_contrastive = getattr(opt, 'use_contrastive', False)
    model_type = 'Contrastivo' if is_contrastive else 'Base'

    # ── ground truth (opcional) ───────────────────────────────────────────────
    # Se a pasta de entrada contiver gt.txt, salva apenas as predições erradas.
    gt_map = {}
    if Path(opt.input).is_dir():
        gt_map = load_gt(opt.input, sensitive=opt.sensitive)
    errors_only = bool(gt_map)   # modo erro-only ativo apenas quando GT disponível

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

    # ── CSV de resultados ────────────────────────────────────────────────────
    csv_suffix = 'results_contrastive.csv' if is_contrastive else 'results_base.csv'
    csv_path = output_dir.parent / csv_suffix
    csv_rows = []   # todos os resultados (corretos + erros); escrito de uma vez no final

    total = len(dataset)
    processed = 0
    correct = 0
    n_errors = 0

    mode_label = 'gt' if errors_only else 'livre'
    bar_desc = f'{model_type} [{mode_label}]'

    with torch.no_grad(), tqdm(
        total=total,
        desc=bar_desc,
        unit='img',
        dynamic_ncols=True,
    ) as pbar:
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
                if is_contrastive:
                    preds = model(images, text_for_pred, is_train=False, return_contrastive=False)
                else:
                    preds = model(images, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            # ── confiança ──────────────────────────────────────────────────
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for img_path, pred, pred_max_prob in zip(image_paths, preds_str, preds_max_prob):
                processed += 1

                # Pós-processamento Attn: cortar após token [s] e limitar comprimento
                if 'Attn' in opt.Prediction:
                    eos_idx = pred.find('[s]')
                    pred = pred[:eos_idx]
                    pred_max_prob = pred_max_prob[:eos_idx]

                if opt.max_label_len > 0:
                    pred = pred[:opt.max_label_len]
                    pred_max_prob = pred_max_prob[:opt.max_label_len]

                try:
                    confidence = float(pred_max_prob.cumprod(dim=0)[-1])
                except Exception:
                    confidence = 0.0

                short_name = Path(img_path).name

                # ── comparação com GT ──────────────────────────────────────
                gt_label = gt_map.get(short_name)
                is_correct = (gt_label is not None) and (pred == gt_label)

                # Acumula todos os resultados no CSV
                if gt_label is not None:
                    csv_rows.append({
                        'img':      short_name,
                        'img_path': img_path,
                        'label':    gt_label,
                        'pred':     pred,
                        'conf':     f'{confidence:.6f}',
                        'correct':  '1' if is_correct else '0',
                    })

                # Atualiza contadores e barra
                if errors_only and gt_label is None:
                    pbar.update(1)
                    continue

                if is_correct:
                    correct += 1
                else:
                    n_errors += 1

                pbar.update(1)
                if errors_only:
                    with_gt_so_far = correct + n_errors
                    acc = correct / with_gt_so_far if with_gt_so_far else 0.0
                    pbar.set_postfix(acc=f'{acc:.2%}', erros=n_errors)

    # ── escreve CSV ─────────────────────────────────────────────────────────
    if csv_rows:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f, fieldnames=['img', 'img_path', 'label', 'pred', 'conf', 'correct'])
            writer.writeheader()
            writer.writerows(csv_rows)
        n_err_csv = sum(1 for r in csv_rows if r['correct'] == '0')
        print(f"[csv] {len(csv_rows)} resultados ({n_err_csv} erros) → {csv_path}")

    if errors_only:
        with_gt = sum(1 for p in dataset.image_path_list if Path(p).name in gt_map)
        accuracy = correct / with_gt if with_gt else 0.0
        print(f"✓ {correct}/{with_gt} corretas ({accuracy:.2%})  |  {n_errors} erros")
    else:
        print(f"✓ {processed}/{total} processadas")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Inferência com visualização Matplotlib — modelos base e contrastivos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── entrada / saída ──────────────────────────────────────────────────────
    parser.add_argument('--input', default='',
                        help='Imagem única (.jpg/.png/…) ou pasta contendo imagens. '
                             'Mutuamente exclusivo com --dataset.')
    parser.add_argument('--dataset', type=str, default='',
                        choices=['cars', 'cars_motors'],
                        help='Subpasta do conjunto de testes em dataset/test/. '
                             'Opções: cars | cars_motors. '
                             'Quando fornecido, sobrepõe --input com dataset/test/<dataset>.')
    parser.add_argument('--output_dir', default='result_visualize/',
                        help='Pasta onde o CSV de resultados será salvo')
    parser.add_argument('--max_label_len', type=int, default=7,
                        help='Comprimento máximo da predição após truncamento por [s]. '
                             'Evita que o decoder Attn emita caracteres extras antes do EOS. '
                             'Use 0 para desativar. default=7')

    # ── modelo ───────────────────────────────────────────────────────────────
    parser.add_argument('--saved_model', default='',
                        help='Caminho completo para o arquivo .pth do modelo treinado. '
                             'Mutuamente exclusivo com --mlflow_run_id.')
    parser.add_argument('--mlflow_run_id', default='',
                        help='ID do experimento MLflow (ex.: 9593e0ac5c274caca533d8af140e9d5e). '
                             'O caminho do modelo é resolvido automaticamente a partir de '
                             '<raiz_projeto>/mlruns/1/<run_id>/artifacts/<mlflow_model>.')
    parser.add_argument('--mlflow_model', default='best_accuracy.pth',
                        help='Nome do arquivo .pth dentro da pasta de artefatos do run. '
                             'Usado apenas quando --mlflow_run_id é fornecido. '
                             'Opções comuns: best_accuracy.pth | best_norm_ED.pth')
    parser.add_argument('--device', type=str, default='',
                        help='Device PyTorch para carregar o modelo: '
                             'cpu | cuda | cuda:0 | cuda:1 | … '
                             '(padrão: cuda se disponível, senão cpu)')

    # ── DataLoader ───────────────────────────────────────────────────────────
    parser.add_argument('--bsize', type=int, default=512,
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

    # ── Contrastivo / Triplet Loss (espelha train.py) ────────────────────────
    parser.add_argument('--use_contrastive', action='store_true',
                        help='Indicar que o modelo foi treinado com perda contrastiva '
                             '(Triplet Loss nos hidden states do decoder de atenção). '
                             'Requer --Prediction Attn.')
    parser.add_argument('--contrastive_margin', type=float, default=0.5,
                        help='Margem do Triplet Loss (distância cosseno). default=0.5')
    parser.add_argument('--contrastive_lambda', type=float, default=0.1,
                        help='Peso da perda contrastiva (usado apenas para log). default=0.1')
    parser.add_argument('--contrastive_mining', type=str, default='semihard',
                        choices=['semihard', 'hard', 'all'],
                        help='Estratégia de mineração de trinças (usado apenas para log). default=semihard')
    parser.add_argument('--contrastive_embedding_dim', type=int, default=128,
                        help='Dimensão do embedding contrastivo (CharContrastiveHead). default=128')
    parser.add_argument('--contrastive_warmup', type=int, default=0,
                        help='Iterações de warm-up (informativo apenas). default=0')

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    opt = parse_args()

    # ── resolução do dataset (--dataset sobrepõe --input) ─────────────────────
    if opt.dataset and opt.input:
        print("[erro] Passe apenas um de --dataset ou --input, não ambos.")
        sys.exit(1)

    if opt.dataset:
        project_root = Path(__file__).resolve().parent
        opt.input = str(project_root / 'dataset' / 'test' / opt.dataset)
        print(f"\n\n[dataset] Usando subpasta: {opt.input}")
    elif not opt.input:
        print("[erro] Fornecer --input <caminho> ou --dataset <cars|cars_motors>.")
        sys.exit(1)

    # ── resolução do caminho do modelo ─────────────────────────────────────────
    if opt.mlflow_run_id and opt.saved_model:
        print("[erro] Passe apenas um de --mlflow_run_id ou --saved_model, não ambos.")
        sys.exit(1)

    if opt.mlflow_run_id:
        project_root = Path(__file__).resolve().parent
        model_path = project_root / 'mlruns' / '1' / opt.mlflow_run_id / 'artifacts' / opt.mlflow_model
        if not model_path.is_file():
            print(f"[erro] Modelo não encontrado: {model_path}")
            print(f"       Verifique o run_id e o nome do arquivo (--mlflow_model).")
            sys.exit(1)
        opt.saved_model = str(model_path)
        print(f"[mlflow] Run ID : {opt.mlflow_run_id}")
        print(f"[mlflow] Modelo : {opt.mlflow_model}")
        print(f"[mlflow] Caminho: {opt.saved_model}")
    elif not opt.saved_model:
        print("[erro] Fornecer --saved_model <caminho> ou --mlflow_run_id <id>.")
        sys.exit(1)

    if opt.sensitive:
        opt.character = string.printable[:-6]   # 94 caracteres (mesmo que ASTER)

    if getattr(opt, 'use_contrastive', False) and 'Attn' not in opt.Prediction:
        print("[aviso] --use_contrastive requer --Prediction Attn. "
              "O flag será ignorado.")
        opt.use_contrastive = False

    # ── resolução do device ──────────────────────────────────────────────────
    if opt.device:
        opt.device = torch.device(opt.device)
    else:
        opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[device] Usando: {opt.device}")

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    run_inference(opt)
