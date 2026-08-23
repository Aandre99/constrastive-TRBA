"""
evaluate_mlflow.py  –  Avaliação Comparativa de Modelos MLflow
==============================================================
Executa a inferência de múltiplos modelos (salvos em runs do MLflow),
calcula as métricas de avaliação e gera:
  1. Gráficos de barras comparando os modelos (salvos em PNG).
  2. Um relatório Markdown com tabelas comparativas por métrica.
  3. Análise de acertos por tipo de veículo (car / motorcycle) usando
     dataset/vehicle_mapping.csv.

Exemplo de uso:
    python evaluate_mlflow.py \
        --mlflow_run_id RUN_ID_1 RUN_ID_2 RUN_ID_3 \
        --dataset cars_motors \
        --mlflow_model best_accuracy.pth \
        --output_dir evaluation_results
"""

import os
import sys
import math
import string
import argparse
import difflib
import csv
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import mlflow

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import AlignCollate, RawDataset
from model import Model
from evaluate import ImageListDataset, load_gt, load_model


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Carregamento do Vehicle Mapping
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_vehicle_mapping(mapping_path: Optional[str] = None) -> Dict[str, str]:
    """Carrega o mapeamento imagem → tipo de veículo a partir de um CSV.

    O CSV deve ter colunas: img, dataset, vehicle_type.
    A coluna 'img' contém o nome do arquivo sem extensão (stem).
    Retorna um dicionário {stem: vehicle_type}.
    """
    if mapping_path is None:
        mapping_path = os.path.join(_PROJECT_ROOT, 'dataset', 'vehicle_mapping.csv')

    mapping: Dict[str, str] = {}
    if not os.path.isfile(mapping_path):
        print(f"[aviso] Arquivo de mapeamento de veículos não encontrado: {mapping_path}")
        return mapping

    with open(mapping_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = row['img'].strip()
            vtype = row['vehicle_type'].strip().lower()
            mapping[stem] = vtype

    print(f"[mapping] Carregado vehicle_mapping.csv: {len(mapping)} entradas "
          f"({len(set(mapping.values()))} tipos de veículo)")
    return mapping


def levenshtein(a: str, b: str) -> int:
    """Calcula a distância de edição Levenshtein entre duas strings."""
    if a == b:
        return 0
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1,
                          prev[j] + 1,
                          prev[j - 1] + cost)
        prev = curr
    return prev[m]


def setup_opt_from_mlflow(opt, run_id: str) -> Tuple[object, mlflow.entities.Run]:
    """Carrega parâmetros da run do MLflow e localiza o checkpoint do modelo."""
    client = mlflow.tracking.MlflowClient()
    try:
        run = client.get_run(run_id)
    except Exception as e:
        print(f"[erro] Não foi possível obter a run MLflow '{run_id}': {e}")
        sys.exit(1)

    run_params = run.data.params
    print(f"[mlflow] Run ID          : {run_id}")
    print(f"[mlflow] Experimento ID  : {run.info.experiment_id}")
    print(f"[mlflow] Hiperparâmetros : {len(run_params)} parâmetros recuperados da run MLflow")

    def _get(key, default, cast_type=str):
        if key in run_params:
            val = run_params[key]
            if cast_type == bool:
                return str(val).lower() in ('true', '1', 'yes')
            return cast_type(val)
        return default

    # Cria cópia do opt para não afetar outros modelos
    import copy
    opt_copy = copy.deepcopy(opt)

    # Hiperparâmetros fundamentais do modelo
    opt_copy.Transformation = _get('Transformation', getattr(opt, 'Transformation', 'None'))
    opt_copy.FeatureExtraction = _get('FeatureExtraction', getattr(opt, 'FeatureExtraction', 'ResNet'))
    opt_copy.SequenceModeling = _get('SequenceModeling', getattr(opt, 'SequenceModeling', 'BiLSTM'))
    opt_copy.Prediction = _get('Prediction', getattr(opt, 'Prediction', 'Attn'))
    opt_copy.num_fiducial = _get('num_fiducial', getattr(opt, 'num_fiducial', 20), int)
    opt_copy.input_channel = _get('input_channel', getattr(opt, 'input_channel', 1), int)
    opt_copy.output_channel = _get('output_channel', getattr(opt, 'output_channel', 512), int)
    opt_copy.hidden_size = _get('hidden_size', getattr(opt, 'hidden_size', 256), int)
    opt_copy.attention_type = _get('attention_type', getattr(opt, 'attention_type', '1D'))
    opt_copy.use_contrastive = _get('use_contrastive', getattr(opt, 'use_contrastive', False), bool)
    opt_copy.contrastive_embedding_dim = _get('contrastive_embedding_dim', getattr(opt, 'contrastive_embedding_dim', 128), int)
    opt_copy.imgH = _get('imgH', getattr(opt, 'imgH', 64), int)
    opt_copy.imgW = _get('imgW', getattr(opt, 'imgW', 100), int)
    opt_copy.rgb = _get('rgb', getattr(opt, 'rgb', False), bool)
    opt_copy.character = _get('character', getattr(opt, 'character', '0123456789abcdefghijklmnopqrstuvwxyz'))
    opt_copy.sensitive = _get('sensitive', getattr(opt, 'sensitive', False), bool)
    opt_copy.PAD = _get('PAD', getattr(opt, 'PAD', False), bool)
    opt_copy.batch_max_length = _get('batch_max_length', getattr(opt, 'batch_max_length', 25), int)

    # Resolução do arquivo de modelo (.pth)
    model_path = None
    try:
        model_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path=opt.mlflow_model
        )
    except Exception:
        project_root = Path(_PROJECT_ROOT)
        candidate = project_root / 'mlruns' / run.info.experiment_id / run_id / 'artifacts' / opt.mlflow_model
        if candidate.is_file():
            model_path = str(candidate)
        else:
            mlruns_dir = project_root / 'mlruns'
            for exp_dir in mlruns_dir.iterdir():
                if exp_dir.is_dir():
                    c = exp_dir / run_id / 'artifacts' / opt.mlflow_model
                    if c.is_file():
                        model_path = str(c)
                        break

    if not model_path or not os.path.exists(model_path):
        print(f"[erro] Não foi possível localizar '{opt.mlflow_model}' para a run '{run_id}'.")
        sys.exit(1)

    opt_copy.saved_model = model_path
    print(f"[mlflow] Checkpoint       : {opt_copy.saved_model}")

    return opt_copy, run


def get_model_label(run: mlflow.entities.Run) -> str:
    """Gera um rótulo legível para o modelo a partir dos parâmetros da run."""
    params = run.data.params
    parts = []
    use_ctr = str(params.get('use_contrastive', 'false')).lower() in ('true', '1', 'yes')
    parts.append("CTR" if use_ctr else "BASE")

    transf = params.get('Transformation', 'None')
    if transf and transf != 'None':
        parts.append(transf)

    attn_type = params.get('attention_type', '1D')
    parts.append(attn_type)

    return " + ".join(parts)


def evaluate_model(opt_copy, run_id: str, vehicle_mapping: Optional[Dict[str, str]] = None) -> Dict:
    """Executa a inferência para um modelo e retorna as métricas.

    Se vehicle_mapping for fornecido, cada resultado será anotado com o
    tipo de veículo correspondente e métricas por tipo serão calculadas.
    """
    device = opt_copy.device
    model, converter = load_model(opt_copy)

    is_contrastive = getattr(opt_copy, 'use_contrastive', False)

    # Ground truth map
    gt_map = load_gt(opt_copy.input, sensitive=opt_copy.sensitive)
    if not gt_map:
        print(f"[erro] Nenhum ground truth (gt.txt) encontrado em: {opt_copy.input}")
        sys.exit(1)

    dataset = ImageListDataset(opt_copy.input, opt_copy)
    collate_fn = AlignCollate(imgH=opt_copy.imgH, imgW=opt_copy.imgW, keep_ratio_with_pad=opt_copy.PAD)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt_copy.bsize,
        shuffle=False,
        num_workers=opt_copy.workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == 'cuda'),
    )

    results = []
    total_gt_chars = 0
    gt_char_counts = defaultdict(int)

    with torch.no_grad(), tqdm(
        total=len(dataset),
        desc=f"Inferência [{run_id[:8]}]",
        unit="img",
        dynamic_ncols=True
    ) as pbar:
        for image_tensors, image_paths in loader:
            batch_size = image_tensors.size(0)
            images = image_tensors.to(device)

            length_for_pred = torch.IntTensor([opt_copy.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt_copy.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt_copy.Prediction:
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

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for img_path, pred, pred_max_prob in zip(image_paths, preds_str, preds_max_prob):
                if 'Attn' in opt_copy.Prediction:
                    eos_idx = pred.find('[s]')
                    pred = pred[:eos_idx]
                    pred_max_prob = pred_max_prob[:eos_idx]

                if opt_copy.max_label_len > 0:
                    pred = pred[:opt_copy.max_label_len]
                    pred_max_prob = pred_max_prob[:opt_copy.max_label_len]

                try:
                    conf = float(pred_max_prob.cumprod(dim=0)[-1])
                except Exception:
                    conf = 0.0

                short_name = Path(img_path).name
                gt_label = gt_map.get(short_name)

                if gt_label is not None:
                    is_correct = (pred == gt_label)
                    ed = levenshtein(pred, gt_label)

                    # ICDAR 2019 Normalized Edit Distance
                    if len(gt_label) == 0 or len(pred) == 0:
                        norm_ed = 0.0
                    elif len(gt_label) > len(pred):
                        norm_ed = 1.0 - (ed / float(len(gt_label)))
                    else:
                        norm_ed = 1.0 - (ed / float(len(pred)))

                    # Resolver tipo de veículo pelo stem do nome do arquivo
                    img_stem = Path(short_name).stem
                    vtype = vehicle_mapping.get(img_stem, 'unknown') if vehicle_mapping else 'unknown'

                    results.append({
                        'img': short_name,
                        'label': gt_label,
                        'pred': pred,
                        'conf': conf,
                        'correct': is_correct,
                        'edit_dist': ed,
                        'norm_ed': norm_ed,
                        'vehicle_type': vtype,
                    })

                    # Contagem de caracteres do GT
                    gt_label_cls = gt_label.upper() if not opt_copy.sensitive else gt_label
                    total_gt_chars += len(gt_label_cls)
                    for char in gt_label_cls:
                        gt_char_counts[char] += 1

                pbar.update(1)

    # ── Cálculo das Métricas ──────────────────────────────────────────────────
    total_samples = len(results)
    n_errors = sum(1 for r in results if not r['correct'])
    n_correct = total_samples - n_errors
    plate_accuracy = (n_correct / total_samples) if total_samples > 0 else 0.0
    total_edit_distance = sum(r['edit_dist'] for r in results)
    mean_edit_distance = total_edit_distance / total_samples if total_samples > 0 else 0.0
    mean_norm_ed = sum(r['norm_ed'] for r in results) / total_samples if total_samples > 0 else 0.0
    cer = (total_edit_distance / total_gt_chars) if total_gt_chars > 0 else 0.0

    # Contagem de erros por classe (utilizando SequenceMatcher)
    class_errors = defaultdict(int)
    for r in results:
        if not r['correct']:
            label_str = r['label'].upper() if not opt_copy.sensitive else r['label']
            pred_str = r['pred'].upper() if not opt_copy.sensitive else r['pred']
            matcher = difflib.SequenceMatcher(None, label_str, pred_str)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag in ('replace', 'delete'):
                    for char in label_str[i1:i2]:
                        class_errors[char] += 1

    # Definir conjunto de caracteres
    if opt_copy.sensitive:
        charset = sorted(list(set(opt_copy.character) | set(gt_char_counts.keys())))
    else:
        charset = sorted(list(set(opt_copy.character.upper()) | set(gt_char_counts.keys())))

    # Taxas por classe
    class_error_rate_total = {}
    class_error_rate_class = {}
    for c in charset:
        err_count = class_errors[c]
        class_error_rate_total[c] = (err_count / total_gt_chars) if total_gt_chars > 0 else 0.0
        class_error_rate_class[c] = (err_count / gt_char_counts[c]) if gt_char_counts[c] > 0 else 0.0

    print(f"\n{'=' * 65}")
    print(f" RESULTADOS  |  Run ID: {run_id}")
    print(f"{'=' * 65}")
    print(f" Total de Placas (GT)    : {total_samples}")
    print(f" Number of Errors        : {n_errors}")
    print(f" Plate Accuracy          : {plate_accuracy:.4%} ({n_correct}/{total_samples})")
    print(f" Total Edition Distance  : {total_edit_distance}")
    print(f" Mean Edition Distance   : {mean_edit_distance:.4f}")
    print(f" CER (Char Error Rate)   : {cer:.4%} ({total_edit_distance}/{total_gt_chars} chars)")
    print(f" Normalized Edit Dist    : {mean_norm_ed:.4f}")
    print(f"{'=' * 65}")

    # ── Métricas por Tipo de Veículo ─────────────────────────────────────────
    vehicle_type_metrics: Dict[str, Dict] = {}
    vehicle_types_found = sorted(set(r['vehicle_type'] for r in results))
    for vtype in vehicle_types_found:
        vt_results = [r for r in results if r['vehicle_type'] == vtype]
        vt_total = len(vt_results)
        vt_correct = sum(1 for r in vt_results if r['correct'])
        vt_errors = vt_total - vt_correct
        vt_accuracy = vt_correct / vt_total if vt_total > 0 else 0.0
        vt_total_ed = sum(r['edit_dist'] for r in vt_results)
        vt_mean_ed = vt_total_ed / vt_total if vt_total > 0 else 0.0
        vt_mean_ned = sum(r['norm_ed'] for r in vt_results) / vt_total if vt_total > 0 else 0.0
        vt_gt_chars = sum(len(r['label']) for r in vt_results)
        vt_cer = vt_total_ed / vt_gt_chars if vt_gt_chars > 0 else 0.0

        vehicle_type_metrics[vtype] = {
            'total': vt_total,
            'correct': vt_correct,
            'errors': vt_errors,
            'accuracy': vt_accuracy,
            'total_edit_distance': vt_total_ed,
            'mean_edit_distance': vt_mean_ed,
            'mean_norm_ed': vt_mean_ned,
            'total_gt_chars': vt_gt_chars,
            'cer': vt_cer,
        }
        print(f"  [{vtype:>12}] Accuracy: {vt_accuracy:.4%} ({vt_correct}/{vt_total})  "
              f"CER: {vt_cer:.4%}  MED: {vt_mean_ed:.4f}")

    return {
        'run_id': run_id,
        'total_samples': total_samples,
        'n_errors': n_errors,
        'n_correct': n_correct,
        'plate_accuracy': plate_accuracy,
        'total_edit_distance': total_edit_distance,
        'mean_edit_distance': mean_edit_distance,
        'mean_norm_ed': mean_norm_ed,
        'cer': cer,
        'total_gt_chars': total_gt_chars,
        'class_errors': dict(class_errors),
        'class_error_rate_total': class_error_rate_total,
        'class_error_rate_class': class_error_rate_class,
        'charset': charset,
        'gt_char_counts': dict(gt_char_counts),
        'results': results,
        'vehicle_type_metrics': vehicle_type_metrics,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Geração de Plots
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Paleta de cores profissional
_COLORS = [
    '#4C72B0', '#DD8452', '#55A868', '#C44E52',
    '#8172B3', '#937860', '#DA8BC3', '#8C8C8C',
    '#CCB974', '#64B5CD', '#725CA5', '#E07B39',
]


def _apply_style(ax, title: str, ylabel: str):
    """Aplica estilo visual consistente aos gráficos."""
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=9, rotation=30)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')


def plot_global_metrics(all_metrics: List[Dict], labels: List[str], output_dir: str):
    """Gera gráficos de barras das métricas globais."""
    metric_defs = [
        ('plate_accuracy', 'Plate Accuracy', 'Acurácia', True),
        ('cer', 'CER (Character Error Rate)', 'Taxa de Erro', True),
        ('mean_edit_distance', 'Mean Edition Distance', 'Distância Média', False),
        ('mean_norm_ed', 'Normalized Edit Distance', 'NED', False),
        ('n_errors', 'Number of Errors', 'Qtd. Erros', False),
        ('total_edit_distance', 'Total Edition Distance', 'Distância Total', False),
    ]

    n_metrics = len(metric_defs)
    n_cols = 3
    n_rows = math.ceil(n_metrics / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5.5 * n_rows))
    axes = axes.flatten()

    n_models = len(labels)
    x = np.arange(n_models)
    bar_width = max(0.25, 0.8 / max(n_models, 1))

    for idx, (key, title, ylabel, is_pct) in enumerate(metric_defs):
        ax = axes[idx]
        values = [m[key] for m in all_metrics]
        colors = [_COLORS[i % len(_COLORS)] for i in range(n_models)]
        bars = ax.bar(x, values, width=bar_width, color=colors, edgecolor='white', linewidth=0.8)

        # Valor acima de cada barra
        for bar, val in zip(bars, values):
            text = f"{val:.2%}" if is_pct else (f"{val:.3f}" if isinstance(val, float) else str(int(val)))
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    text, ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, ha='right')
        if is_pct:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        _apply_style(ax, title, ylabel)

    # Ocultar eixos extras
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Comparação de Métricas Globais entre Modelos',
                 fontsize=18, fontweight='bold', y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, 'global_metrics.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[plot] Salvo: {path}")


def plot_class_errors(all_metrics: List[Dict], labels: List[str], output_dir: str):
    """Gera gráfico de barras agrupadas de erros por classe de caractere."""
    # Charset unificado
    all_chars = set()
    for m in all_metrics:
        all_chars.update(m['charset'])
    charset = sorted(all_chars)

    if not charset:
        return

    n_models = len(labels)
    x = np.arange(len(charset))
    bar_width = 0.8 / max(n_models, 1)

    # ── Gráfico 1: Contagem absoluta de erros por classe ──
    fig, ax = plt.subplots(figsize=(max(14, len(charset) * 0.6), 6))
    for i, (m, label) in enumerate(zip(all_metrics, labels)):
        vals = [m['class_errors'].get(c, 0) for c in charset]
        ax.bar(x + i * bar_width, vals, width=bar_width,
               label=label, color=_COLORS[i % len(_COLORS)],
               edgecolor='white', linewidth=0.5)

    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(charset, fontsize=9, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    _apply_style(ax, 'Erros por Classe de Caractere (Absoluto)', 'Quantidade de Erros')
    fig.tight_layout()
    path = os.path.join(output_dir, 'class_errors_absolute.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[plot] Salvo: {path}")

    # ── Gráfico 2: Taxa de erro por classe (relativa à classe) ──
    fig, ax = plt.subplots(figsize=(max(14, len(charset) * 0.6), 6))
    for i, (m, label) in enumerate(zip(all_metrics, labels)):
        vals = [m['class_error_rate_class'].get(c, 0.0) for c in charset]
        ax.bar(x + i * bar_width, vals, width=bar_width,
               label=label, color=_COLORS[i % len(_COLORS)],
               edgecolor='white', linewidth=0.5)

    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(charset, fontsize=9, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    _apply_style(ax, 'Taxa de Erro por Classe (Relativa à Classe)', 'Taxa de Erro')
    fig.tight_layout()
    path = os.path.join(output_dir, 'class_error_rate_by_class.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[plot] Salvo: {path}")

    # ── Gráfico 3: Taxa de erro por classe (relativa ao total) ──
    fig, ax = plt.subplots(figsize=(max(14, len(charset) * 0.6), 6))
    for i, (m, label) in enumerate(zip(all_metrics, labels)):
        vals = [m['class_error_rate_total'].get(c, 0.0) for c in charset]
        ax.bar(x + i * bar_width, vals, width=bar_width,
               label=label, color=_COLORS[i % len(_COLORS)],
               edgecolor='white', linewidth=0.5)

    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels(charset, fontsize=9, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    _apply_style(ax, 'Taxa de Erro por Classe (Relativa ao Total de Caracteres)', 'Taxa de Erro')
    fig.tight_layout()
    path = os.path.join(output_dir, 'class_error_rate_by_total.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[plot] Salvo: {path}")


def plot_vehicle_type_accuracy(all_metrics: List[Dict], labels: List[str], output_dir: str):
    """Gera gráficos de barras agrupadas de acurácia e CER por tipo de veículo."""
    # Reunir todos os tipos de veículo encontrados em todos os modelos
    all_vtypes = set()
    for m in all_metrics:
        all_vtypes.update(m.get('vehicle_type_metrics', {}).keys())
    # Remover 'unknown' se houver tipos reais
    vtypes = sorted(all_vtypes - {'unknown'}) or sorted(all_vtypes)
    if not vtypes:
        return

    n_models = len(labels)
    x = np.arange(len(vtypes))
    bar_width = 0.8 / max(n_models, 1)

    # ── Gráfico 1: Plate Accuracy por Tipo de Veículo ──
    fig, ax = plt.subplots(figsize=(max(8, len(vtypes) * 2.5), 6))
    for i, (m, label) in enumerate(zip(all_metrics, labels)):
        vtm = m.get('vehicle_type_metrics', {})
        vals = [vtm.get(vt, {}).get('accuracy', 0.0) for vt in vtypes]
        bars = ax.bar(x + i * bar_width, vals, width=bar_width,
                      label=label, color=_COLORS[i % len(_COLORS)],
                      edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2%}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels([vt.capitalize() for vt in vtypes], fontsize=11, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    _apply_style(ax, 'Plate Accuracy por Tipo de Veículo', 'Acurácia')
    fig.tight_layout()
    path = os.path.join(output_dir, 'vehicle_type_accuracy.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[plot] Salvo: {path}")

    # ── Gráfico 2: CER por Tipo de Veículo ──
    fig, ax = plt.subplots(figsize=(max(8, len(vtypes) * 2.5), 6))
    for i, (m, label) in enumerate(zip(all_metrics, labels)):
        vtm = m.get('vehicle_type_metrics', {})
        vals = [vtm.get(vt, {}).get('cer', 0.0) for vt in vtypes]
        bars = ax.bar(x + i * bar_width, vals, width=bar_width,
                      label=label, color=_COLORS[i % len(_COLORS)],
                      edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2%}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x + bar_width * (n_models - 1) / 2)
    ax.set_xticklabels([vt.capitalize() for vt in vtypes], fontsize=11, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    _apply_style(ax, 'CER por Tipo de Veículo', 'Taxa de Erro (CER)')
    fig.tight_layout()
    path = os.path.join(output_dir, 'vehicle_type_cer.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[plot] Salvo: {path}")

    # ── Gráfico 3: Contagem de Amostras por Tipo de Veículo ──
    fig, ax = plt.subplots(figsize=(max(8, len(vtypes) * 2.5), 6))
    # Usar apenas o primeiro modelo como referência (mesmas imagens)
    vtm_ref = all_metrics[0].get('vehicle_type_metrics', {})
    totals = [vtm_ref.get(vt, {}).get('total', 0) for vt in vtypes]
    colors = [_COLORS[i % len(_COLORS)] for i in range(len(vtypes))]
    bars = ax.bar(x, totals, width=0.5, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([vt.capitalize() for vt in vtypes], fontsize=11, fontweight='bold')
    _apply_style(ax, 'Distribuição de Amostras por Tipo de Veículo', 'Quantidade')
    fig.tight_layout()
    path = os.path.join(output_dir, 'vehicle_type_distribution.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[plot] Salvo: {path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Geração do Relatório Markdown
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_markdown_report(all_metrics: List[Dict], labels: List[str],
                             run_ids: List[str], dataset: str, output_dir: str):
    """Gera o relatório Markdown comparativo."""
    lines = []
    lines.append("# 📊 Relatório Comparativo de Modelos\n")
    lines.append(f"**Dataset**: `{dataset}`  ")
    lines.append(f"**Data**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Modelos avaliados**: {len(labels)}\n")

    # ── Tabela: Informações dos Modelos ───────────────────────────────────────
    lines.append("---\n")
    lines.append("## 🏷️ Modelos Avaliados\n")
    lines.append("| # | Label | Run ID |")
    lines.append("|---|-------|--------|")
    for i, (label, rid) in enumerate(zip(labels, run_ids), 1):
        lines.append(f"| {i} | **{label}** | `{rid}` |")
    lines.append("")

    # ── Seção: Plate Accuracy ─────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## ✅ Plate Accuracy\n")
    lines.append("Proporção de placas reconhecidas corretamente (match exato).\n")
    lines.append("| Modelo | Acertos | Erros | Total | Accuracy |")
    lines.append("|--------|---------|-------|-------|----------|")
    for m, label in zip(all_metrics, labels):
        lines.append(f"| **{label}** | {m['n_correct']} | {m['n_errors']} "
                      f"| {m['total_samples']} | **{m['plate_accuracy']:.4%}** |")
    # Destacar melhor
    best_idx = max(range(len(all_metrics)), key=lambda i: all_metrics[i]['plate_accuracy'])
    lines.append(f"\n> 🏆 **Melhor modelo**: {labels[best_idx]} ({all_metrics[best_idx]['plate_accuracy']:.4%})\n")

    # ── Seção: CER ────────────────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## 📉 CER (Character Error Rate)\n")
    lines.append("Taxa de erro a nível de caractere (edit distance / total GT chars).\n")
    lines.append("| Modelo | Edit Distance Total | Total GT Chars | CER |")
    lines.append("|--------|---------------------|----------------|-----|")
    for m, label in zip(all_metrics, labels):
        lines.append(f"| **{label}** | {m['total_edit_distance']} "
                      f"| {m['total_gt_chars']} | **{m['cer']:.4%}** |")
    best_idx = min(range(len(all_metrics)), key=lambda i: all_metrics[i]['cer'])
    lines.append(f"\n> 🏆 **Melhor modelo**: {labels[best_idx]} (CER = {all_metrics[best_idx]['cer']:.4%})\n")

    # ── Seção: Edition Distance ───────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## 📏 Edition Distance\n")
    lines.append("Distância de edição Levenshtein entre predição e ground truth.\n")
    lines.append("| Modelo | Total | Média | Normalized ED |")
    lines.append("|--------|-------|-------|---------------|")
    for m, label in zip(all_metrics, labels):
        lines.append(f"| **{label}** | {m['total_edit_distance']} "
                      f"| {m['mean_edit_distance']:.4f} | {m['mean_norm_ed']:.4f} |")
    best_idx = min(range(len(all_metrics)), key=lambda i: all_metrics[i]['mean_edit_distance'])
    lines.append(f"\n> 🏆 **Melhor modelo**: {labels[best_idx]} "
                  f"(Média = {all_metrics[best_idx]['mean_edit_distance']:.4f})\n")

    # ── Seção: Number of Errors ───────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## ❌ Number of Errors\n")
    lines.append("Número de placas com pelo menos um caractere incorreto.\n")
    lines.append("| Modelo | Erros | Total | Taxa de Erro |")
    lines.append("|--------|-------|-------|--------------|")
    for m, label in zip(all_metrics, labels):
        err_rate = m['n_errors'] / m['total_samples'] if m['total_samples'] > 0 else 0
        lines.append(f"| **{label}** | {m['n_errors']} "
                      f"| {m['total_samples']} | {err_rate:.4%} |")
    best_idx = min(range(len(all_metrics)), key=lambda i: all_metrics[i]['n_errors'])
    lines.append(f"\n> 🏆 **Melhor modelo**: {labels[best_idx]} ({all_metrics[best_idx]['n_errors']} erros)\n")

    # ── Seção: Erros por Classe ───────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## 🔤 Erros por Classe de Caractere\n")
    lines.append("Contagem absoluta de erros de reconhecimento por classe.\n")

    all_chars = set()
    for m in all_metrics:
        all_chars.update(m['charset'])
    charset = sorted(all_chars)

    # Tabela: erros absolutos
    header = "| Char | " + " | ".join(f"**{l}**" for l in labels) + " |"
    sep = "|------|" + "|".join(["------"] * len(labels)) + "|"
    lines.append(header)
    lines.append(sep)
    for c in charset:
        row = f"| **{c}** |"
        for m in all_metrics:
            row += f" {m['class_errors'].get(c, 0)} |"
        lines.append(row)
    lines.append("")

    # ── Seção: Taxa de Erro por Classe (relativa à classe) ────────────────────
    lines.append("---\n")
    lines.append("## 📊 Taxa de Erro por Classe (Relativa à Classe)\n")
    lines.append("Proporção de caracteres incorretos em relação ao total de ocorrências da classe no GT.\n")

    header = "| Char | " + " | ".join(f"**{l}**" for l in labels) + " |"
    sep = "|------|" + "|".join(["------"] * len(labels)) + "|"
    lines.append(header)
    lines.append(sep)
    for c in charset:
        row = f"| **{c}** |"
        for m in all_metrics:
            rate = m['class_error_rate_class'].get(c, 0.0)
            row += f" {rate:.2%} |"
        lines.append(row)
    lines.append("")

    # ── Seção: Taxa de Erro por Classe (relativa ao total) ────────────────────
    lines.append("---\n")
    lines.append("## 📊 Taxa de Erro por Classe (Relativa ao Total de Caracteres)\n")
    lines.append("Proporção de erros da classe em relação ao total de caracteres do GT.\n")

    header = "| Char | " + " | ".join(f"**{l}**" for l in labels) + " |"
    sep = "|------|" + "|".join(["------"] * len(labels)) + "|"
    lines.append(header)
    lines.append(sep)
    for c in charset:
        row = f"| **{c}** |"
        for m in all_metrics:
            rate = m['class_error_rate_total'].get(c, 0.0)
            row += f" {rate:.4%} |"
        lines.append(row)
    lines.append("")

    # ── Seção: Resumo Geral ───────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## 🏅 Resumo Geral\n")
    lines.append("| Métrica | Melhor Modelo | Valor |")
    lines.append("|---------|---------------|-------|")

    best_acc = max(range(len(all_metrics)), key=lambda i: all_metrics[i]['plate_accuracy'])
    best_cer = min(range(len(all_metrics)), key=lambda i: all_metrics[i]['cer'])
    best_med = min(range(len(all_metrics)), key=lambda i: all_metrics[i]['mean_edit_distance'])
    best_ned = max(range(len(all_metrics)), key=lambda i: all_metrics[i]['mean_norm_ed'])
    best_err = min(range(len(all_metrics)), key=lambda i: all_metrics[i]['n_errors'])

    lines.append(f"| Plate Accuracy | **{labels[best_acc]}** | {all_metrics[best_acc]['plate_accuracy']:.4%} |")
    lines.append(f"| CER | **{labels[best_cer]}** | {all_metrics[best_cer]['cer']:.4%} |")
    lines.append(f"| Mean Edit Distance | **{labels[best_med]}** | {all_metrics[best_med]['mean_edit_distance']:.4f} |")
    lines.append(f"| Normalized ED | **{labels[best_ned]}** | {all_metrics[best_ned]['mean_norm_ed']:.4f} |")
    lines.append(f"| N. Erros | **{labels[best_err]}** | {all_metrics[best_err]['n_errors']} |")
    lines.append("")

    # ── Seção: Análise por Tipo de Veículo ─────────────────────────────────────
    # Verificar se há dados de tipo de veículo
    has_vtype_data = any(m.get('vehicle_type_metrics') for m in all_metrics)
    if has_vtype_data:
        all_vtypes_set = set()
        for m in all_metrics:
            all_vtypes_set.update(m.get('vehicle_type_metrics', {}).keys())
        vtypes_sorted = sorted(all_vtypes_set - {'unknown'}) or sorted(all_vtypes_set)

        lines.append("---\n")
        lines.append("## 🚗 Análise por Tipo de Veículo\n")
        lines.append("Acurácia e métricas discriminadas por tipo de veículo "
                      "(mapeamento: `dataset/vehicle_mapping.csv`).\n")

        # Tabela: Plate Accuracy por tipo de veículo
        lines.append("### Plate Accuracy por Tipo de Veículo\n")
        header = "| Modelo |" + " | ".join(f"**{vt.capitalize()}**" for vt in vtypes_sorted) + " |"
        sep = "|--------|" + "|".join(["--------"] * len(vtypes_sorted)) + "|"
        lines.append(header)
        lines.append(sep)
        for m, label in zip(all_metrics, labels):
            vtm = m.get('vehicle_type_metrics', {})
            row = f"| **{label}** |"
            for vt in vtypes_sorted:
                vt_data = vtm.get(vt, {})
                acc = vt_data.get('accuracy', 0.0)
                correct = vt_data.get('correct', 0)
                total = vt_data.get('total', 0)
                row += f" {acc:.4%} ({correct}/{total}) |"
            lines.append(row)
        lines.append("")

        # Tabela: CER por tipo de veículo
        lines.append("### CER por Tipo de Veículo\n")
        header = "| Modelo |" + " | ".join(f"**{vt.capitalize()}**" for vt in vtypes_sorted) + " |"
        sep = "|--------|" + "|".join(["--------"] * len(vtypes_sorted)) + "|"
        lines.append(header)
        lines.append(sep)
        for m, label in zip(all_metrics, labels):
            vtm = m.get('vehicle_type_metrics', {})
            row = f"| **{label}** |"
            for vt in vtypes_sorted:
                vt_data = vtm.get(vt, {})
                cer_val = vt_data.get('cer', 0.0)
                row += f" {cer_val:.4%} |"
            lines.append(row)
        lines.append("")

        # Tabela: Detalhamento completo por tipo de veículo
        lines.append("### Detalhamento por Tipo de Veículo\n")
        for vt in vtypes_sorted:
            lines.append(f"#### {vt.capitalize()}\n")
            lines.append("| Modelo | Acertos | Erros | Total | Accuracy | CER | MED | NED |")
            lines.append("|--------|---------|-------|-------|----------|-----|-----|-----|")
            for m, label in zip(all_metrics, labels):
                vtm = m.get('vehicle_type_metrics', {})
                vt_data = vtm.get(vt, {})
                lines.append(
                    f"| **{label}** "
                    f"| {vt_data.get('correct', 0)} "
                    f"| {vt_data.get('errors', 0)} "
                    f"| {vt_data.get('total', 0)} "
                    f"| **{vt_data.get('accuracy', 0.0):.4%}** "
                    f"| {vt_data.get('cer', 0.0):.4%} "
                    f"| {vt_data.get('mean_edit_distance', 0.0):.4f} "
                    f"| {vt_data.get('mean_norm_ed', 0.0):.4f} |"
                )
            lines.append("")

        # Melhor modelo por tipo de veículo
        lines.append("### 🏆 Melhor Modelo por Tipo de Veículo\n")
        lines.append("| Tipo | Melhor Modelo | Accuracy |")
        lines.append("|------|---------------|----------|")
        for vt in vtypes_sorted:
            best_i = max(
                range(len(all_metrics)),
                key=lambda i: all_metrics[i].get('vehicle_type_metrics', {}).get(vt, {}).get('accuracy', 0.0)
            )
            best_acc = all_metrics[best_i].get('vehicle_type_metrics', {}).get(vt, {}).get('accuracy', 0.0)
            lines.append(f"| **{vt.capitalize()}** | **{labels[best_i]}** | {best_acc:.4%} |")
        lines.append("")

    # ── Imagens dos plots ─────────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## 📈 Gráficos\n")
    lines.append("### Métricas Globais\n")
    lines.append("![Métricas Globais](global_metrics.png)\n")
    lines.append("### Erros por Classe (Absoluto)\n")
    lines.append("![Erros por Classe](class_errors_absolute.png)\n")
    lines.append("### Taxa de Erro por Classe (Relativa à Classe)\n")
    lines.append("![Taxa de Erro por Classe](class_error_rate_by_class.png)\n")
    lines.append("### Taxa de Erro por Classe (Relativa ao Total)\n")
    lines.append("![Taxa de Erro Total](class_error_rate_by_total.png)\n")
    if has_vtype_data:
        lines.append("### Acurácia por Tipo de Veículo\n")
        lines.append("![Acurácia por Tipo de Veículo](vehicle_type_accuracy.png)\n")
        lines.append("### CER por Tipo de Veículo\n")
        lines.append("![CER por Tipo de Veículo](vehicle_type_cer.png)\n")
        lines.append("### Distribuição de Amostras por Tipo de Veículo\n")
        lines.append("![Distribuição por Tipo](vehicle_type_distribution.png)\n")

    # Salvar
    md_path = os.path.join(output_dir, 'comparison_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[report] Salvo: {md_path}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    parser = argparse.ArgumentParser(
        description="Avaliação comparativa de modelos MLflow com geração de plots e relatório",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--mlflow_run_id', type=str, nargs='+', required=True,
                        help='IDs das runs do MLflow a comparar.')
    parser.add_argument('--dataset', type=str, default='cars_motors',
                        choices=['cars', 'cars_motors', 'rodo', 'ufpr', 'rodo_ufpr'],
                        help='Subpasta do dataset de teste (em dataset/test/<dataset>).')
    parser.add_argument('--input', type=str, default='',
                        help='Caminho direto para pasta de teste (sobrepõe --dataset se informado).')
    parser.add_argument('--mlflow_model', type=str, default='best_accuracy.pth',
                        help='Nome do arquivo de pesos .pth salvo nos artefatos da run.')
    parser.add_argument('--device', type=str, default='',
                        help='Device PyTorch: cuda | cuda:0 | cpu.')
    parser.add_argument('--bsize', type=int, default=512,
                        help='Tamanho do batch para inferência.')
    parser.add_argument('--workers', type=int, default=4,
                        help='Número de trabalhadores no DataLoader.')
    parser.add_argument('--max_label_len', type=int, default=7,
                        help='Comprimento máximo do label predito.')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Diretório de saída para plots e relatório.')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_args()

    # Resolução da pasta de entrada
    if not opt.input and opt.dataset:
        project_root = Path(_PROJECT_ROOT)
        opt.input = str(project_root / 'dataset' / 'test' / opt.dataset)

    if not Path(opt.input).is_dir():
        print(f"[erro] Pasta de teste não encontrada: {opt.input}")
        sys.exit(1)

    # Configuração de device PyTorch
    if opt.device:
        opt.device = torch.device(opt.device)
    else:
        opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cudnn.benchmark = True
    cudnn.deterministic = True

    # Diretório de saída
    os.makedirs(opt.output_dir, exist_ok=True)

    # Avaliar todos os modelos
    all_metrics = []
    all_labels = []
    run_ids = opt.mlflow_run_id

    # Carregar mapeamento de tipos de veículos
    vehicle_mapping = load_vehicle_mapping()

    for run_id in run_ids:
        print(f"\n{'━' * 65}")
        print(f"  Avaliando modelo: {run_id}")
        print(f"{'━' * 65}")

        opt_copy, run = setup_opt_from_mlflow(opt, run_id)
        label = get_model_label(run)
        metrics = evaluate_model(opt_copy, run_id, vehicle_mapping=vehicle_mapping)

        all_metrics.append(metrics)
        all_labels.append(label)

    # Salvar CSV de predições por modelo
    print(f"\n{'━' * 65}")
    print("  Salvando CSVs de predições por modelo...")
    print(f"{'━' * 65}")
    csv_filenames = []
    for m, label, rid in zip(all_metrics, all_labels, run_ids):
        safe_label = label.lower().replace(' ', '_').replace('+', '').replace('/', '_').strip('_')
        safe_label = '_'.join(part for part in safe_label.split('_') if part)
        csv_name = f"predictions_{safe_label}_{rid[:8]}.csv"
        csv_path = os.path.join(opt.output_dir, csv_name)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['img', 'label', 'pred', 'correct', 'edit_dist', 'norm_ed', 'confidence', 'vehicle_type'])
            for r in m['results']:
                writer.writerow([
                    r['img'], r['label'], r['pred'],
                    r['correct'], r['edit_dist'],
                    f"{r['norm_ed']:.4f}", f"{r['conf']:.6f}",
                    r.get('vehicle_type', 'unknown'),
                ])
        csv_filenames.append(csv_name)
        print(f"[csv] Salvo: {csv_path} ({len(m['results'])} linhas)")

    # Gerar plots comparativos
    print(f"\n{'━' * 65}")
    print("  Gerando gráficos comparativos...")
    print(f"{'━' * 65}")
    plot_global_metrics(all_metrics, all_labels, opt.output_dir)
    plot_class_errors(all_metrics, all_labels, opt.output_dir)
    plot_vehicle_type_accuracy(all_metrics, all_labels, opt.output_dir)

    # Gerar relatório Markdown
    print(f"\n{'━' * 65}")
    print("  Gerando relatório Markdown...")
    print(f"{'━' * 65}")
    generate_markdown_report(all_metrics, all_labels, run_ids, opt.dataset, opt.output_dir)

    # Salvar métricas brutas em JSON para reutilização
    json_path = os.path.join(opt.output_dir, 'metrics.json')
    json_data = {
        'dataset': opt.dataset,
        'timestamp': datetime.now().isoformat(),
        'models': []
    }
    for m, label, rid in zip(all_metrics, all_labels, run_ids):
        json_data['models'].append({
            'label': label,
            'run_id': rid,
            'plate_accuracy': m['plate_accuracy'],
            'cer': m['cer'],
            'n_errors': m['n_errors'],
            'mean_edit_distance': m['mean_edit_distance'],
            'mean_norm_ed': m['mean_norm_ed'],
            'total_edit_distance': m['total_edit_distance'],
            'total_samples': m['total_samples'],
            'total_gt_chars': m['total_gt_chars'],
            'class_errors': m['class_errors'],
            'class_error_rate_total': m['class_error_rate_total'],
            'class_error_rate_class': m['class_error_rate_class'],
            'vehicle_type_metrics': m.get('vehicle_type_metrics', {}),
        })
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"[json] Salvo: {json_path}")

    print(f"\n{'━' * 65}")
    print(f"  ✅ Avaliação comparativa concluída!")
    print(f"  📁 Resultados em: {os.path.abspath(opt.output_dir)}/")
    print(f"     • global_metrics.png")
    print(f"     • class_errors_absolute.png")
    print(f"     • class_error_rate_by_class.png")
    print(f"     • class_error_rate_by_total.png")
    print(f"     • vehicle_type_accuracy.png")
    print(f"     • vehicle_type_cer.png")
    print(f"     • vehicle_type_distribution.png")
    print(f"     • comparison_report.md")
    print(f"     • metrics.json")
    for csv_name in csv_filenames:
        print(f"     • {csv_name}")
    print(f"{'━' * 65}")
