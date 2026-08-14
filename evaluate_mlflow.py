"""
evaluate_mlflow.py  –  Inferência e Registro de Métricas no MLflow
==================================================================
Executa a inferência de um modelo salvo em um run_id do MLflow, calcula
as métricas de avaliação (Number of Errors, Plate Accuracy, Edition Distance,
CER, erros por classe e taxa de erro em relação ao total de caracteres) e
registra as métricas diretamente no experimento correspondente do MLflow.

Exemplo de uso:
    python evaluate_mlflow.py \
        --mlflow_run_id 684fc5e44b1340ccaff068a8244eff15 \
        --dataset cars_motors \
        --mlflow_model best_accuracy.pth
"""

import os
import sys
import math
import string
import argparse
import difflib
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import mlflow

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import AlignCollate, RawDataset
from model import Model
from evaluate import ImageListDataset, load_gt, load_model


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


def setup_opt_from_mlflow(opt) -> Tuple[object, mlflow.entities.Run]:
    """Carrega parâmetros da run do MLflow e localiza o checkpoint do modelo."""
    client = mlflow.tracking.MlflowClient()
    try:
        run = client.get_run(opt.mlflow_run_id)
    except Exception as e:
        print(f"[erro] Não foi possível obter a run MLflow '{opt.mlflow_run_id}': {e}")
        sys.exit(1)

    run_params = run.data.params
    print(f"[mlflow] Run ID          : {opt.mlflow_run_id}")
    print(f"[mlflow] Experimento ID  : {run.info.experiment_id}")
    print(f"[mlflow] Hiperparâmetros : {len(run_params)} parâmetros recuperados da run MLflow")

    def _get(key, default, cast_type=str):
        if key in run_params:
            val = run_params[key]
            if cast_type == bool:
                return str(val).lower() in ('true', '1', 'yes')
            return cast_type(val)
        return getattr(opt, key, default)

    # Hiperparâmetros fundamentais do modelo
    opt.Transformation = _get('Transformation', getattr(opt, 'Transformation', 'None'))
    opt.FeatureExtraction = _get('FeatureExtraction', getattr(opt, 'FeatureExtraction', 'ResNet'))
    opt.SequenceModeling = _get('SequenceModeling', getattr(opt, 'SequenceModeling', 'BiLSTM'))
    opt.Prediction = _get('Prediction', getattr(opt, 'Prediction', 'Attn'))
    opt.num_fiducial = _get('num_fiducial', getattr(opt, 'num_fiducial', 20), int)
    opt.input_channel = _get('input_channel', getattr(opt, 'input_channel', 1), int)
    opt.output_channel = _get('output_channel', getattr(opt, 'output_channel', 512), int)
    opt.hidden_size = _get('hidden_size', getattr(opt, 'hidden_size', 256), int)
    opt.attention_type = _get('attention_type', getattr(opt, 'attention_type', '1D'))
    opt.use_contrastive = _get('use_contrastive', getattr(opt, 'use_contrastive', False), bool)
    opt.contrastive_embedding_dim = _get('contrastive_embedding_dim', getattr(opt, 'contrastive_embedding_dim', 128), int)
    opt.imgH = _get('imgH', getattr(opt, 'imgH', 64), int)
    opt.imgW = _get('imgW', getattr(opt, 'imgW', 100), int)
    opt.rgb = _get('rgb', getattr(opt, 'rgb', False), bool)
    opt.character = _get('character', getattr(opt, 'character', '0123456789abcdefghijklmnopqrstuvwxyz'))
    opt.sensitive = _get('sensitive', getattr(opt, 'sensitive', False), bool)
    opt.PAD = _get('PAD', getattr(opt, 'PAD', False), bool)
    opt.batch_max_length = _get('batch_max_length', getattr(opt, 'batch_max_length', 25), int)

    # Resolução do arquivo de modelo (.pth)
    model_path = None
    try:
        model_path = mlflow.artifacts.download_artifacts(
            run_id=opt.mlflow_run_id, artifact_path=opt.mlflow_model
        )
    except Exception:
        project_root = Path(_PROJECT_ROOT)
        candidate = project_root / 'mlruns' / run.info.experiment_id / opt.mlflow_run_id / 'artifacts' / opt.mlflow_model
        if candidate.is_file():
            model_path = str(candidate)
        else:
            mlruns_dir = project_root / 'mlruns'
            for exp_dir in mlruns_dir.iterdir():
                if exp_dir.is_dir():
                    c = exp_dir / opt.mlflow_run_id / 'artifacts' / opt.mlflow_model
                    if c.is_file():
                        model_path = str(c)
                        break

    if not model_path or not os.path.exists(model_path):
        print(f"[erro] Não foi possível localizar '{opt.mlflow_model}' para a run '{opt.mlflow_run_id}'.")
        sys.exit(1)

    opt.saved_model = model_path
    print(f"[mlflow] Checkpoint       : {opt.saved_model}")
    return opt, run


def evaluate_and_log(opt, run: mlflow.entities.Run) -> Dict[str, float]:
    """Executa a inferência e registra métricas no MLflow."""
    device = opt.device
    model, converter = load_model(opt)

    is_contrastive = getattr(opt, 'use_contrastive', False)

    # Ground truth map
    gt_map = load_gt(opt.input, sensitive=opt.sensitive)
    if not gt_map:
        print(f"[erro] Nenhum ground truth (gt.txt) encontrado em: {opt.input}")
        sys.exit(1)

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

    results = []
    total_gt_chars = 0
    gt_char_counts = defaultdict(int)

    with torch.no_grad(), tqdm(
        total=len(dataset),
        desc=f"Incerência [{opt.mlflow_run_id[:8]}]",
        unit="img",
        dynamic_ncols=True
    ) as pbar:
        for image_tensors, image_paths in loader:
            batch_size = image_tensors.size(0)
            images = image_tensors.to(device)

            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

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

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            for img_path, pred, pred_max_prob in zip(image_paths, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    eos_idx = pred.find('[s]')
                    pred = pred[:eos_idx]
                    pred_max_prob = pred_max_prob[:eos_idx]

                if opt.max_label_len > 0:
                    pred = pred[:opt.max_label_len]
                    pred_max_prob = pred_max_prob[:opt.max_label_len]

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

                    results.append({
                        'img': short_name,
                        'label': gt_label,
                        'pred': pred,
                        'conf': conf,
                        'correct': is_correct,
                        'edit_dist': ed,
                        'norm_ed': norm_ed,
                    })

                    # Contagem de caracteres do GT
                    gt_label_cls = gt_label.upper() if not opt.sensitive else gt_label
                    total_gt_chars += len(gt_label_cls)
                    for char in gt_label_cls:
                        gt_char_counts[char] += 1

                pbar.update(1)

    # ── Cálculo das Métricas Solicitadas ─────────────────────────────────────
    total_samples = len(results)
    n_errors = sum(1 for r in results if not r['correct'])
    n_correct = total_samples - n_errors
    plate_accuracy = (n_correct / total_samples) if total_samples > 0 else 0.0
    total_edit_distance = sum(r['edit_dist'] for r in results)
    mean_edit_distance = total_edit_distance / total_samples if total_samples > 0 else 0.0
    mean_norm_ed = sum(r['norm_ed'] for r in results) / total_samples if total_samples > 0 else 0.0
    cer = (total_edit_distance / total_gt_chars) if total_gt_chars > 0 else 0.0

    # Contagem de erros por classe (utilizando SequenceMatcher no alinhamento de texto)
    class_errors = defaultdict(int)
    for r in results:
        if not r['correct']:
            label_str = r['label'].upper() if not opt.sensitive else r['label']
            pred_str = r['pred'].upper() if not opt.sensitive else r['pred']
            matcher = difflib.SequenceMatcher(None, label_str, pred_str)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag in ('replace', 'delete'):
                    for char in label_str[i1:i2]:
                        class_errors[char] += 1

    # Definir conjunto completo de caracteres a serem logados
    # Inclui todos os caracteres do conjunto de caracteres do modelo + GT
    if opt.sensitive:
        charset = sorted(list(set(opt.character) | set(gt_char_counts.keys())))
    else:
        charset = sorted(list(set(opt.character.upper()) | set(gt_char_counts.keys())))

    print("\n" + "=" * 65)
    print(f" RESULTADOS DA AVALIAÇÃO  |  MLflow Run ID: {opt.mlflow_run_id}")
    print("=" * 65)
    print(f" Total de Placas (GT)    : {total_samples}")
    print(f" Number of Errors        : {n_errors}")
    print(f" Plate Accuracy          : {plate_accuracy:.4%} ({n_correct}/{total_samples})")
    print(f" Total Edition Distance  : {total_edit_distance}")
    print(f" Mean Edition Distance   : {mean_edit_distance:.4f}")
    print(f" CER (Char Error Rate)   : {cer:.4%} ({total_edit_distance}/{total_gt_chars} chars)")
    print(f" Normalized Edit Dist    : {mean_norm_ed:.4f}")
    print("=" * 65)

    # Dicionário de métricas para o MLflow
    mlflow_metrics = {
        # Métricas Globais Padrão
        "Number of Errors": float(n_errors),
        "Plate Accuracy": float(plate_accuracy),
        "Edition Distance": float(total_edit_distance),
        "CER": float(cer),
        # Métricas com prefixo test/ para organização no MLflow UI
        "test/number_of_errors": float(n_errors),
        "test/plate_accuracy": float(plate_accuracy),
        "test/total_edition_distance": float(total_edit_distance),
        "test/mean_edition_distance": float(mean_edit_distance),
        "test/norm_ED": float(mean_norm_ed),
        "test/cer": float(cer),
        "test/total_samples": float(total_samples),
        "test/total_gt_chars": float(total_gt_chars),
    }

    # Log de erros por classe (garantindo 0 para classes sem erros)
    for c in charset:
        err_count = class_errors[c]  # defaultdict retorna 0 se não existir
        rate_total_chars = (err_count / total_gt_chars) if total_gt_chars > 0 else 0.0
        rate_class_chars = (err_count / gt_char_counts[c]) if gt_char_counts[c] > 0 else 0.0

        mlflow_metrics[f"test/class_errors/{c}"] = float(err_count)
        mlflow_metrics[f"test/class_error_rate_total/{c}"] = float(rate_total_chars)
        mlflow_metrics[f"test/class_error_rate_class/{c}"] = float(rate_class_chars)

    # Registra no MLflow na run existente
    print(f"[mlflow] Loggando {len(mlflow_metrics)} métricas na run {opt.mlflow_run_id}...")
    with mlflow.start_run(run_id=opt.mlflow_run_id):
        mlflow.log_metrics(mlflow_metrics)
    print(f"[mlflow] Métricas registradas com sucesso no MLflow!")

    return mlflow_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Avaliação com logging de métricas no MLflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--mlflow_run_id', type=str, required=True,
                        help='ID da run do MLflow para carregar o modelo e registrar as métricas.')
    parser.add_argument('--dataset', type=str, default='cars_motors',
                        choices=['cars', 'cars_motors'],
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
                        help='Comprimento máximo do label predito (default=7).')
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

    # Recupera parâmetros do MLflow e o checkpoint do modelo
    opt, run = setup_opt_from_mlflow(opt)

    # Executa inferência e loga métricas no MLflow
    evaluate_and_log(opt, run)
