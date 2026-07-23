"""
evaluate_multiple.py – Execução repetida de avaliação e agregação estatística
==============================================================================
Executa run_inference() de evaluate.py N vezes com os mesmos argumentos e
computa média e desvio padrão das métricas produzidas em cada rodada:

    • accuracy           – acurácia por contagem exata (correct / total_with_gt)
    • mean_confidence    – média da confiança de todas as predições
    • n_correct          – número de predições corretas
    • n_total            – total de imagens com GT
    • n_errors           – número de erros

As N execuções usam o mesmo checkpoint e dataset; a variação natural vem de
qualquer estocasticidade residual (e.g. ordem dos workers, Dropout em eval, etc.).

Uso:
    python evaluate_multiple.py \
        --n_runs 5 \
        --dataset cars \
        --mlflow_run_id <id> \
        --Transformation TPS --FeatureExtraction ResNet \
        --SequenceModeling BiLSTM --Prediction Attn \
        --output_dir outputs/cars/base_multi

O resultado é impresso no terminal e salvo em <output_dir>/multi_stats.json e
<output_dir>/multi_stats.csv.
"""

import os
import sys
import csv
import string
import argparse
import copy
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Importa funções reutilizáveis de evaluate.py
from evaluate import run_inference, parse_args as _base_parse_args


# ─────────────────────────────────────────────────────────────────────────────
# Versão estendida de run_inference que retorna métricas em vez de só imprimir
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_with_metrics(opt) -> Dict[str, float]:
    """Executa inferência e retorna dicionário com métricas agregadas."""
    import csv as _csv
    from pathlib import Path as _Path

    device = opt.device
    from evaluate import load_model, ImageListDataset, load_gt, AlignCollate
    import torch
    import torch.utils.data
    import torch.nn.functional as F

    model, converter = load_model(opt)
    is_contrastive = getattr(opt, 'use_contrastive', False)

    gt_map = {}
    if _Path(opt.input).is_dir():
        gt_map = load_gt(opt.input, sensitive=opt.sensitive)

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

    correct = 0
    n_errors = 0
    confidence_list: List[float] = []

    with torch.no_grad():
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

                confidence_list.append(conf)
                short_name = _Path(img_path).name
                gt_label = gt_map.get(short_name)
                is_correct = (gt_label is not None) and (pred == gt_label)
                if gt_label is not None:
                    if is_correct:
                        correct += 1
                    else:
                        n_errors += 1

    n_total = correct + n_errors
    accuracy = correct / n_total if n_total > 0 else 0.0
    mean_conf = float(np.mean(confidence_list)) if confidence_list else 0.0

    return {
        'accuracy':        accuracy,
        'mean_confidence': mean_conf,
        'n_correct':       correct,
        'n_total':         n_total,
        'n_errors':        n_errors,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agregação estatística
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(runs: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Calcula média e desvio padrão para cada métrica."""
    keys = list(runs[0].keys())
    result = {}
    for k in keys:
        values = [r[k] for r in runs]
        result[k] = {
            'mean':   float(np.mean(values)),
            'std':    float(np.std(values, ddof=1) if len(values) > 1 else 0.0),
            'min':    float(np.min(values)),
            'max':    float(np.max(values)),
            'values': values,
        }
    return result


def print_summary(stats: Dict, n_runs: int, model_tag: str):
    """Imprime tabela resumo no terminal."""
    sep = '─' * 62
    print(f'\n{sep}')
    print(f'  Resumo de {n_runs} execuções  |  modelo: {model_tag}')
    print(sep)
    fmt_metric = '{:<22}  {:>10}  {:>10}  {:>10}  {:>10}'
    print(fmt_metric.format('Métrica', 'Média', 'Desvio Padrão', 'Mín', 'Máx'))
    print('─' * 62)
    for k, v in stats.items():
        if k in ('n_correct', 'n_total', 'n_errors'):
            fmt = '{:<22}  {:>10.1f}  {:>10.3f}  {:>10.0f}  {:>10.0f}'
        else:
            fmt = '{:<22}  {:>10.4f}  {:>10.6f}  {:>10.4f}  {:>10.4f}'
        print(fmt.format(k, v['mean'], v['std'], v['min'], v['max']))
    print(sep)


def _fmt(k: str, v: float) -> str:
    """Formata um valor de métrica para exibição."""
    if k in ('n_correct', 'n_total', 'n_errors'):
        return f'{v:.1f}'
    return f'{v:.4f}'


def save_markdown(stats: Dict, runs: List[Dict], output_dir: Path, n_runs: int, model_tag: str):
    """Gera relatório Markdown com tabela de resumo e tabela por execução."""
    from datetime import datetime
    keys = list(runs[0].keys())

    metric_labels = {
        'accuracy':        'Acurácia',
        'mean_confidence': 'Confiança Média',
        'n_correct':       'Corretas',
        'n_total':         'Total c/ GT',
        'n_errors':        'Erros',
    }

    lines = []
    lines.append(f'# Relatório de Avaliação Múltipla')
    lines.append('')
    lines.append(f'| | |')
    lines.append(f'|---|---|')
    lines.append(f'| **Modelo** | {model_tag} |')
    lines.append(f'| **Execuções** | {n_runs} |')
    lines.append(f'| **Gerado em** | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |')
    lines.append('')

    # ── Tabela de resumo estatístico ────────────────────────────────────────
    lines.append('## Resumo Estatístico')
    lines.append('')
    lines.append('| Métrica | Média | Desvio Padrão | Mín | Máx |')
    lines.append('|---|---:|---:|---:|---:|')
    for k in keys:
        v = stats[k]
        label = metric_labels.get(k, k)
        lines.append(
            f'| {label} '
            f'| {_fmt(k, v["mean"])} '
            f'| {_fmt(k, v["std"])} '
            f'| {_fmt(k, v["min"])} '
            f'| {_fmt(k, v["max"])} |'
        )
    lines.append('')

    # ── Tabela por execução ─────────────────────────────────────────────────
    lines.append('## Resultados por Execução')
    lines.append('')
    header_cols = ['Run'] + [metric_labels.get(k, k) for k in keys]
    lines.append('| ' + ' | '.join(header_cols) + ' |')
    lines.append('|' + '|'.join(['---'] + ['---:'] * len(keys)) + '|')
    for i, row in enumerate(runs, start=1):
        cols = [str(i)] + [_fmt(k, row[k]) for k in keys]
        lines.append('| ' + ' | '.join(cols) + ' |')
    # Linha de média
    mean_cols = ['**média**'] + [f'**{_fmt(k, stats[k]["mean"])}**' for k in keys]
    lines.append('| ' + ' | '.join(mean_cols) + ' |')
    # Linha de desvio padrão
    std_cols  = ['**std**']  + [f'**{_fmt(k, stats[k]["std"])}**'  for k in keys]
    lines.append('| ' + ' | '.join(std_cols)  + ' |')
    lines.append('')

    # ── Destaques ────────────────────────────────────────────────────────────
    best_run_idx = int(np.argmax([r['accuracy'] for r in runs])) + 1
    worst_run_idx = int(np.argmin([r['accuracy'] for r in runs])) + 1
    lines.append('## Destaques')
    lines.append('')
    lines.append(f'- 🏆 **Melhor execução:** Run {best_run_idx} '
                 f'(acurácia = {_fmt("accuracy", runs[best_run_idx - 1]["accuracy"])})')
    lines.append(f'- 📉 **Pior execução:** Run {worst_run_idx} '
                 f'(acurácia = {_fmt("accuracy", runs[worst_run_idx - 1]["accuracy"])})')
    lines.append(f'- 📊 **Intervalo de acurácia:** '
                 f'{_fmt("accuracy", stats["accuracy"]["min"])} – '
                 f'{_fmt("accuracy", stats["accuracy"]["max"])}')
    lines.append('')

    md_path = output_dir / 'multi_stats.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'[multi] MD   salvo em: {md_path}')


def save_results(stats: Dict, runs: List[Dict], output_dir: Path, n_runs: int, model_tag: str = ''):
    """Salva resultados em CSV e Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV por rodada
    csv_path = output_dir / 'multi_stats.csv'
    keys = list(runs[0].keys())
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['run'] + keys)
        writer.writeheader()
        for i, row in enumerate(runs, start=1):
            writer.writerow({'run': i, **row})
        # Linha de média e desvio padrão
        writer.writerow({'run': 'mean',  **{k: f"{stats[k]['mean']:.6f}"  for k in keys}})
        writer.writerow({'run': 'std',   **{k: f"{stats[k]['std']:.6f}"   for k in keys}})
    print(f'[multi] CSV  salvo em: {csv_path}')

    # Markdown formatado
    save_markdown(stats, runs, output_dir, n_runs, model_tag)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    """Parser que herda todos os args de evaluate.py e adiciona --n_runs."""
    # Reutiliza o parser base, mas precisa reformular porque parse_args() chama
    # parser.parse_args() internamente. Replicamos aqui a mesma lógica.
    parser = argparse.ArgumentParser(
        description='Executa avaliação N vezes e computa média/desvio padrão das métricas',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── parâmetro novo ───────────────────────────────────────────────────────
    parser.add_argument('--n_runs', type=int, default=5,
                        help='Número de execuções de avaliação a realizar')

    # ── entrada / saída (espelha evaluate.py) ────────────────────────────────
    parser.add_argument('--input', default='')
    parser.add_argument('--dataset', type=str, default='',
                        choices=['cars', 'cars_motors'])
    parser.add_argument('--output_dir', default='result_visualize_multi/')
    parser.add_argument('--max_label_len', type=int, default=7)

    # ── modelo ───────────────────────────────────────────────────────────────
    parser.add_argument('--saved_model', default='')
    parser.add_argument('--mlflow_run_id', default='')
    parser.add_argument('--mlflow_model', default='best_accuracy.pth')
    parser.add_argument('--device', type=str, default='')

    # ── DataLoader ───────────────────────────────────────────────────────────
    parser.add_argument('--bsize', type=int, default=512)
    parser.add_argument('--workers', type=int, default=4)

    # ── pré-processamento ────────────────────────────────────────────────────
    parser.add_argument('--batch_max_length', type=int, default=25)
    parser.add_argument('--imgH', type=int, default=32)
    parser.add_argument('--imgW', type=int, default=100)
    parser.add_argument('--rgb', action='store_true')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz')
    parser.add_argument('--sensitive', action='store_true')
    parser.add_argument('--PAD', action='store_true')

    # ── arquitetura ──────────────────────────────────────────────────────────
    parser.add_argument('--Transformation', type=str, required=True)
    parser.add_argument('--FeatureExtraction', type=str, required=True)
    parser.add_argument('--SequenceModeling', type=str, required=True)
    parser.add_argument('--Prediction', type=str, required=True)
    parser.add_argument('--num_fiducial', type=int, default=20)
    parser.add_argument('--input_channel', type=int, default=1)
    parser.add_argument('--output_channel', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=256)

    # ── Contrastivo / Triplet Loss ───────────────────────────────────────────
    parser.add_argument('--use_contrastive', action='store_true')
    parser.add_argument('--contrastive_margin', type=float, default=0.5)
    parser.add_argument('--contrastive_lambda', type=float, default=0.1)
    parser.add_argument('--contrastive_mining', type=str, default='semihard',
                        choices=['semihard', 'hard', 'all'])
    parser.add_argument('--contrastive_embedding_dim', type=int, default=128)
    parser.add_argument('--contrastive_warmup', type=int, default=0)

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    opt = parse_args()

    # ── resolução do dataset ──────────────────────────────────────────────────
    if opt.dataset and opt.input:
        print('[erro] Passe apenas um de --dataset ou --input, não ambos.')
        sys.exit(1)

    if opt.dataset:
        project_root = Path(__file__).resolve().parent
        opt.input = str(project_root / 'dataset' / 'test' / opt.dataset)
        print(f'[dataset] Usando subpasta: {opt.input}')
    elif not opt.input:
        print('[erro] Fornecer --input <caminho> ou --dataset <cars|cars_motors>.')
        sys.exit(1)

    # ── resolução do modelo ───────────────────────────────────────────────────
    if opt.mlflow_run_id and opt.saved_model:
        print('[erro] Passe apenas um de --mlflow_run_id ou --saved_model, não ambos.')
        sys.exit(1)

    if opt.mlflow_run_id:
        project_root = Path(__file__).resolve().parent
        model_path = (
            project_root / 'mlruns' / '1' / opt.mlflow_run_id
            / 'artifacts' / opt.mlflow_model
        )
        if not model_path.is_file():
            print(f'[erro] Modelo não encontrado: {model_path}')
            sys.exit(1)
        opt.saved_model = str(model_path)
#        print(f'[mlflow] Run ID : {opt.mlflow_run_id}')
#        print(f'[mlflow] Modelo : {opt.mlflow_model}')
#        print(f'[mlflow] Caminho: {opt.saved_model}')
    elif not opt.saved_model:
        print('[erro] Fornecer --saved_model <caminho> ou --mlflow_run_id <id>.')
        sys.exit(1)

    if opt.sensitive:
        opt.character = string.printable[:-6]

    if getattr(opt, 'use_contrastive', False) and 'Attn' not in opt.Prediction:
        print('[aviso] --use_contrastive requer --Prediction Attn. O flag será ignorado.')
        opt.use_contrastive = False

    if opt.device:
        opt.device = torch.device(opt.device)
    else:
        opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] Usando: {opt.device}')

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    # ── N execuções ───────────────────────────────────────────────────────────
    n_runs = opt.n_runs
    is_contrastive = getattr(opt, 'use_contrastive', False)
    model_tag = 'Contrastivo' if is_contrastive else 'Base'

    print(f'\n[multi] Iniciando {n_runs} execuções  |  modelo: {model_tag}')
    #print(f'[multi] Dataset : {opt.input}')
    #print(f'[multi] Checkpoint: {opt.saved_model}\n')

    from tqdm import tqdm

    runs_results: List[Dict[str, float]] = []
    with tqdm(
        total=n_runs,
        desc=f'Runs [{model_tag}]',
        unit='run',
        dynamic_ncols=True,
        leave=True,
    ) as pbar:
        for run_idx in range(1, n_runs + 1):
            pbar.set_description(f'Runs [{model_tag}] — execução {run_idx}/{n_runs}')
            metrics = run_inference_with_metrics(opt)
            runs_results.append(metrics)
            pbar.set_postfix(
                acc=f'{metrics["accuracy"]:.4f}',
                conf=f'{metrics["mean_confidence"]:.4f}',
                ok=f'{metrics["n_correct"]}/{metrics["n_total"]}',
            )
            pbar.update(1)

    # ── estatísticas ─────────────────────────────────────────────────────────
    stats = aggregate(runs_results)
    print_summary(stats, n_runs, model_tag)
    save_results(stats, runs_results, Path(opt.output_dir), n_runs, model_tag)
