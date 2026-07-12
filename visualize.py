"""
visualize.py  –  Geração de imagens de visualização a partir de CSV de avaliação
==================================================================================
Lê um ou mais CSVs gerados por evaluate.py e produz imagens PNG para cada
predição selecionada, usando a mesma renderização Matplotlib do script original.

A coluna `img_path` do CSV aponta diretamente para a imagem original, portanto
nenhum modelo precisa ser carregado — a visualização é independente da GPU.

Uso básico:
    # Apenas erros do modelo contrastivo (padrão):
    python visualize.py outputs/cars_motors/results_contrastive.csv

    # Todos (corretos + erros) do modelo base:
    python visualize.py outputs/cars_motors/results_base.csv \\
        --filter all --output_dir viz/base_all/

    # Até 200 erros aleatórios de ambos os CSVs:
    python visualize.py outputs/cars_motors/ \\
        --filter errors --limit 200 --shuffle

    # Sem imagens — apenas conta e imprime sumário:
    python visualize.py outputs/cars_motors/results_contrastive.csv --dry_run
"""

import argparse
import csv
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Renderização (espelho de evaluate.py — sem dependência de importar o módulo)
# ─────────────────────────────────────────────────────────────────────────────

def _confidence_bar_color(score: float) -> str:
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
    """Salva uma figura Matplotlib com a imagem de entrada e a predição."""
    try:
        orig = Image.open(img_path).convert('RGB')
    except Exception:
        orig = Image.new('RGB', (100, 32), color=(40, 40, 40))
    orig_np = np.asarray(orig)

    fig = plt.figure(figsize=(7, 3.2), facecolor='#1a1a2e')
    gs = gridspec.GridSpec(
        2, 1,
        height_ratios=[5, 1],
        hspace=0.08,
        left=0.04, right=0.96, top=0.82, bottom=0.04,
    )

    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(orig_np, aspect='auto')
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    for spine in ax_img.spines.values():
        spine.set_edgecolor('#4a4a6a')
        spine.set_linewidth(1.5)

    ax_bar = fig.add_subplot(gs[1])
    bar_color = _confidence_bar_color(confidence)
    ax_bar.barh(0, confidence, height=0.6, color=bar_color, alpha=0.85)
    ax_bar.barh(0, 1.0,        height=0.6, color='#2c2c4a', alpha=0.6, zorder=0)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.5, 0.5)
    ax_bar.axis('off')
    ax_bar.text(confidence + 0.02, 0, f'{confidence:.2%}',
                va='center', ha='left', color=bar_color, fontsize=8, fontweight='bold')
    ax_bar.text(-0.01, 0, 'conf',
                va='center', ha='right', color='#888888', fontsize=7)

    display_pred = pred_text if pred_text else '(vazio)'
    if gt_text is not None:
        display_gt = gt_text if gt_text else '(vazio)'
        title = f'GT:   {display_gt}\nPred: {display_pred}'
        title_color = '#ff6b6b'
        title_size  = 13
    else:
        title = display_pred
        title_color = '#e0e0ff'
        title_size  = 16

    fig.suptitle(title, fontsize=title_size, fontweight='bold',
                 color=title_color, y=0.97, fontfamily='monospace', linespacing=1.5)

    tag_color = '#a855f7' if model_type == 'Contrastivo' else '#3b82f6'
    fig.text(0.04, 0.99, f'[{model_type}]',
             ha='left', va='top', fontsize=6, color=tag_color, fontweight='bold')
    fig.text(0.96, 0.01, Path(img_path).name,
             ha='right', va='bottom', fontsize=6, color='#555577')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Leitura e filtragem do CSV
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(csv_path: Path, filter_mode: str) -> list:
    """Retorna lista de dicts lida do CSV, filtrada por filter_mode."""
    rows = []
    with open(csv_path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        if 'img_path' not in (reader.fieldnames or []):
            print(f"[erro] CSV sem coluna 'img_path': {csv_path}")
            print("       Regere o CSV com a versão atual de evaluate.py.")
            sys.exit(1)
        for row in reader:
            if filter_mode == 'errors'  and row['correct'] == '1':
                continue
            if filter_mode == 'correct' and row['correct'] == '0':
                continue
            rows.append(row)
    return rows


def find_csvs(path: Path) -> list:
    """Dado um arquivo ou diretório, retorna lista de CSVs a processar."""
    if path.is_file():
        return [path]
    csvs = sorted(path.glob('results_*.csv'))
    if not csvs:
        print(f"[erro] Nenhum CSV results_*.csv encontrado em: {path}")
        sys.exit(1)
    return csvs


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Geração de imagens de visualização a partir de CSV de avaliação',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('csv', metavar='CSV_OU_DIR',
                        help='Arquivo CSV gerado por evaluate.py, ou diretório '
                             'contendo um ou mais results_*.csv.')
    parser.add_argument('--output_dir', default='',
                        help='Pasta de destino das imagens PNG. '
                             'Padrão: <pasta_do_csv>/viz/')
    parser.add_argument('--filter', default='errors',
                        choices=['errors', 'correct', 'all'],
                        help='Quais predições visualizar: '
                             'errors (erros), correct (acertos), all (todos).')
    parser.add_argument('--limit', type=int, default=0,
                        help='Máximo de imagens a gerar. 0 = sem limite.')
    parser.add_argument('--shuffle', action='store_true',
                        help='Embaralha as linhas antes de aplicar --limit.')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI das imagens de saída.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Threads paralelas para salvar imagens.')
    parser.add_argument('--dry_run', action='store_true',
                        help='Apenas conta e exibe estatísticas, sem gerar imagens.')

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def process_csv(csv_path: Path, opt) -> int:
    """Processa um CSV e gera as imagens. Retorna número de imagens salvas."""
    # Detecta tipo de modelo pelo nome do arquivo
    model_type = 'Contrastivo' if 'contrastive' in csv_path.name else 'Base'

    rows = load_csv(csv_path, opt.filter)
    if not rows:
        print(f"[{csv_path.name}] Nenhuma linha após filtro '{opt.filter}'.")
        return 0

    if opt.shuffle:
        random.shuffle(rows)
    if opt.limit > 0:
        rows = rows[:opt.limit]

    # Diretório de saída
    if opt.output_dir:
        output_dir = Path(opt.output_dir)
    else:
        # Ex.: results_contrastive.csv → viz_contrastive/
        suffix = csv_path.stem.replace('results_', '')
        output_dir = csv_path.parent / f'viz_{suffix}'
    output_dir.mkdir(parents=True, exist_ok=True)

    n_errors  = sum(1 for r in rows if r['correct'] == '0')
    n_correct = len(rows) - n_errors
    print(f"\n[{model_type}] {csv_path.name}")
    print(f"  filtro: {opt.filter}  |  total selecionado: {len(rows)}"
          f"  (erros: {n_errors}, corretos: {n_correct})")
    print(f"  saída : {output_dir}/")

    if opt.dry_run:
        return 0

    # Monta tarefas
    tasks = []
    for row in rows:
        img_path = row['img_path']
        pred     = row['pred']
        conf     = float(row['conf'])
        gt       = row.get('label') if opt.filter != 'correct' else None
        is_err   = row['correct'] == '0'
        suffix_  = '_error' if is_err else '_ok'
        out_name = Path(img_path).stem + suffix_ + '.png'
        out_path = str(output_dir / out_name)
        tasks.append((img_path, pred, conf, out_path, model_type, gt))

    saved = 0
    save_errors = 0
    futures = []

    with ThreadPoolExecutor(max_workers=opt.workers) as executor:
        for (img_path, pred, conf, out_path, mt, gt) in tasks:
            futures.append(executor.submit(
                save_result_image, img_path, pred, conf, out_path, mt, gt, opt.dpi
            ))

        with tqdm(total=len(futures), desc=f'Salvando [{model_type}]',
                  unit='img', dynamic_ncols=True) as pbar:
            for f in as_completed(futures):
                exc = f.exception()
                if exc:
                    save_errors += 1
                else:
                    saved += 1
                pbar.update(1)

    if save_errors:
        print(f"[aviso] {save_errors} imagem(ns) não puderam ser salvas.")
    print(f"✓ {saved} imagem(ns) salva(s) em: {output_dir}/")
    return saved


def main():
    opt = parse_args()
    csv_input = Path(opt.csv)

    if not csv_input.exists():
        print(f"[erro] Caminho não encontrado: {csv_input}")
        sys.exit(1)

    csvs = find_csvs(csv_input)
    total_saved = 0
    for csv_path in csvs:
        total_saved += process_csv(csv_path, opt)

    if len(csvs) > 1:
        print(f"\n[total] {total_saved} imagem(ns) gerada(s) de {len(csvs)} CSV(s).")


if __name__ == '__main__':
    main()
