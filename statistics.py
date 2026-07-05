"""
statistics.py  –  Comparação estatística entre modelo Base e Contrastivo
=========================================================================
Lê os CSVs gerados pelo evaluate.py a partir de uma pasta de resultados
e produz um relatório .md com:

  • Acurácia de predição global
  • Character Error Rate (CER)
  • Distribuição de erros por classe (label)
  • Top-10 classes mais confundidas

CSV esperado (gerado por evaluate.py):
    img, label, pred, conf, correct

Uso:
    python statistics.py results/
"""

import csv
import sys
import argparse
from pathlib import Path
from collections import defaultdict
import difflib


def parse_gt(gt_path: Path) -> tuple:
    """Retorna (total_samples, total_chars, char_counts) do arquivo gt.txt."""
    char_counts = defaultdict(int)
    if not gt_path.is_file():
        return 0, 0, char_counts
    total_samples = 0
    total_chars = 0
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) == 2:
                total_samples += 1
                label = parts[1].strip().upper()
                total_chars += len(label)
                for char in label:
                    char_counts[char] += 1
    return total_samples, total_chars, char_counts


# ─────────────────────────────────────────────────────────────────────────────
# Levenshtein (sem dependências externas)
# ─────────────────────────────────────────────────────────────────────────────

def levenshtein(a: str, b: str) -> int:
    """Distância de edição entre duas strings."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Leitura dos CSVs
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(csv_path: str) -> list:
    """Retorna lista de dicts com os campos do CSV de resultados."""
    rows = []
    with open(csv_path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            is_correct = row['correct'] == '1' if 'correct' in row else row['label'] == row['pred']
            rows.append({
                'img':     row['img'],
                'label':   row['label'],
                'pred':    row['pred'],
                'conf':    float(row['conf']),
                'correct': is_correct,
            })
    return rows


def discover_csvs(results_dir: Path) -> tuple:
    """
    Auto-descobre os CSVs de base e contrastivo dentro de results_dir.
    Retorna (base_path, contra_path) ou lança FileNotFoundError.
    """
    base_path    = results_dir / 'results_base.csv'
    contra_path  = results_dir / 'results_contrastive.csv'

    missing = [p for p in (base_path, contra_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f'CSV(s) não encontrado(s) em {results_dir}:\n' +
            '\n'.join(f'  {p}' for p in missing)
        )
    return base_path, contra_path


# ─────────────────────────────────────────────────────────────────────────────
# Métricas
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(rows: list, total_gt: int = 0, total_gt_chars: int = 0, gt_char_counts: dict = None) -> dict:
    """
    Computa todas as métricas a partir do CSV completo (corretos + erros).

    Colunas esperadas: img, label, pred, conf, correct (bool)
    """
    fallback_total = len(rows)
    fallback_chars = sum(len(r['label']) for r in rows)

    total = total_gt if total_gt > 0 else fallback_total
    
    n_errors = sum(1 for r in rows if not r['correct'])
    n_correct = total - n_errors
    accuracy  = n_correct / total if total else 0.0

    # ── CER ─────────────────────────────────────────────────────────────────
    total_chars = total_gt_chars if total_gt_chars > 0 else fallback_chars
    edit_sum    = sum(levenshtein(r['pred'], r['label']) for r in rows if not r['correct'])
    cer = edit_sum / total_chars if total_chars else 0.0

    # ── Distribuição por classe (por caractere) ───────────────────────────────
    if gt_char_counts:
        class_total = defaultdict(int, gt_char_counts)
    else:
        class_total = defaultdict(int)
        for r in rows:
            for char in r['label'].upper():
                class_total[char] += 1

    class_errors = defaultdict(int)
    for r in rows:
        if not r['correct']:
            label_up = r['label'].upper()
            pred_up = r['pred'].upper()
            matcher = difflib.SequenceMatcher(None, label_up, pred_up)
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag in ('replace', 'delete'):
                    for char in label_up[i1:i2]:
                        class_errors[char] += 1

    class_error_rate = {
        cls: (class_errors[cls] / class_total[cls]) if class_total.get(cls, 0) > 0 else 0.0
        for cls in set(list(class_total.keys()) + list(class_errors.keys()))
    }

    classes_with_errors = [c for c in class_total if class_errors[c] > 0]

    top10_by_rate = sorted(
        classes_with_errors,
        key=lambda c: class_error_rate[c],
        reverse=True,
    )[:10]

    top10_by_count = sorted(
        classes_with_errors,
        key=lambda c: class_errors[c],
        reverse=True,
    )[:10]

    return {
        'total':            total,
        'n_correct':        n_correct,
        'n_errors':         n_errors,
        'accuracy':         accuracy,
        'total_chars':      total_chars,
        'edit_sum':         edit_sum,
        'cer':              cer,
        'class_total':      dict(class_total),
        'class_errors':     dict(class_errors),
        'class_error_rate': class_error_rate,
        'top10_by_rate':    top10_by_rate,
        'top10_by_count':   top10_by_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Geração do relatório Markdown
# ─────────────────────────────────────────────────────────────────────────────

def _pct(val: float) -> str:
    return f'{val:.2%}'


def _delta(base_val: float, contra_val: float, lower_is_better: bool = False, is_pct: bool = False) -> str:
    diff = contra_val - base_val
    if abs(diff) < 1e-9:
        return '='
    better = diff < 0 if lower_is_better else diff > 0
    sign   = '+' if diff > 0 else ''
    arrow  = '▲' if better else '▼'
    if is_pct:
        return f'{arrow} {sign}{diff:.2%}'
    else:
        if isinstance(base_val, int) and isinstance(contra_val, int):
            return f'{arrow} {sign}{int(diff)}'
        return f'{arrow} {sign}{diff:.4f}'


def generate_report(base_stats: dict, contra_stats: dict, results_dir: Path) -> str:
    b = base_stats
    c = contra_stats

    lines = []
    a = lines.append

    a('# Comparação de Modelos: Base vs Contrastivo')
    a('')
    a('> Gerado automaticamente por `statistics.py`')
    a('')
    a(f'- **Pasta de resultados**: `{results_dir}`')
    a(f'- **Conjunto de teste**: {b["total"]} imagens')
    a('')
    a('---')
    a('')

    # ── 1. Acurácia ──────────────────────────────────────────────────────────
    a('## 1. Acurácia de Predição')
    a('')
    a('Predição exata: `label == pred`')
    a('')
    a('| Métrica | Base | Contrastivo | Δ (Contrastivo − Base) |')
    a('|---------|------|-------------|------------------------|')
    a(f'| Corretas | {b["n_correct"]}/{b["total"]} | {c["n_correct"]}/{c["total"]} | |')
    a(f'| Erros | {b["n_errors"]} | {c["n_errors"]} | {_delta(b["n_errors"], c["n_errors"], lower_is_better=True)} |')
    a(f'| **Acurácia** | **{_pct(b["accuracy"])}** | **{_pct(c["accuracy"])}** | **{_delta(b["accuracy"], c["accuracy"], is_pct=True)}** |')
    a('')

    # ── 2. CER ───────────────────────────────────────────────────────────────
    a('## 2. Character Error Rate (CER)')
    a('')
    a('`CER = Σ levenshtein(pred, label) / Σ len(label)` — calculado sobre todo o conjunto.')
    a('')
    a('| Métrica | Base | Contrastivo | Δ |')
    a('|---------|------|-------------|---|')
    a(f'| Total de caracteres (GT) | {b["total_chars"]} | {c["total_chars"]} | |')
    a(f'| Soma das distâncias de edição | {b["edit_sum"]} | {c["edit_sum"]} | {_delta(b["edit_sum"], c["edit_sum"], lower_is_better=True)} |')
    a(f'| **CER** | **{_pct(b["cer"])}** | **{_pct(c["cer"])}** | **{_delta(b["cer"], c["cer"], lower_is_better=True, is_pct=True)}** |')
    a('')

    # ── 3. Distribuição por classe ────────────────────────────────────────────
    a('## 3. Distribuição de Erros por Classe')
    a('')
    a('Classes com pelo menos 1 erro em qualquer modelo, ordenadas alfabeticamente (Letras -> Números).')
    a('')
    a('| Classe | Total | Erros Base | Erros Contra | Taxa Base | Taxa Contra | Δ Taxa |')
    a('|--------|-------|------------|--------------|-----------|-------------|--------|')

    all_error_classes = {cls for cls, errs in b['class_errors'].items() if errs > 0} | \
                        {cls for cls, errs in c['class_errors'].items() if errs > 0}
    for cls in sorted(all_error_classes, key=lambda x: (not x.isalpha(), x)):
        total_cls = b['class_total'].get(cls, c['class_total'].get(cls, 0))
        b_err     = b['class_errors'].get(cls, 0)
        c_err     = c['class_errors'].get(cls, 0)
        b_rate    = b['class_error_rate'].get(cls, 0.0)
        c_rate    = c['class_error_rate'].get(cls, 0.0)
        delta     = c_rate - b_rate
        sign      = '+' if delta > 0 else ''
        a(f'| `{cls}` | {total_cls} | {b_err} | {c_err} | {_pct(b_rate)} | {_pct(c_rate)} | {sign}{delta:.2%} |')
    a('')

    a('## 4. Top Classes Mais Confundidas')
    a('')
    a('| # | Classe | Erros Base | Erros Contra |')
    a('|---|--------|------------|--------------|')
    
    top_classes = set(b['top10_by_rate']) | set(c['top10_by_rate'])
    top_classes_sorted = sorted(top_classes, key=lambda x: (not x.isalpha(), x))
    
    for i, cls in enumerate(top_classes_sorted, 1):
        b_err = b['class_errors'].get(cls, 0)
        c_err = c['class_errors'].get(cls, 0)
        a(f'| {i} | `{cls}` | {b_err} | {c_err} |')
    a('')

    return '\n'.join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Gera relatório Markdown comparando modelos Base e Contrastivo.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('results_dir',
                        help='Pasta de resultados gerada pelo evaluate.py '
                             '(deve conter results_base.csv e results_contrastive.csv)')
    parser.add_argument('--gt_path', default='dataset/test/gt.txt',
                        help='Caminho para o arquivo gt.txt (padrão: dataset/test/gt.txt)')
    return parser.parse_args()


if __name__ == '__main__':
    opt = parse_args()

    results_dir = Path(opt.results_dir)
    if not results_dir.is_dir():
        print(f'[erro] Pasta não encontrada: {results_dir}')
        sys.exit(1)

    # ── descoberta dos CSVs ───────────────────────────────────────────────────
    try:
        base_path, contra_path = discover_csvs(results_dir)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    output_path = results_dir / 'comparison.md'

    # ── carrega dados ─────────────────────────────────────────────────────────
    print(f'[base]   {base_path}')
    base_rows = load_csv(str(base_path))
    print(f'         {len(base_rows)} amostras')

    print(f'[contra] {contra_path}')
    contra_rows = load_csv(str(contra_path))
    print(f'         {len(contra_rows)} amostras')

    gt_path = Path(opt.gt_path)
    total_gt, total_gt_chars, gt_char_counts = parse_gt(gt_path)
    if total_gt == 0:
        print(f"[aviso] gt.txt não encontrado ou vazio em {gt_path}. Usando apenas os dados do CSV.")

    # ── métricas ──────────────────────────────────────────────────────────────
    print('[stats]  Calculando...')
    base_stats   = compute_stats(base_rows, total_gt, total_gt_chars, gt_char_counts)
    contra_stats = compute_stats(contra_rows, total_gt, total_gt_chars, gt_char_counts)

    # ── relatório ─────────────────────────────────────────────────────────────
    report = generate_report(base_stats, contra_stats, results_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding='utf-8')
    print(f'[ok]     Relatório salvo em: {output_path}')

    # ── resumo no terminal ────────────────────────────────────────────────────
    print()
    print(f"{'─'*55}")
    print(f"{'Métrica':<25} {'Base':>10} {'Contrastivo':>14}")
    print(f"{'─'*55}")
    print(f"{'Acurácia':<25} {base_stats['accuracy']:>10.2%} {contra_stats['accuracy']:>14.2%}")
    print(f"{'CER':<25} {base_stats['cer']:>10.2%} {contra_stats['cer']:>14.2%}")
    print(f"{'Erros':<25} {base_stats['n_errors']:>10} {contra_stats['n_errors']:>14}")
    print(f"{'─'*55}")
