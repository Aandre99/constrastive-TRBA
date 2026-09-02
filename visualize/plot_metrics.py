"""
plot_metrics.py – Gera gráficos científicos a partir dos metrics.json usando SciencePlots.

Reproduz os mesmos gráficos do results/visualizer.html:
  1. Model Accuracy (por dataset)
  2. Errors per Class (por dataset)
  3. Accuracy per Vehicle Type (por dataset)
  4. Global CER (por dataset)

Uso:
    python visualize/plot_metrics.py                       # processa todos os datasets
    python visualize/plot_metrics.py --datasets rodo_ufpr  # apenas um dataset
    python visualize/plot_metrics.py --output figures/      # diretório de saída customizado
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as PathEffects
import numpy as np

# SciencePlots styles
import scienceplots  # noqa: F401 – register styles

# ── Constantes ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

# Mapeamento de nomes antigos → novos (mesma lógica do visualizer.html)
MODEL_NAME_MAP = [
    (re.compile(r"^BASE\+1D\+TPS$", re.I), "TRBA"),
    (re.compile(r"^CTR\+1D\+TPS$", re.I), "CTRBA"),
    (re.compile(r"^BASE\+2D\+TPS$", re.I), "TRB2A"),
    (re.compile(r"^CTR\+2D\+TPS$", re.I), "CTRB2A"),
    (re.compile(r"^BASE\b.*\bTPS\b.*\b1D\b", re.I), "TRBA"),
    (re.compile(r"^BASE\b.*\b1D\b.*\bTPS\b", re.I), "TRBA"),
    (re.compile(r"^CTR\b.*\bTPS\b.*\b1D\b", re.I), "CTRBA"),
    (re.compile(r"^CTR\b.*\b1D\b.*\bTPS\b", re.I), "CTRBA"),
    (re.compile(r"^BASE\b.*\bTPS\b.*\b2D\b", re.I), "TRB2A"),
    (re.compile(r"^BASE\b.*\b2D\b.*\bTPS\b", re.I), "TRB2A"),
    (re.compile(r"^CTR\b.*\bTPS\b.*\b2D\b", re.I), "CTRB2A"),
    (re.compile(r"^CTR\b.*\b2D\b.*\bTPS\b", re.I), "CTRB2A"),
]

ALLOWED_MODELS = {"TRBA", "CTRBA", "TRB2A", "CTRB2A"}

# Ordem canônica para exibição (melhor desempenho primeiro, tipicamente)
MODEL_ORDER = ["CTRB2A", "CTRBA", "TRB2A", "TRBA"]

# Paleta de cores consistente por modelo
MODEL_COLORS = {
    "CTRB2A": "#3B82F6",  # Blue
    "CTRBA": "#8B5CF6",   # Violet
    "TRB2A": "#F97316",   # Orange
    "TRBA": "#6B7280",    # Gray (baseline)
}

MODEL_HATCHES: dict = {}  # sem hatching

COLOR_GOOD = "#16a34a"   # verde – diferença favorável
COLOR_BAD  = "#dc2626"   # vermelho – diferença desfavorável
COLOR_REF  = "#6b7280"   # cinza – barra de referência (TRBA)


# ── Helpers ───────────────────────────────────────────────────────────────────


def normalize_label(label: str) -> str:
    """Normaliza o label de um modelo para o nome canônico."""
    if not label:
        return label
    if label in ALLOWED_MODELS:
        return label
    for pattern, canonical in MODEL_NAME_MAP:
        if pattern.search(label.strip()):
            return canonical
    return label


def load_metrics(json_path: Path) -> dict[str, Any]:
    """Carrega e normaliza um metrics.json."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Normaliza labels e filtra modelos permitidos
    for m in data.get("models", []):
        m["label"] = normalize_label(m["label"])
    data["models"] = [m for m in data["models"] if m["label"] in ALLOWED_MODELS]

    # Ordena por acurácia descendente (mesma lógica do HTML)
    data["models"].sort(key=lambda m: m["plate_accuracy"], reverse=True)

    return data


def ordered_models(models: list[dict]) -> list[dict]:
    """Retorna os modelos na ordem canônica definida em MODEL_ORDER."""
    label_to_model = {m["label"]: m for m in models}
    return [label_to_model[lbl] for lbl in MODEL_ORDER if lbl in label_to_model]


def _save_fig(fig: plt.Figure, output_dir: Path, filename: str, dpi: int = 300):
    """Salva a figura em PNG e PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    for ext in (".png", ".pdf"):
        path = output_dir / f"{stem}{ext}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] Saved: {output_dir / stem}.{{png,pdf}}")


def _annotate_bar(
    ax, bar, value: float, ref_value,
    higher_is_better: bool = True,
    value_fmt: str = "{:.2f}%",
    delta_fmt: str = "{:+.2f} p.p.",
    fontsize_value: int = 7,
    fontsize_delta: int = 5,
):
    """Escreve o valor absoluto acima da barra e o delta DENTRO da barra, com contorno."""
    cx = bar.get_x() + bar.get_width() / 2
    top = bar.get_height()
    
    if top > 50:
        outset = 0.25
        inset = 0.25
    else:
        outset = 0.02
        inset = 0.02

    # Linha 1: valor absoluto — acima do topo
    ax.text(
        cx, top + outset,
        value_fmt.format(value),
        ha="center", va="bottom",
        fontsize=fontsize_value, fontweight="bold",
        color="black",
    )

    # Linha 2: delta vs baseline (dentro da barra)
    if ref_value is None:
        return
    delta = value - ref_value
    if abs(delta) < 1e-9:
        ax.text(
            cx, top - inset,
            "Ref",
            ha="center", va="top",
            fontsize=fontsize_delta, color="black", style="italic", alpha=0.8,
        )
        return
    txt = ax.text(
        cx, top - inset,
        delta_fmt.format(delta),
        ha="center", va="top",
        fontsize=fontsize_delta, fontweight="bold", color="black",
    )
    # Adiciona um contorno branco
    #txt.set_path_effects([PathEffects.withStroke(linewidth=0.5, foreground='white')])


# ── Gráfico 1: Model Accuracy ────────────────────────────────────────────────


def plot_model_accuracy(
    data: dict, output_dir: Path, dataset_name: str
):
    """Gráfico de barras com a acurácia global de cada modelo."""
    models = ordered_models(data["models"])
    if not models:
        return

    with plt.style.context(["science", "no-latex", "grid"]):
        fig, ax = plt.subplots(figsize=(5, 4))

        labels = [m["label"] for m in models]
        accuracies = [m["plate_accuracy"] * 100 for m in models]
        colors = [MODEL_COLORS.get(lbl, "#999") for lbl in labels]

        # Valor de referência = TRBA
        trba_acc = next((m["plate_accuracy"] * 100 for m in models if m["label"] == "TRBA"), None)

        x = np.arange(len(labels))
        bars = ax.bar(x, accuracies, color=colors, edgecolor="black", linewidth=0.5, width=0.6)

        for bar, lbl, acc in zip(bars, labels, accuracies):
            ref = None if lbl == "TRBA" else trba_acc
            _annotate_bar(ax, bar, acc, ref, higher_is_better=True)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Model Accuracy")

        y_min = max(0, min(accuracies) - 5)
        ax.set_ylim(y_min, 101)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

        fig.tight_layout()
        _save_fig(fig, output_dir, f"{dataset_name}_model_accuracy")



# ── Gráfico 1b: Rodo/UFPR Split Accuracy ──────────────────────────────────────

import csv

def plot_rodo_ufpr_split_accuracy(
    data: dict, output_dir: Path, dataset_name: str, metrics_path: Path
):
    """Gera uma figura contendo dois subplots de acurácia: rodosol-alpr e ufpr-alpr."""
    if dataset_name != "rodo_ufpr":
        return

    models = ordered_models(data.get("models", []))
    if not models:
        return

    results_dir = metrics_path.parent
    rodosol_accs = []
    ufpr_accs = []
    
    labels = [m["label"] for m in models]
    colors = [MODEL_COLORS.get(lbl, "#999") for lbl in labels]
    
    for m in models:
        run_id = m["run_id"]
        csv_files = list(results_dir.glob(f"predictions_*{run_id[:8]}*.csv"))
        if not csv_files:
            print(f"  [WARN] Missing CSV for {m['label']} ({run_id})")
            rodosol_accs.append(0)
            ufpr_accs.append(0)
            continue
            
        csv_path = csv_files[0]
        rodo_correct = 0
        rodo_total = 0
        ufpr_correct = 0
        ufpr_total = 0
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img = row["img"]
                correct = (row["correct"] == "True")
                if img.startswith("img_"):
                    rodo_total += 1
                    if correct: rodo_correct += 1
                else:
                    ufpr_total += 1
                    if correct: ufpr_correct += 1
                    
        rodosol_accs.append((rodo_correct / rodo_total * 100) if rodo_total > 0 else 0)
        ufpr_accs.append((ufpr_correct / ufpr_total * 100) if ufpr_total > 0 else 0)

    with plt.style.context(["science", "no-latex", "grid"]):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        x = np.arange(len(labels))
        
        # Subplot 1: rodosol-alpr
        ax1 = axes[0]
        bars1 = ax1.bar(x, rodosol_accs, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
        trba_acc_rodo = next((acc for lbl, acc in zip(labels, rodosol_accs) if lbl == "TRBA"), None)
        for bar, lbl, acc in zip(bars1, labels, rodosol_accs):
            ref = None if lbl == "TRBA" else trba_acc_rodo
            _annotate_bar(ax1, bar, acc, ref, higher_is_better=True)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=8)
        ax1.set_title("RodoSol-ALPR", fontsize=12)
        ax1.set_ylabel("Accuracy (%)")
        
        # Subplot 2: ufpr-alpr
        ax2 = axes[1]
        bars2 = ax2.bar(x, ufpr_accs, color=colors, edgecolor="black", linewidth=0.5, width=0.6)
        trba_acc_ufpr = next((acc for lbl, acc in zip(labels, ufpr_accs) if lbl == "TRBA"), None)
        for bar, lbl, acc in zip(bars2, labels, ufpr_accs):
            ref = None if lbl == "TRBA" else trba_acc_ufpr
            _annotate_bar(ax2, bar, acc, ref, higher_is_better=True)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=8)
        ax2.set_title("UFPR-ALPR", fontsize=12)
        
        all_accs = rodosol_accs + ufpr_accs
        if all_accs:
            y_min = max(0, min(all_accs) - 5)
            axes[0].set_ylim(y_min, 101)
            axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
            
        fig.tight_layout()
        _save_fig(fig, output_dir, f"{dataset_name}_split_accuracy")

# ── Gráfico 2: Errors per Class ──────────────────────────────────────────────


def plot_errors_per_class(
    data: dict, output_dir: Path, dataset_name: str
):
    """Gráfico de barras agrupadas mostrando erros por classe de caractere."""
    models = ordered_models(data["models"])
    if not models:
        return

    # Coleta todas as classes
    all_classes = sorted(
        set(cls for m in models for cls in m.get("class_errors", {}).keys())
    )

    if not all_classes:
        return

    with plt.style.context(["science", "no-latex", "grid"]):
        fig, ax = plt.subplots(figsize=(10, 3.5))

        n_classes = len(all_classes)
        n_models = len(models)
        bar_width = 0.8 / n_models
        x = np.arange(n_classes)

        for i, m in enumerate(models):
            lbl = m["label"]
            errors = [m.get("class_errors", {}).get(cls, 0) for cls in all_classes]
            offset = (i - (n_models - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                errors,
                width=bar_width,
                label=lbl,
                color=MODEL_COLORS.get(lbl, "#999"),
                edgecolor="black",
                linewidth=0.3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(all_classes, fontsize=7)
        ax.set_ylabel("Number of Errors")
        ax.set_title(f"Errors per Character Class")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

        fig.tight_layout()
        _save_fig(fig, output_dir, f"{dataset_name}_errors_per_class")


# ── Gráfico 3: Accuracy per Vehicle Type ─────────────────────────────────────


def plot_vehicle_type_accuracy(
    data: dict, output_dir: Path, dataset_name: str
):
    """Gráfico de barras agrupadas: acurácia por tipo de veículo (car/motorcycle)."""
    models = ordered_models(data["models"])
    # Filtra modelos que possuem vehicle_type_metrics
    models = [m for m in models if m.get("vehicle_type_metrics")]
    if not models:
        return

    vehicle_types = ["car", "motorcycle"]
    vehicle_labels = ["Car", "Motorcycle"]

    # Referência TRBA por tipo de veículo
    trba_model = next((m for m in models if m["label"] == "TRBA"), None)
    trba_accs_vt = {
        vt: (trba_model["vehicle_type_metrics"].get(vt, {}).get("accuracy", 0) * 100
             if trba_model else None)
        for vt in vehicle_types
    }

    with plt.style.context(["science", "no-latex", "grid"]):
        fig, ax = plt.subplots(figsize=(5, 4))

        n_vehicles = len(vehicle_types)
        n_models = len(models)
        bar_width = 0.8 / n_models
        x = np.arange(n_vehicles)

        for i, m in enumerate(models):
            lbl = m["label"]
            accs = [
                m["vehicle_type_metrics"].get(vt, {}).get("accuracy", 0) * 100
                for vt in vehicle_types
            ]

            offset = (i - (n_models - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset, accs,
                width=bar_width, label=lbl,
                color=MODEL_COLORS.get(lbl, "#999"),
                edgecolor="black", linewidth=0.5,
            )

            for bar, vt, acc in zip(bars, vehicle_types, accs):
                ref = None if lbl == "TRBA" else trba_accs_vt.get(vt)
                _annotate_bar(ax, bar, acc, ref, higher_is_better=True,
                              fontsize_value=6, fontsize_delta=4)

        ax.set_xticks(x)
        ax.set_xticklabels(vehicle_labels, fontsize=9)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Accuracy per Vehicle Type")
        ax.legend(fontsize=7, loc="lower left", framealpha=0.9)

        y_min = max(0, min(
            m["vehicle_type_metrics"].get(vt, {}).get("accuracy", 0) * 100
            for m in models for vt in vehicle_types
        ) - 5)
        ax.set_ylim(y_min, 102)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

        fig.tight_layout()
        _save_fig(fig, output_dir, f"{dataset_name}_vehicle_type_accuracy")


# ── Gráfico 4: CER (Character Error Rate) ────────────────────────────────────


def plot_cer(
    data: dict, output_dir: Path, dataset_name: str
):
    """Gráfico de barras com o CER global de cada modelo."""
    models = ordered_models(data["models"])
    if not models:
        return

    with plt.style.context(["science", "no-latex", "grid"]):
        fig, ax = plt.subplots(figsize=(5, 4))

        labels = [m["label"] for m in models]
        cers = [m["cer"] * 100 for m in models]
        colors = [MODEL_COLORS.get(lbl, "#999") for lbl in labels]

        # Referência = TRBA
        trba_cer = next((m["cer"] * 100 for m in models if m["label"] == "TRBA"), None)

        x = np.arange(len(labels))
        bars = ax.bar(x, cers, color=colors, edgecolor="black", linewidth=0.5, width=0.6)

        for bar, lbl, cer in zip(bars, labels, cers):
            ref = None if lbl == "TRBA" else trba_cer
            _annotate_bar(ax, bar, cer, ref, higher_is_better=False)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("CER (%)")
        ax.set_title(f"Character Error Rate (CER)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

        fig.tight_layout()
        _save_fig(fig, output_dir, f"{dataset_name}_cer")


# ── Gráfico 5: CER per Vehicle Type ──────────────────────────────────────────


def plot_vehicle_type_cer(
    data: dict, output_dir: Path, dataset_name: str
):
    """Gráfico de barras agrupadas: CER por tipo de veículo (car/motorcycle)."""
    models = ordered_models(data["models"])
    models = [m for m in models if m.get("vehicle_type_metrics")]
    if not models:
        return

    vehicle_types = ["car", "motorcycle"]
    vehicle_labels = ["Car", "Motorcycle"]

    # Referência TRBA por tipo de veículo
    trba_model = next((m for m in models if m["label"] == "TRBA"), None)
    trba_cers_vt = {
        vt: (trba_model["vehicle_type_metrics"].get(vt, {}).get("cer", 0) * 100
             if trba_model else None)
        for vt in vehicle_types
    }

    with plt.style.context(["science", "no-latex", "grid"]):
        fig, ax = plt.subplots(figsize=(5, 4))

        n_vehicles = len(vehicle_types)
        n_models = len(models)
        bar_width = 0.8 / n_models
        x = np.arange(n_vehicles)

        for i, m in enumerate(models):
            lbl = m["label"]
            cers = [
                m["vehicle_type_metrics"].get(vt, {}).get("cer", 0) * 100
                for vt in vehicle_types
            ]

            offset = (i - (n_models - 1) / 2) * bar_width
            bars = ax.bar(
                x + offset, cers,
                width=bar_width, label=lbl,
                color=MODEL_COLORS.get(lbl, "#999"),
                edgecolor="black", linewidth=0.5,
            )

            for bar, vt, cer in zip(bars, vehicle_types, cers):
                ref = None if lbl == "TRBA" else trba_cers_vt.get(vt)
                _annotate_bar(ax, bar, cer, ref, higher_is_better=False,
                              fontsize_value=6, fontsize_delta=4)

        ax.set_xticks(x)
        ax.set_xticklabels(vehicle_labels, fontsize=9)
        ax.set_ylabel("CER (%)")
        ax.set_title(f"CER per Vehicle Type")
        ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

        fig.tight_layout()
        _save_fig(fig, output_dir, f"{dataset_name}_vehicle_type_cer")


# ── Gráfico 6: Class Error Rate (by class) ───────────────────────────────────


def plot_class_error_rate(
    data: dict, output_dir: Path, dataset_name: str
):
    """Gráfico de barras agrupadas: taxa de erro por classe (class_error_rate_class)."""
    models = ordered_models(data["models"])
    models = [m for m in models if m.get("class_error_rate_class")]
    if not models:
        return

    all_classes = sorted(
        set(cls for m in models for cls in m.get("class_error_rate_class", {}).keys())
    )

    if not all_classes:
        return

    with plt.style.context(["science", "no-latex", "grid"]):
        fig, ax = plt.subplots(figsize=(10, 3.5))

        n_classes = len(all_classes)
        n_models = len(models)
        bar_width = 0.8 / n_models
        x = np.arange(n_classes)

        for i, m in enumerate(models):
            lbl = m["label"]
            rates = [
                m.get("class_error_rate_class", {}).get(cls, 0) * 100
                for cls in all_classes
            ]
            offset = (i - (n_models - 1) / 2) * bar_width
            ax.bar(
                x + offset,
                rates,
                width=bar_width,
                label=lbl,
                color=MODEL_COLORS.get(lbl, "#999"),
                edgecolor="black",
                linewidth=0.3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(all_classes, fontsize=7)
        ax.set_ylabel("Error Rate (%)")
        ax.set_title(f"Class Error Rate (per class)")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

        fig.tight_layout()
        _save_fig(fig, output_dir, f"{dataset_name}_class_error_rate")


# ── Main ──────────────────────────────────────────────────────────────────────


def discover_datasets(results_dir: Path) -> list[tuple[str, Path]]:
    """Descobre subpastas contendo metrics.json."""
    datasets = []
    for child in sorted(results_dir.iterdir()):
        if child.is_dir():
            metrics_file = child / "metrics.json"
            if metrics_file.exists():
                datasets.append((child.name, metrics_file))
    return datasets


def process_dataset(
    dataset_name: str, metrics_path: Path, output_dir: Path
):
    """Processa um único dataset e gera todos os gráficos."""
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Source:  {metrics_path}")
    print(f"  Output:  {output_dir}")
    print(f"{'='*60}")

    data = load_metrics(metrics_path)

    if not data.get("models"):
        print("  [WARN] No allowed models found. Skipping.")
        return None

    model_labels = [m["label"] for m in data["models"]]
    print(f"  Models:  {', '.join(model_labels)}")

    plot_model_accuracy(data, output_dir, dataset_name)
    plot_cer(data, output_dir, dataset_name)
    plot_errors_per_class(data, output_dir, dataset_name)
    plot_class_error_rate(data, output_dir, dataset_name)
    plot_vehicle_type_accuracy(data, output_dir, dataset_name)
    plot_rodo_ufpr_split_accuracy(data, output_dir, dataset_name, metrics_path)
    plot_vehicle_type_cer(data, output_dir, dataset_name)
    return data


    return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate scientific plots from metrics.json files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset folder names to process (e.g., rodo_ufpr ufpr rodo). "
        "Default: all datasets found in results/.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for plots. Default: results/<dataset>/plots/",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Path to the results directory.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for rasterized outputs (PNG).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        sys.exit(f"Results directory not found: {results_dir}")

    all_datasets = discover_datasets(results_dir)
    if not all_datasets:
        sys.exit(f"No metrics.json found in subdirectories of: {results_dir}")

    # Filtra datasets se especificado
    # Filtra datasets: Apenas rodo_ufpr
    all_datasets = [(name, path) for name, path in all_datasets if name == "rodo_ufpr"]
    if not all_datasets:
        sys.exit("Dataset 'rodo_ufpr' not found in results directory.")

    print(f"Found {len(all_datasets)} dataset(s): {', '.join(n for n, _ in all_datasets)}")

    for dataset_name, metrics_path in all_datasets:
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = metrics_path.parent / "plots"

        process_dataset(dataset_name, metrics_path, output_dir)

    print(f"\n[DONE] All done! Generated plots for {len(all_datasets)} dataset(s).")


if __name__ == "__main__":
    main()
