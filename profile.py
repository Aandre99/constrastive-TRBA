#!/usr/bin/env python3
"""
profile_mlflow_models.py
========================

Profiling comparativo das quatro variantes do experimento 2x2:

    TRBA   = 1D + sem contrastive
    CTRBA  = 1D + contrastive
    TRB2A  = 2D + sem contrastive
    CTRB2A = 2D + contrastive

O script:
  1. Recupera a configuração de cada run do MLflow.
  2. Baixa/carrega o checkpoint correspondente.
  3. Reconstrói exatamente o Model(opt) usado na run.
  4. Conta parâmetros totais e por componente.
  5. Calcula FLOPs de inferência para batch=1.
  6. Calcula o overhead da branch contrastiva separadamente.
  7. Mede tempo de inferência com CUDA quando disponível.
  8. Gera CSV, JSON e Markdown.

IMPORTANTE SOBRE FLOPs
----------------------
O profiler usa contagem analítica para Conv2d, Linear, LSTM/LSTMCell,
operações de atenção e operações matriciais principais. A convenção usada é:

    1 multiplicação + 1 soma = 2 FLOPs

Operações elementares (ReLU, sigmoid, softmax, dropout etc.) não são
incluídas na contagem principal. Isso é uma convenção comum para comparação
arquitetural e torna a comparação entre as quatro variantes consistente.

Para a atenção autoregressiva, o número de passos é batch_max_length + 1,
igual ao comportamento do Attention do projeto.

Uso:
    python profile_mlflow_models.py \
        --mlflow_run_id RUN_TRBA RUN_CTRBA RUN_TRB2A RUN_CTRB2A

Opcional:
    --device cuda
    --warmup 20
    --iterations 100
    --output_dir profiling_results
"""

import os
import sys
import json
import csv
import time
import copy
import argparse
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import mlflow

# -------------------------------------------------------------------------
# Projeto
# -------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import Model


# -------------------------------------------------------------------------
# MLflow / configuração
# -------------------------------------------------------------------------

def _cast(value, default, cast_type):
    if value is None:
        return default

    if cast_type is bool:
        return str(value).lower() in ("true", "1", "yes")

    try:
        return cast_type(value)
    except (TypeError, ValueError):
        return default


def get_run_and_opt(base_opt, run_id, mlflow_model):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params

    opt = copy.deepcopy(base_opt)

    def get(name, default, typ=str):
        return _cast(params.get(name), default, typ)

    # Mesmos campos recuperados pelo evaluate_mlflow_bootstrap.py
    opt.Transformation = get("Transformation", "None")
    opt.FeatureExtraction = get("FeatureExtraction", "ResNet")
    opt.SequenceModeling = get("SequenceModeling", "BiLSTM")
    opt.Prediction = get("Prediction", "Attn")

    opt.num_fiducial = get("num_fiducial", 20, int)
    opt.input_channel = get("input_channel", 1, int)
    opt.output_channel = get("output_channel", 512, int)
    opt.hidden_size = get("hidden_size", 256, int)

    opt.attention_type = get("attention_type", "1D")
    opt.use_contrastive = get("use_contrastive", False, bool)
    opt.contrastive_embedding_dim = get(
        "contrastive_embedding_dim", 128, int
    )

    opt.imgH = get("imgH", 64, int)
    opt.imgW = get("imgW", 100, int)
    opt.rgb = get("rgb", False, bool)
    opt.character = get(
        "character",
        "0123456789abcdefghijklmnopqrstuvwxyz",
        str,
    )
    opt.sensitive = get("sensitive", False, bool)
    opt.PAD = get("PAD", False, bool)
    opt.batch_max_length = get("batch_max_length", 25, int)

    opt.num_class = len(opt.character) + 2

    # Localiza checkpoint
    model_path = None

    try:
        model_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path=mlflow_model,
        )
    except Exception:
        pass

    if not model_path or not os.path.isfile(model_path):
        exp_id = run.info.experiment_id
        candidate = PROJECT_ROOT / "mlruns" / exp_id / run_id / "artifacts" / mlflow_model
        if candidate.is_file():
            model_path = str(candidate)

    if not model_path:
        mlruns_dir = PROJECT_ROOT / "mlruns"
        if mlruns_dir.is_dir():
            for exp_dir in mlruns_dir.iterdir():
                candidate = exp_dir / run_id / "artifacts" / mlflow_model
                if candidate.is_file():
                    model_path = str(candidate)
                    break

    if not model_path:
        raise FileNotFoundError(
            f"Checkpoint '{mlflow_model}' não encontrado para run {run_id}"
        )

    opt.saved_model = model_path

    return opt, run


def experiment_config(run):
    p = run.data.params
    attn = p.get("attention_type", "1D")
    ctr = str(p.get("use_contrastive", "false")).lower() in (
        "true", "1", "yes"
    )

    if attn == "1D" and not ctr:
        return "TRBA"
    if attn == "1D" and ctr:
        return "CTRBA"
    if attn == "2D" and not ctr:
        return "TRB2A"
    if attn == "2D" and ctr:
        return "CTRB2A"

    return f"{'CTR' if ctr else 'BASE'}-{attn}"


# -------------------------------------------------------------------------
# Checkpoint
# -------------------------------------------------------------------------

def load_checkpoint(model, checkpoint_path, device):
    # PyTorch >= 2.x aceita `weights_only`; versões antigas, como
    # PyTorch 1.3.1, não conhecem esse argumento. Mantemos compatibilidade
    # com ambos os ambientes.
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state = checkpoint["model"]
        else:
            state = checkpoint
    else:
        state = checkpoint

    # Remove DataParallel prefix if present
    cleaned = OrderedDict()
    for key, value in state.items():
        new_key = key[7:] if key.startswith("module.") else key
        cleaned[new_key] = value

    result = model.load_state_dict(cleaned, strict=False)

    if result.missing_keys:
        print("[checkpoint] Missing keys:", result.missing_keys)
    if result.unexpected_keys:
        print("[checkpoint] Unexpected keys:", result.unexpected_keys)


# -------------------------------------------------------------------------
# Parameter profiling
# -------------------------------------------------------------------------

def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def parameter_breakdown(model):
    result = OrderedDict()

    for name, module in model.named_children():
        result[name] = count_parameters(module)

    result["TOTAL"] = count_parameters(model)
    return result


# -------------------------------------------------------------------------
# FLOP counter
# -------------------------------------------------------------------------

class FlopCounter:
    """
    Contador analítico para as operações principais da arquitetura.

    Convenção:
        FLOPs = 2 * número de MACs

    São contabilizados:
      - Conv2d
      - Linear
      - LSTM
      - LSTMCell
      - operações de atenção BMM
      - projeções da atenção
      - TPS como parte de Conv/Linear; operações geométricas customizadas
        são estimadas separadamente quando possível.
    """

    def __init__(self):
        self.total = 0
        self.by_module = OrderedDict()
        self.handles = []

    def add(self, name, flops):
        self.total += int(flops)
        self.by_module[name] = self.by_module.get(name, 0) + int(flops)

    @staticmethod
    def _numel(x):
        if not isinstance(x, torch.Tensor):
            return 0
        return x.numel()

    def _hook(self, name, module, inputs, output):
        try:
            flops = self._estimate(module, inputs, output)
            if flops:
                self.add(name, flops)
        except Exception as exc:
            print(
                f"[flops] Aviso: não foi possível contar {name}: {exc}"
            )

    def _estimate(self, module, inputs, output):
        # -------------------------------------------------------------
        # Conv2d
        # -------------------------------------------------------------
        if isinstance(module, nn.Conv2d):
            x = inputs[0]
            if x.ndim != 4 or not isinstance(output, torch.Tensor):
                return 0

            batch = x.shape[0]
            out_h = output.shape[-2]
            out_w = output.shape[-1]

            kernel = module.kernel_size[0] * module.kernel_size[1]
            cin = module.in_channels
            groups = module.groups
            cout = module.out_channels

            macs = (
                batch
                * out_h
                * out_w
                * cout
                * (cin // groups)
                * kernel
            )
            return 2 * macs

        # -------------------------------------------------------------
        # Linear
        # -------------------------------------------------------------
        if isinstance(module, nn.Linear):
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return 0

            batch_positions = x.numel() // x.shape[-1]
            macs = batch_positions * module.in_features * module.out_features
            return 2 * macs

        # -------------------------------------------------------------
        # LSTMCell
        #
        # PyTorch LSTMCell:
        #   4 gates * (input_size + hidden_size) * hidden_size
        #
        # Cada multiply-add = 2 FLOPs.
        # -------------------------------------------------------------
        if isinstance(module, nn.LSTMCell):
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return 0

            batch = x.shape[0]
            hidden = module.hidden_size
            inp = module.input_size

            # input-hidden + hidden-hidden, 4 gates
            macs = batch * 4 * hidden * (inp + hidden)
            return 2 * macs

        # -------------------------------------------------------------
        # LSTM
        #
        # A camada do projeto é batch_first=True e bidirectional=True.
        # -------------------------------------------------------------
        if isinstance(module, nn.LSTM):
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return 0

            batch, seq_len, inp = x.shape
            hidden = module.hidden_size
            directions = 2 if module.bidirectional else 1

            layers = module.num_layers

            total = 0

            layer_input = inp

            for _ in range(layers):
                macs_per_step = (
                    directions
                    * 4
                    * hidden
                    * (layer_input + hidden)
                )
                total += batch * seq_len * macs_per_step

                layer_input = hidden * directions

            return 2 * total

        return 0

    def attach(self, model):
        for name, module in model.named_modules():
            # Só registrar folhas para evitar dupla contagem.
            if len(list(module.children())) == 0:
                handle = module.register_forward_hook(
                    lambda m, i, o, n=name: self._hook(n, m, i, o)
                )
                self.handles.append(handle)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


# -------------------------------------------------------------------------
# Atenção: operações que não aparecem como módulos PyTorch contáveis
# -------------------------------------------------------------------------

def estimate_attention_flops(
    attention_module,
    batch_size,
    sequence_length,
    input_size,
    hidden_size,
    num_classes,
    steps,
):
    """
    Conta explicitamente operações da célula de atenção que são feitas
    com operações tensoriais (projeção, score, BMM).

    A projeção i2h, h2h e score são nn.Linear e já são contabilizadas
    pelos hooks. Aqui contamos apenas as BMM principais que não são
    representadas por módulos:
        context = alpha^T @ batch_H

    A operação alpha^T @ batch_H:
        [B, 1, S] @ [B, S, C] -> [B, 1, C]
    """
    # BMM: S*C MACs por batch e por step.
    bmm_macs = batch_size * steps * sequence_length * input_size

    # Não adicionar Linear aqui: elas já estão no contador.
    return 2 * bmm_macs


# -------------------------------------------------------------------------
# Profiling
# -------------------------------------------------------------------------

def make_dummy_inputs(opt, device):
    batch_size = 1

    channels = 3 if getattr(opt, "rgb", False) else 1

    images = torch.randn(
        batch_size,
        channels,
        opt.imgH,
        opt.imgW,
        device=device,
    )

    text = torch.zeros(
        batch_size,
        opt.batch_max_length + 1,
        dtype=torch.long,
        device=device,
    )

    return images, text


def profile_model(model, opt, device, warmup=10, iterations=50):
    model.eval()

    images, text = make_dummy_inputs(opt, device)

    # -------------------------------------------------------------
    # Parameter count
    # -------------------------------------------------------------
    params_total = count_parameters(model)
    params_breakdown = parameter_breakdown(model)

    # -------------------------------------------------------------
    # FLOPs
    # -------------------------------------------------------------
    counter = FlopCounter()
    counter.attach(model)

    with torch.no_grad():
        # Warmup / graph construction
        for _ in range(2):
            model(
                images,
                text,
                is_train=False,
                return_contrastive=False,
            )

        # Uma execução para contar hooks.
        counter.total = 0
        counter.by_module.clear()

        model(
            images,
            text,
            is_train=False,
            return_contrastive=False,
        )

    counter.remove()

    base_flops = counter.total

    # -------------------------------------------------------------
    # Atenção BMM
    # -------------------------------------------------------------
    attention = getattr(model, "Prediction", None)

    attention_extra = 0

    if attention is not None:
        seq_length = None

        # Derive feature sequence length sem depender de valores
        # hard-coded.
        with torch.no_grad():
            if opt.Transformation != "None":
                x = model.Transformation(images)
            else:
                x = images

            visual = model.FeatureExtraction(x)

            if opt.attention_type == "2D":
                visual = model.pos_encoding(visual)
                _, _, h, w = visual.shape
                seq_length = h * w
            else:
                visual = model.AdaptiveAvgPool(
                    visual.permute(0, 3, 1, 2)
                )
                visual = visual.squeeze(3)
                seq_length = visual.shape[1]

            input_size = visual.shape[-1]

        attention_extra = estimate_attention_flops(
            attention,
            batch_size=1,
            sequence_length=seq_length,
            input_size=input_size,
            hidden_size=opt.hidden_size,
            num_classes=opt.num_class,
            steps=opt.batch_max_length + 1,
        )

    inference_flops = base_flops + attention_extra

    # -------------------------------------------------------------
    # Contrastive branch
    # -------------------------------------------------------------
    contrastive_params = 0
    contrastive_flops = 0

    if getattr(opt, "use_contrastive", False):
        head = getattr(model, "contrastive_head", None)

        if head is not None:
            contrastive_params = count_parameters(head)

            # Precisamos de N = batch_max_length válido por imagem.
            # No pior caso utilizado pelo profiling, consideramos todos
            # os passos de saída como válidos.
            n = opt.batch_max_length + 1

            hidden = opt.hidden_size
            emb_dim = opt.contrastive_embedding_dim

            # Dois Linear:
            # hidden -> hidden
            # hidden -> embedding_dim
            contrastive_flops += 2 * n * hidden * hidden
            contrastive_flops += 2 * n * hidden * emb_dim

            # F.normalize: aproximadamente 2 operações por elemento
            contrastive_flops += 2 * n * emb_dim

    # -------------------------------------------------------------
    # Timing
    # -------------------------------------------------------------
    if device.type == "cuda":
        torch.cuda.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            model(
                images,
                text,
                is_train=False,
                return_contrastive=False,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(iterations):
            model(
                images,
                text,
                is_train=False,
                return_contrastive=False,
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    latency_ms = (elapsed / iterations) * 1000.0

    return {
        "params_total": params_total,
        "params_breakdown": params_breakdown,
        "contrastive_params": contrastive_params,

        "inference_flops": inference_flops,
        "inference_gflops": inference_flops / 1e9,

        "contrastive_flops": contrastive_flops,
        "contrastive_gflops": contrastive_flops / 1e9,

        "training_estimated_flops": inference_flops + contrastive_flops,
        "training_estimated_gflops": (
            inference_flops + contrastive_flops
        ) / 1e9,

        "latency_ms": latency_ms,

        "input_shape": [
            1,
            3 if getattr(opt, "rgb", False) else 1,
            opt.imgH,
            opt.imgW,
        ],

        "batch_max_length": opt.batch_max_length,
        "num_classes": opt.num_class,
        "attention_type": opt.attention_type,
        "use_contrastive": opt.use_contrastive,
    }


# -------------------------------------------------------------------------
# Relatório
# -------------------------------------------------------------------------

def write_csv(results, path):
    rows = []

    for r in results:
        row = {
            "config": r["config"],
            "run_id": r["run_id"],
            "params_total": r["params_total"],
            "params_millions": r["params_total"] / 1e6,
            "contrastive_params": r["contrastive_params"],
            "inference_gflops": r["inference_gflops"],
            "contrastive_gflops": r["contrastive_gflops"],
            "training_estimated_gflops": r["training_estimated_gflops"],
            "latency_ms": r["latency_ms"],
            "imgH": r["opt"]["imgH"],
            "imgW": r["opt"]["imgW"],
            "attention_type": r["opt"]["attention_type"],
            "use_contrastive": r["opt"]["use_contrastive"],
        }

        # Componentes principais
        for name, value in r["params_breakdown"].items():
            row[f"params_{name}"] = value

        rows.append(row)

    if not rows:
        return

    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(results, path):
    lines = []

    lines.append("# Model Complexity Profiling")
    lines.append("")
    lines.append(
        "Profiling das quatro configurações do experimento 2×2."
    )
    lines.append("")

    lines.append("## Resumo")
    lines.append("")
    lines.append(
        "| Modelo | Params (M) | Inferência (GFLOPs) | "
        "Contrastive (GFLOPs) | Treino estimado (GFLOPs) | Latência (ms) |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    for r in results:
        lines.append(
            f"| **{r['config']}** "
            f"| {r['params_total']/1e6:.3f} "
            f"| {r['inference_gflops']:.3f} "
            f"| {r['contrastive_gflops']:.3f} "
            f"| {r['training_estimated_gflops']:.3f} "
            f"| {r['latency_ms']:.3f} |"
        )

    lines.append("")
    lines.append("## Parâmetros por componente")
    lines.append("")

    names = []
    for r in results:
        for name in r["params_breakdown"]:
            if name not in names:
                names.append(name)

    lines.append("| Modelo | " + " | ".join(names) + " | Total |")
    lines.append("|---|" + "|".join(["---:"] * len(names)) + "|---:|")

    for r in results:
        values = [
            f"{r['params_breakdown'].get(name, 0)/1e6:.3f}"
            for name in names
        ]

        lines.append(
            f"| **{r['config']}** | "
            + " | ".join(values)
            + f" | {r['params_total']/1e6:.3f} |"
        )

    lines.append("")
    lines.append("## Diferenças relativas ao TRBA")
    lines.append("")

    baseline = next(
        (r for r in results if r["config"] == "TRBA"),
        None,
    )

    if baseline:
        lines.append(
            "| Modelo | Δ Params (M) | Δ Params (%) | "
            "Δ GFLOPs | Δ GFLOPs (%) |"
        )
        lines.append("|---|---:|---:|---:|---:|")

        for r in results:
            dp = r["params_total"] - baseline["params_total"]
            dg = r["inference_gflops"] - baseline["inference_gflops"]

            dp_pct = (
                100 * dp / baseline["params_total"]
                if baseline["params_total"]
                else 0
            )

            dg_pct = (
                100 * dg / baseline["inference_gflops"]
                if baseline["inference_gflops"]
                else 0
            )

            lines.append(
                f"| **{r['config']}** "
                f"| {dp/1e6:+.3f} "
                f"| {dp_pct:+.2f}% "
                f"| {dg:+.3f} "
                f"| {dg_pct:+.2f}% |"
            )

    lines.append("")
    lines.append("## Configurações")
    lines.append("")

    for r in results:
        o = r["opt"]
        lines.append(f"### {r['config']}")
        lines.append("")
        lines.append(f"- Run ID: `{r['run_id']}`")
        lines.append(f"- Input: `{r['input_shape']}`")
        lines.append(f"- Transformation: `{o['Transformation']}`")
        lines.append(f"- FeatureExtraction: `{o['FeatureExtraction']}`")
        lines.append(f"- SequenceModeling: `{o['SequenceModeling']}`")
        lines.append(f"- Prediction: `{o['Prediction']}`")
        lines.append(f"- Attention: `{o['attention_type']}`")
        lines.append(f"- Contrastive: `{o['use_contrastive']}`")
        lines.append(f"- Hidden size: `{o['hidden_size']}`")
        lines.append(f"- Output channels: `{o['output_channel']}`")
        lines.append(f"- Batch max length: `{o['batch_max_length']}`")
        lines.append("")

    lines.append("## Convenção de FLOPs")
    lines.append("")
    lines.append(
        "A contagem utiliza 2 FLOPs por multiply-add. "
        "Operações elementares como ReLU, softmax e dropout não "
        "fazem parte da contagem principal. A branch contrastiva é "
        "reportada separadamente, pois é uma cabeça auxiliar de treinamento."
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Profiling das quatro variantes do experimento 2x2 via MLflow",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mlflow_run_id",
        nargs="+",
        required=True,
        help="Run IDs do MLflow.",
    )

    parser.add_argument(
        "--mlflow_model",
        default="best_accuracy.pth",
        help="Nome do checkpoint salvo no MLflow.",
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda, cuda:0 ou cpu.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Iterações de warmup para medir latência.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Iterações usadas na medição de latência.",
    )

    parser.add_argument(
        "--output_dir",
        default="profiling_results",
        help="Diretório dos resultados.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base opt mínima; os valores reais vêm do MLflow.
    base_opt = argparse.Namespace()
    base_opt.device = device

    results = []

    print("=" * 80)
    print("MODEL COMPLEXITY PROFILING")
    print("=" * 80)
    print(f"Device      : {device}")
    print(f"Warmup      : {args.warmup}")
    print(f"Iterations  : {args.iterations}")
    print()

    for run_id in args.mlflow_run_id:
        print("=" * 80)
        print(f"Run: {run_id}")
        print("=" * 80)

        opt, run = get_run_and_opt(
            base_opt,
            run_id,
            args.mlflow_model,
        )

        config = experiment_config(run)

        print(f"Config      : {config}")
        print(f"Attention   : {opt.attention_type}")
        print(f"Contrastive : {opt.use_contrastive}")
        print(f"Input       : {opt.imgH}x{opt.imgW}")
        print(f"Checkpoint  : {opt.saved_model}")

        model = Model(opt).to(device)

        load_checkpoint(
            model,
            opt.saved_model,
            device,
        )

        result = profile_model(
            model,
            opt,
            device,
            warmup=args.warmup,
            iterations=args.iterations,
        )

        result["config"] = config
        result["run_id"] = run_id
        result["params_breakdown"] = result["params_breakdown"]

        result["opt"] = {
            "Transformation": opt.Transformation,
            "FeatureExtraction": opt.FeatureExtraction,
            "SequenceModeling": opt.SequenceModeling,
            "Prediction": opt.Prediction,
            "attention_type": opt.attention_type,
            "use_contrastive": opt.use_contrastive,
            "imgH": opt.imgH,
            "imgW": opt.imgW,
            "hidden_size": opt.hidden_size,
            "output_channel": opt.output_channel,
            "batch_max_length": opt.batch_max_length,
            "contrastive_embedding_dim": opt.contrastive_embedding_dim,
        }

        results.append(result)

        print()
        print(f"Parameters  : {result['params_total']:,}")
        print(f"Inference   : {result['inference_gflops']:.4f} GFLOPs")
        print(f"Contrastive : {result['contrastive_gflops']:.4f} GFLOPs")
        print(
            f"Train est.  : {result['training_estimated_gflops']:.4f} GFLOPs"
        )
        print(f"Latency     : {result['latency_ms']:.3f} ms")

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------------------------------------------
    # Ordenar na ordem natural do experimento
    # -------------------------------------------------------------
    order = {
        "TRBA": 0,
        "CTRBA": 1,
        "TRB2A": 2,
        "CTRB2A": 3,
    }

    results.sort(key=lambda x: order.get(x["config"], 99))

    # -------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------
    json_path = output_dir / "model_complexity.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # -------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------
    csv_path = output_dir / "model_complexity.csv"
    write_csv(results, csv_path)

    # -------------------------------------------------------------
    # Markdown
    # -------------------------------------------------------------
    md_path = output_dir / "model_complexity.md"
    write_markdown(results, md_path)

    # -------------------------------------------------------------
    # Resumo no terminal
    # -------------------------------------------------------------
    print()
    print("=" * 100)
    print("RESUMO")
    print("=" * 100)

    print(
        f"{'Modelo':<10}"
        f"{'Params(M)':>12}"
        f"{'Infer GFLOPs':>16}"
        f"{'CTR GFLOPs':>15}"
        f"{'Treino est.':>16}"
        f"{'Latency(ms)':>15}"
    )

    print("-" * 100)

    for r in results:
        print(
            f"{r['config']:<10}"
            f"{r['params_total']/1e6:>12.3f}"
            f"{r['inference_gflops']:>16.4f}"
            f"{r['contrastive_gflops']:>15.4f}"
            f"{r['training_estimated_gflops']:>16.4f}"
            f"{r['latency_ms']:>15.3f}"
        )

    print("=" * 100)
    print()
    print(f"[OK] JSON : {json_path}")
    print(f"[OK] CSV  : {csv_path}")
    print(f"[OK] MD   : {md_path}")


if __name__ == "__main__":
    main()