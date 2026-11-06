# Where Should a Gate Look? — code and experiments

Reproducibility code for *Where Should a Gate Look? Input vs. State Gating in
Selective State Space Models*.

The repository reproduces every
number in Tables 1–3. All models share the same architecture —
`embedding → SSM core → output layer` — and the same recurrence; they differ
only in the signal from which the gate is computed, so any difference in
behaviour is attributable to the source of the gate.

| Paper | Gate | SSM parameters | Command |
| --- | --- | --- | --- |
| **IG** — input-gated | `σ(w_u · u_i(k) + b)` | `3nD`, `4nD` with bias | `model=ig` |
| **SG** — state-gated | `σ(w_x ⊙ x(k−1) + b)` | `3nD`, `4nD` with bias | `model=sg` |
| **DG** — dual-gated | `σ(w_u · u_i(k) + w_x ⊙ x(k−1) + b)` | `4nD`, `5nD` with bias | `model=dg` |
| **S6** — input-selective baseline | Mamba selective core | `3nD + D²` | `model=s6` |

The gate bias `b` is optional and off by default; see
[Gate bias](#gate-bias) below.

---

## 1. Installation

The project uses [`uv`](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync
```

This creates a virtual environment from the locked dependencies (`uv.lock`).
Python ≥ 3.9. All commands below are written as `uv run python ...`, which runs
inside that environment.

> **Apple Silicon (MPS).** `torch.linalg.qr`, used by the orthogonal embedding
> initialisation, has no MPS kernel. On a Mac, prefix commands with
> `PYTORCH_ENABLE_MPS_FALLBACK=1`. Not needed on CUDA or CPU.

> **HPC.** If a source build of `scikit-learn` fails for lack of a C++ compiler,
> either load one (`module load gcc`) or pin a Python version with a prebuilt
> wheel: `uv sync --python 3.11`.

## 2. Data

Induction Head and Selective Copying sequences are generated on the fly; nothing
to download. MNIST must be exported once as 2D tensors:

```bash
uv run python preprocessing/MNIST.py
```

This downloads MNIST through `torchvision` and writes tensors under `./data/`.

## 3. Running an experiment

A run is fully specified by a **task** and a **model**:

```bash
uv run python training_script.py experiment=<TASK> model=<MODEL> seed=<SEED> [overrides...]
```

| `experiment=` | Setting |
| --- | --- |
| `induction_head` | `\|V\|=1024`, `L_seq=128`, `D=64`, `n=8`, `lr=3e-3`, 50 epochs |
| `selective_copying` | `A=10`, `L_noise=100`, `D=64`, `n=8`, `lr=3e-3`, 50 epochs |
| `mnist` | 28 row-tokens, four SSM instances, `n=2`, `lr=1e-2`, 50 epochs |
| `induction_head_c6` | `\|V\|=1024`, `L_seq=128`, `D=64`, `n=8`, `lr=3e-3`, 50 epochs |

| `model=` | Module |
| --- | --- |
| `ig`, `sg`, `dg` | `models/gated_ssm.py` |
| `s6` | `models/mamba.py` |

Results are written to `./outputs/<task>/<model>/<job_id>/`: per-epoch
checkpoints, the best model by accuracy and by loss, and a log ending with the
test accuracy and loss.

### State transition range

The transition diagonal is built as `1 − c·σ(A)·Δ`:

| `model.positive_transition` | `c` | Transition range |
| --- | --- | --- |
| `true`  | 1 | `(0, 1)` |
| `false` | 2 | `(−1, 1)` |

Default is set to `false` in the configuration files. To change that, 

```bash
uv run python training_script.py experiment=<TASK> model=<MODEL> seed=<SEED> model.positive_transition=true
```

### Gate bias

A learnable bias inside the gate's sigmoid, `σ(… + b)`, as in the gates of GRUs
and LSTMs. It is off by default and is enabled per run:

```bash
uv run python training_script.py experiment=<TASK> model=<MODEL> seed=<SEED> \
    model.gate_bias=true model.out_dir_name=<MODEL>_bias
```

`model.out_dir_name` is worth setting: results otherwise land in the same
`outputs/<task>/<model>/` tree as the runs without a bias.

The bias is a vector `b ∈ ℝⁿ` per input feature, so it adds `nD` parameters.
It is initialised to zero, which leaves the gate unchanged at
initialisation.

### Accuracy metric

Selected by `dataset.task_type`, which is set per task and needs no override:

| `task_type` | Task | Metric |
| --- | --- | --- |
| `sequence` | Induction Head, Selective Copying | exact match — a sample counts as correct only if every output token matches |
| `classification` | MNIST | top-1 over the ten classes |

## 4. Reproducing the paper tables

Every reported number is the **mean ± std over 3 seeds**, `seed ∈ {1, 2, 3}`.
Run each command once per seed and aggregate.

### Table 1 — Induction Head, matched width (`n = 8`)

```bash
for M in ig sg dg s6; do
  uv run python training_script.py experiment=induction_head model=$M seed=1
done
```

### Table 2 — Selective Copying, `M ∈ {2, 5}`

`M` is the number of content tokens to be copied. Pass it explicitly:

```bash
for M in 2 5; do
  for MODEL in ig sg dg s6; do
    uv run python training_script.py experiment=selective_copying model=$MODEL \
      dataset.sequence_target_length=$M seed=1
  done
done
```

### Table 3 — MNIST

The architecture uses four SSM instances (rows, columns, and both reversed); the
reported parameter counts include all four.

```bash
for M in ig sg dg, s6; do
  uv run python training_script.py experiment=mnist model=$M seed=1
done
```

### Table Appendix C6 — Induction Head

```bash
uv run python training_script.py experiment=induction_head_c6 model=sg seed=1
```

### Bias

For the gate-bias runs add `model.gate_bias=true` to the commands above.

## 5. Repository layout

```text
configs/
  config.yaml                  root config: defaults, output directory
  model/{ig,sg,dg,s6}.yaml     the four models — gate source and parameterisation
  experiment/                  one file per task: induction_head, selective_copying, mnist
models/
  gated_ssm.py                 IG / SG / DG — one recurrence, three gate sources
  mamba.py                     Mamba (https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py) commented, only S6 (Mamba selective core)
  factory.py                   builds embedding + SSM + head from the config
  layer_smoothing_problem.py   the four-SSM layer used for MNIST
  NN.py, embeddings.py, output_layers.py
datasets/
  ih.py                        fixed-trigger Induction Head
  selective_copying.py         Selective Copying
  MNIST.py, factory.py
preprocessing/MNIST.py         one-off MNIST download and 2D tensor export
training_script.py             training, validation and test entry point
utils/                         device selection, paths, job id
```
