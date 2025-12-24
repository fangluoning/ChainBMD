# ChainBMD: An Stroke Evaluation and Guidance Framework

ChainBMD aligns plantar pressure, EMG, and joint Euler angles into 2.5 s (150-frame) sequences. A chained GCN captures body kinetic-chain relations, a Transformer models temporal dynamics, and LRP produces node/time importance. Results can be compared to expert references, used to generate training advice, and explored interactively in a PyQt5 GUI.

---

## 1. Environment Setup

```bash
pip install torch h5py numpy pandas scipy matplotlib scikit-learn pyqt5 requests
# If you use a local LLM (Ollama), install it and pull a model first, e.g., deepseek-r1:7b
```

---

## 2. Data Preprocessing

MultiSenseBadminton dataset download: https://doi.org/10.6084/m9.figshare.c.6725706.v1
After downloading, place the raw data in `Data_Archive/`, then run the scripts below to generate HDF5 files. All downstream training, explanation, and visualization steps use the generated HDF5.

| Script | Description | Output |
|------|------|------|
| `python data_preprocessing.py` | Merge all sensor streams, keep Forehand Clear only, and generate the 38-dim feature matrix plus LRP metadata. | `data_processed/data_processed_allStreams_60hz_onlyForehand_skill_level.hdf5` |
| `python data_preprocessing_skeleton.py` | Extract `pns-joint/global-position` only for the GUI 3D skeleton. | `data_processed/data_processed_allStreams_60hz_onlyForehand_skeleton_skill_level.hdf5` |

> If you only want to reproduce downstream steps, you can reuse the prepared HDF5 files.

---

## 3. Train / Validate / Test

| Scenario | Command | Notes |
|------|------|------|
| Train ChainBMD (default) | `python train/train.py` | Random sample split 70/20/10, checkpoint saved to `outputs/checkpoints/chainbmd_best.pt`. |
| Evaluate only | `python test/test.py --checkpoint outputs/checkpoints/chainbmd_best.pt` | Uses the same split and feature configuration as training. |
| Explain a full split | `python script/explain.py --split test --samples -1` | Uses training-time settings by default; can add `--subject-split/--feature-subset`. |
| Explain a single sample | `python script/explain_sample.py --split test --index 0` | Uses training-time settings; supports `--checkpoint` for any model file. |
| GUI / LRP visualization | `python app/skill_visualizer.py` | Auto-loads `train/config.py`. For feature ablation, set it in config or CLI first. |

> You can override config via env vars, e.g., `CHAINBMD_EPOCHS=50 python train/train.py --subject-split`.
> Training uses early stopping by default (`patience=20`, based on val accuracy). Adjust `early_stopping_patience` in `train/config.py` if needed.

### 3.1 Data Splits and Cross-Validation

| Command | Notes |
|------|------|
| `python train/train.py --subject-split` | Subject-level split (same player never appears in multiple splits), avoids identity leakage. |
| `python train/train.py --kfold 5` | 5-fold cross-validation (supports `--subject-split`). Reports mean val accuracy; only logs train/val and does not produce test plots. |
| `python test/test.py --subject-split` | Uses the same subject split as training. |

All entry points (train/test/explain) accept `--subject-split` to keep training and inference consistent. For node ablation, add `--feature-subset <key>` as shown below.

### 3.2 Node Feature Ablation

`train/config.py` defines a 7-node to 38-dim mapping. You can mask columns with `--feature-subset`:

| Key | Node set | Column indices |
|-----|---------|--------------|
| `node13` | {1,3} | `[0,1,2,6,7,8,9]` |
| `node123` | {1,2,3} | `[0..9]` |
| `node1234` | {1,2,3,4} | `[0..18]` |
| `node12345` | {1,2,3,4,5} | `[0..26]` |
| `node123456` | {1,2,3,4,5,6} | `[0..34]` |
| `node1234567` / `all` | all nodes | `[0..37]` |

Example: train GCN + Transformer with the first 4 nodes only

```bash
python train/train.py --feature-subset node1234
python test/test.py --feature-subset node1234 --checkpoint outputs/checkpoints/chainbmd_best.pt
```

### 3.3 Multi-Model / Ablation Combinations

| Goal | Example command |
|------|------|
| ChainBMD + all features | `python train/train.py` |
| ChainBMD + node ablation + subject split | `python train/train.py --feature-subset node1234 --subject-split` |
| K-fold + subject split | `python train/train.py --kfold 5 --subject-split` (run a non-k-fold training if you need test metrics/plots) |

> Test and explain scripts also accept `--subject-split` and `--feature-subset` to mirror these combinations.

### 3.4 ChainBMD Module Ablation Experiments

> Goal: remove key submodules in `models/chainbmd_model.py` one at a time to verify the necessity of chained GCN, Transformer, CLS pooling, and the LRP analyzer.

#### Experimental Setup

- **Data & split**: reuse the default Forehand HDF5 (sample-level 70/20/10 split, seed=42), keep batch size/epochs/optimizer consistent. The reference run can reuse `outputs/checkpoints/chainbmd_best.pt` (`Accuracy 0.8777 / Macro-F1 0.8720 / Macro-AUC 0.9493`, from `python test/test.py`).
- **Metrics**: top-1 accuracy, macro F1, macro ROC-AUC; keep confusion matrix and ROC curves for visual comparison.
- **Procedure**: (1) train/evaluate the reference run; (2) change one component per run; (3) record differences in `outputs/logs/ablation_chainbmd.md` (optional).

#### Ablation Variants

| Variant | Key change | Expected observation | Conclusion |
|---------|------------|----------------------|-----------|
| **Full ChainBMD** | Default: 3-layer chained GCN + 3-layer Transformer + CLS token + LRP. | Match reference metrics in `test/test.py`. | Control group showing upper bound. |
| **w/o Chain GCN** | Set `ChainBMDConfig(gcn_layers=0, gcn_hidden=in_features)` in `train/config.py`. | Node interaction removed, val/test accuracy drops, confusion matrix skews to majority class. | Without cross-node kinetic relations, Transformer sees only independent node averages. |
| **w/o Temporal Transformer** | Temporarily replace Transformer + CLS in `ChainBMDModel.forward` with `temporal_tokens.mean(dim=1)` (keep LayerNorm + MLP). | Metrics collapse toward random, ROC close to diagonal, loss hard to decrease. | Temporal dynamics require Transformer; simple averaging loses order. |
| **w/o Positional Encoding / CLS** | Comment out `self.positional_encoding` and `cls_token`, use last frame or mean output. | Better than removing Transformer, but still degraded; model focuses on a few frames. | Positional encoding + CLS pooling preserves global temporal semantics. |

> Save each run's `python test/test.py` output (values + plots) under `outputs/figures/test_metrics/<variant>_*` and cite in the README/report to show each module contributes.

#### One-Click Runner

To avoid manual config edits and restarts, use `script/run_chainbmd_ablation.py`:

```bash
# Runs five variants by default, 200 epochs; adjust as needed
python script/run_chainbmd_ablation.py \
    --epochs 200 \
    --batch-size 32 \
    --variants full,no_gcn,no_transformer,no_pos_cls \
    --feature-subsets all
```

- `--variants`: comma-separated, matches the table keys (`full` / `no_gcn` / `no_transformer` / `no_pos_cls`).
- `--feature-subsets`: comma-separated node keys (`all`/`node13`/...). The script writes results for multiple subsets into one JSON. The legacy `--feature-subset node1234` is still supported.
- `--subject-split`: inherited from the training script, for subject-level split reproduction; other hyperparams can be overridden via env vars.
- `--output`: path to save results JSON, default `outputs/logs/chainbmd_ablation.json`:
  ```json
  [
    {
      "variant": "full",
      "description": "Default ChainBMD (GCN + Transformer + CLS + LRP)",
      "val_metrics": {"accuracy": 0.88, "f1": 0.87, ...},
      "test_metrics": {"accuracy": 0.8777, "f1": 0.8720, ...}
    },
    ...
  ]
  ```
  You can cite it directly in reports or load it in `script/draw.py` for visualization.

The script trains each variant separately (shared split), selects the best checkpoint by `val_acc`, and evaluates on test for fair comparison.

#### One-Click Node Subset Loop

To compare multiple node sets in one run:

```bash
python script/run_chainbmd_ablation.py \
    --epochs 30 \
    --variants full \
    --feature-subsets node13,node123,node1234,node12345,node123456,node1234567 \
    --output outputs/logs/node_ablation_summary.json
```

- The script rebuilds dataloaders and node mappings per subset (`train/config.py` handles it) and writes `feature_subset` in the JSON for easy comparison or further module ablation.
- If you still want separate files, call the script multiple times with different `--output` paths.

---

## 4. Generate LRP Explanations

```bash
python script/explain.py --split test --samples 5 --output outputs/figures/explanations.json
```

- Use `--samples -1` to traverse an entire split.
- Filter a skill level (e.g., expert = 2) with `--filter-skill 2`:
  ```bash
  python script/explain.py --split test --samples -1 \
      --filter-skill 2 \
      --output outputs/figures/explanations_skill2.json
  ```

JSON fields include: true/predicted labels, class probabilities, 7-node contributions, and 150-frame time importance. For a single sample:

```bash
python script/explain_sample.py --split test --index 0 --output outputs/figures/sample_0.json
```

### 4.1 Expert LRP Benchmark Generation

To keep the "expert/high-skill" reference consistent, run the following after training:

1. **Batch expert explanations**: Use training config and checkpoint to explain all expert (skill=2) samples in train or test.
   ```bash
   python script/explain.py \
       --split train \
       --samples -1 \
       --filter-skill 2 \
       --output outputs/figures/expert_explanations.json
   ```
   This traverses the split, loads ChainBMD, and outputs JSON with node/time contributions.

2. **Build expert benchmark**: Aggregate LRP into a mean contribution template, used for expert curves/heatmaps.
   ```bash
   python script/build_benchmark.py \
       --input outputs/figures/expert_explanations.json \
       --output outputs/figures/expert_benchmark.json
   ```
   The script computes mean and std for the 7x150 contributions and outputs JSON for downstream visualization or diagnostics.

3. **Compare and visualize**: Overlay the expert range vs. a target player.
   ```bash
   MPLCONFIGDIR=./outputs python script/visualize_benchmark.py \
       --benchmark outputs/figures/expert_benchmark.json \
       --record outputs/figures/sample_0.json \
       --output_dir outputs/figures/benchmark_vs_sample
   ```
   This produces plots such as expert band vs. single sample or expert radar charts. To show only the expert range, omit `--record`.

This full chain runs in one command sequence without manual sample stitching. We use the same `expert_explanations.json` and `expert_benchmark.json` in the experiments.

### 4.2 LRP Offline Faithfulness (Deletion/Insertion)

Rank features by LRP importance, then delete/insert to evaluate output changes and curve AUC.

- **Base evaluation (zero reference)**
  ```bash
  python script/eval_lrp_faithfulness.py \
      --split test \
      --samples -1 \
      --output outputs/metrics/lrp_faithfulness.csv
  ```

- **Plot curves**
  ```bash
  python script/eval_lrp_faithfulness.py \
      --split test \
      --samples -1 \
      --output outputs/metrics/lrp_faithfulness.csv \
      --plot-curves \
      --plot-all \
      --plot-output-dir outputs/metrics/curves
  ```

### 4.3 LRP-Guided Improvement (In-Model Proxy)

Modify only the time windows of nodes flagged by the guidance, nudging them toward the expert mean, and observe whether the target skill output improves.

```bash
python script/eval_lrp_guided_improvement.py \
    --split test \
    --samples -1 \
    --target-skill 2 \
    --deviation-threshold 1.0 \
    --alphas 0.5,1.0 \
    --opt-steps 40 \
    --opt-lrs 0.03,0.05,0.08,0.1 \
    --opt-patience 5 \
    --opt-min-delta 1e-4 \
    --opt-objective prob \
    --benchmark outputs/figures/benchmark_skill2.json \
    --output outputs/metrics/lrp_guided_improvement.csv
```

By default, only samples with `pred_before != target_skill` are counted. Add `--include-target` to include already-targeted samples.

## 5. Visualization (All Based on LRP JSON)

| Script | Purpose |
|------|------|
| `script/visualize_explanations.py` | Seven-node time series (gray grid, Times font, optional event markers). |
| `script/visualize_heatmap.py` | Node-by-time heatmap to locate high-importance windows. |
| `script/visualize_radar.py` | Node contribution radar chart to compare samples. |
| `script/visualize_comparison.py` | Group by true or predicted label, plot mean +/- std curves. |
| `script/visualize_benchmark.py` | Overlay expert benchmark with a single sample. |

All commands (copy-ready):

```bash
# 1) Multi-panel node curves
MPLCONFIGDIR=./outputs python script/visualize_explanations.py \
    --input outputs/figures/explanations.json \
    --output_dir outputs/figures/viz \
    --events 70 90

# 2) Node-time heatmap
MPLCONFIGDIR=./outputs python script/visualize_heatmap.py \
    --input outputs/figures/explanations.json \
    --output_dir outputs/figures/heatmaps

# 3) Node contribution radar
MPLCONFIGDIR=./outputs python script/visualize_radar.py \
    --input outputs/figures/explanations.json \
    --output_dir outputs/figures/radar

# 4) Grouped comparison
MPLCONFIGDIR=./outputs python script/visualize_comparison.py \
    --input outputs/figures/explanations.json \
    --group true_label \
    --output_dir outputs/figures/group_compare
```

---

## 6. Build Expert Benchmark, Diagnose Deviations, and Trigger LLM

1. **Collect target-skill explanations (optional)**
   ```bash
   python script/explain.py --split test --samples -1 --filter-skill 2 \
       --output outputs/figures/explanations_skill2.json
   ```

2. **Build the benchmark** (default skill level = 2)
   ```bash
   python script/build_benchmark.py \
       --input outputs/figures/explanations_skill2.json \
       --output outputs/figures/benchmark_skill2.json
   ```
   You can visualize the group range directly:
   ```bash
   MPLCONFIGDIR=./outputs python script/visualize_benchmark.py \
       --benchmark outputs/figures/benchmark_skill2.json \
       --output_dir outputs/figures/benchmark_viz
   ```

3. **Benchmark vs. single sample**
MPLCONFIGDIR=./outputs python script/visualize_benchmark.py \
    --benchmark outputs/figures/benchmark_skill2.json \
    --record outputs/figures/sample_0.json \
    --output_dir outputs/figures/benchmark_viz

4. **Diagnose a single sample and generate advice**
   ```bash
   python script/analyze_deviation.py \
       --record outputs/figures/sample_0.json \
       --benchmark outputs/figures/benchmark_skill2.json \
       --threshold 1.0 \
       --use_llm --llm_model deepseek-r1:7b
   ```
   - Start Ollama (default at `http://localhost:11434`).
   - With `--use_llm --llm_model deepseek-r1:7b`, the script sends node/time deviation summaries to the LLM and prints coaching feedback in Chinese or English.

---

## 7. PyQt5 Visualization App

```bash
python app/skill_visualizer.py
```

Features:
- Left: read HDF5 rows by index (if no subject_id, uses `sample_xxxx`), plot EMG and insole/joint curves with gray grid and Times font.
- Center: node contribution summary, LLM coaching feedback/chat, AI coach button (triggers LLM).
- Upper right: 2.5 s skeleton animation (auto-play by default, can pause), driven by skeleton HDF5.
- Lower right: local LLM chat panel, supports CN/EN input.
- Language is toggled in settings (`language=en/cn`), and button/status text follows.
