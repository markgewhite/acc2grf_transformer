# Experiment Log: ACC → GRF Prediction

This document records the experiments conducted to develop a model that predicts vertical ground reaction force (vGRF) from lower-back accelerometer data during countermovement jumps.

## Objective

Develop a machine learning model to map accelerometer signals to vGRF, with the ultimate goal of accurately predicting biomechanical metrics (jump height, peak power) from the predicted GRF — enabling force plate-quality metrics from a single wearable sensor.

---

## Current Status (January 2026)

**🎯 BREAKTHROUGH: First positive Jump Height R²**

**Best configuration:**
```bash
python src/train.py \
    --model-type mlp --mlp-hidden 128 \
    --use-triaxial \
    --input-transform fpc --output-transform fpc \
    --loss reconstruction \
    --simple-normalization \
    --n-trials 5 --seed 42 \
    --epochs 200
```

**Results (5-trial mean ± std, triaxial input, h=128):**

| Metric | Unweighted | Weighted | Difference |
|--------|-----------|----------|------------|
| JH R² | **0.823 ± 0.030** | 0.820 ± 0.021 | Not significant |
| JH Median AE | 0.035 ± 0.002 m | 0.035 ± 0.002 m | Identical |
| PP R² | 0.798 ± 0.029 | **0.801 ± 0.020** | Not significant |
| PP Median AE | 2.66 ± 0.07 W/kg | 2.58 ± 0.19 W/kg | Not significant |
| Signal R² (BW) | 0.971 ± 0.001 | 0.971 ± 0.001 | Identical |

- Parameters: ~12K (45×128 + 128 + 128×15 + 15) — 3 channels × 15 FPCs input
- Invalid samples: **0** (no negative JH predictions)

**Note:** Temporal weighting (reconstruction_weighted) makes no significant difference. Both configurations perform equivalently within run-to-run variability.

**Resultant baseline (5-trial mean ± std):**

| Metric | Resultant | Triaxial | Improvement |
|--------|-----------|----------|-------------|
| JH R² | 0.673 ± 0.015 | **0.823 ± 0.030** | **+0.15** (~5σ) |
| PP R² | 0.690 ± 0.013 | **0.798 ± 0.029** | **+0.11** (~4σ) |
| Signal R² | 0.960 ± 0.001 | **0.971 ± 0.001** | +0.011 |

**Key insights:**
1. **Triaxial + FPC + MLP** is the winning combination — JH R² 0.82 ± 0.03, PP R² 0.80 ± 0.03
2. **Triaxial provides a statistically significant improvement** over resultant (+0.15 JH R², ~5 standard deviations)
3. FPC representation captures the features needed for jump height prediction
4. A simple MLP (~12K params) massively outperforms the transformer (~750K params)
5. JH R² 0.82 approaches the reference baseline of 0.87 (actual 500ms curves vs ground truth)
6. **Temporal weighting makes no difference** — previous single-run observations were within noise

**Hybrid architecture comparison (5-trial mean ± std):**

| Metric | MLP | Hybrid Residual | Hybrid Sequential |
|--------|-----|-----------------|-------------------|
| JH R² | **0.823 ± 0.030** | 0.735 ± 0.013 | 0.763 ± 0.022 |
| JH Median AE | 0.035 ± 0.002 m | **0.033 ± 0.001 m** | 0.037 ± 0.002 m |
| PP R² | **0.798 ± 0.029** | 0.702 ± 0.020 | 0.730 ± 0.029 |
| PP Median AE | 2.66 ± 0.07 W/kg | **2.52 ± 0.15 W/kg** | 2.79 ± 0.10 W/kg |
| Signal R² (BW) | 0.971 ± 0.001 | **0.971 ± 0.001** | 0.969 ± 0.001 |

**Key trade-off:** Hybrid residual achieves **lower absolute errors** (better for tracking individual athletes) while hybrid sequential achieves **higher R²** (better for population-level variance explanation). MLP remains best overall for R² metrics.

---

## Summary & Publication Potential

### The Journey

This project began with a transformer architecture (~750K parameters) achieving negative JH R² values — the model could reconstruct GRF curves beautifully (Signal R² > 0.9) but the predicted curves produced meaningless jump height estimates. After extensive experimentation, we discovered that:

1. **Representation matters more than architecture**: A simple MLP with 12K parameters outperforms a 750K-parameter transformer
2. **FPC representation is the key**: Functional Principal Components capture biomechanically relevant features that raw signals and B-splines miss
3. **Triaxial input preserves critical information**: Directional acceleration data improves JH R² by 0.15 compared to resultant magnitude

### Final Results (5-trial validation)

| Metric | Value | Context |
|--------|-------|---------|
| **Jump Height R²** | **0.82 ± 0.03** | Reference baseline: 0.87 |
| **Jump Height Median AE** | **3.5 ± 0.2 cm** | Clinically meaningful |
| **Peak Power R²** | **0.80 ± 0.03** | Reference baseline: 0.99 |
| **Peak Power Median AE** | **2.7 ± 0.1 W/kg** | Acceptable for monitoring |
| **Signal R² (BW)** | **0.971 ± 0.001** | Excellent curve reconstruction |
| **Invalid predictions** | **0** | No negative jump heights |

### Why This Matters

1. **Practical application**: Force plates cost £5,000-50,000 and are immobile. A lower-back accelerometer costs £50 and goes anywhere.

2. **Validated accuracy**: JH R² of 0.82 approaches the theoretical maximum (0.87) set by the 500ms signal window limitation. The remaining gap is partly due to information lost by truncating the signal.

3. **Robust methodology**: 5-trial validation with different random seeds provides confidence intervals, not just point estimates.

4. **Surprising findings**:
   - Simple beats complex (MLP > Transformer)
   - Temporal weighting doesn't help (contrary to intuition)
   - Triaxial significantly outperforms resultant (preserving direction matters)

### Publication Angles

1. **Methods paper**: "Predicting Ground Reaction Force from Wearable Accelerometers using Functional Principal Components"
   - Focus on the FPC-MLP pipeline
   - Emphasize the representation learning insight
   - Compare against raw/B-spline baselines

2. **Applied paper**: "Accurate Jump Height Estimation from a Single Wearable Sensor"
   - Focus on practical deployment
   - Emphasize 3.5 cm median error
   - Compare against existing wearable solutions

3. **Lessons learned paper**: "Why Transformers Fail at Biomechanical Signal Prediction"
   - Focus on the negative results
   - Explain why attention doesn't help for this task
   - Guide future researchers away from over-engineering

### Limitations to Address

1. **Single dataset**: 69 subjects, 346 jumps (noarms vertical CMJs) — need external validation
2. **Fixed sensor location**: Lower back only — generalization to other placements unknown
3. **CMJ only**: May not transfer to other jump types or movements
4. **500ms window**: Truncation explains some of the JH R² gap vs ground truth

---

**Key lessons learned:**

1. **Normalization is critical:**
   - MAD-based normalization created extreme values (range -27 to +52) that networks couldn't predict
   - Simple global z-score normalization fixed this
   - Signal R² improved from 0.65 to 0.94

2. **FPC representation is transformative:**
   - Mean function captures the CMJ template — model only learns deviations
   - Variance-ordered components naturally weight importance
   - 15 FPCs vs 500 raw samples — massive dimensionality reduction

3. **Triaxial > Resultant (statistically confirmed):**
   - Directional information in x/y/z improves prediction
   - JH R² improves by 0.15 (5 standard deviations)
   - The vertical axis likely provides direct physical correspondence to vertical force

4. **Simple architectures suffice:**
   - MLP (12K params) outperforms Transformer (750K params)
   - Attention mechanism adds no value for this mapping
   - Hybrid linear projection + MLP does not improve upon simple MLP
   - The FPC representation does the heavy lifting

5. **Multi-trial validation essential:**
   - Single-run results can be misleading (e.g., "weighted is worse")
   - Run-to-run std of ~0.03 in R² metrics
   - 5 trials minimum for reliable comparisons

**Previous findings (still relevant):**

5. **Triaxial vs Resultant depends on transform type:**
   - Raw signals: Triaxial > Resultant (preserves directional information)
   - FPC transforms: Resultant > Triaxial (triaxial causes dimensionality explosion)

6. **scikit-fda adoption caused performance regression:**
   - Original custom FPCA: JH R² ≈ 0.61
   - After scikit-fda refactoring: JH R² ≈ 0.34 (best case)
   - Score scaling experiments did not recover original performance

---

## Data Overview

- **Source**: `data/cmj_dataset.npz` (generated from MATLAB files by `scripts/prepare_dataset.py`)
- **Condition**: Vertical countermovement jumps without arm swing (noarms)
- **Subjects**: 69 unique (4 duplicate participant IDs merged)
- **Valid jumps**: 346 total, ~69 in validation set
- **ACC sampling rate**: 250 Hz (lower back sensor, triaxial)
- **GRF sampling rate**: 1000 Hz → downsampled to 250 Hz (already in BW units)
- **Sequence length**: Padded/truncated to 500 samples (2000 ms pre-takeoff)
- **Train/Val split**: 80/20 at subject level (no data leakage)

---

## Experiment Summary

| Run | Input | Model Size | Loss | Signal R² | JH Median AE | JH R² | PP R² | Notes |
|-----|-------|------------|------|-----------|--------------|-------|-------|-------|
| test_run | resultant | d=64, ff=128 | MSE | 0.831 | - | -12.1 | 0.10 | 5 epochs only |
| test_run3 | resultant | d=64, ff=128 | MSE | 0.913 | 0.214 m | -2.79 | 0.11 | Baseline |
| test_run4 | **triaxial** | d=64, ff=128 | MSE | 0.925 | - | -0.23 | 0.29 | Triaxial helps |
| test_run5 | triaxial | **d=128, ff=512** | MSE | 0.908 | 0.214 m | -2.79 | 0.11 | Larger model |
| jh_loss | triaxial | d=128, ff=512 | **JH only** | -8.53 | 1.57 m | -90.8 | -88.5 | Collapsed |
| comb_loss2 | triaxial | d=128, ff=512 | MSE+0.1×JH | 0.868 | **0.094 m** | -1.33 | 0.24 | JH median good |
| comb_loss3 | triaxial | d=128, ff=512 | MSE+0.1×PP | **0.933** | 0.251 m | -2.28 | **0.64** | Best overall |
| extended_1000 | triaxial+1s post | d=128, ff=512 | MSE+0.1×PP | 0.884 | 0.276 m | -13.9 | 0.59 | Worse than comb_loss3 |
| comb_loss4 | triaxial | d=128, ff=512 | MSE+0.1×JH+0.1×PP | 0.885 | 0.126 m | -1.62 | 0.21 | JH dominates PP |
| comb_loss5 | triaxial | d=128, ff=512 | MSE+0.01×JH+0.1×PP | 0.883 | 0.192 m | -8.01 | 0.31 | JH still harmful |
| weighted_1 | triaxial | d=128, ff=512 | Weighted MSE | 0.908 | 0.160 m | -2.21 | -0.14 | Biases, PP worse |
| bspline-jh_branch | resultant, bspline | d=64, ff=128 | Reconstruction | 0.933 | 0.222 m | -3.60 | 0.08 | Baseline (no scalar branch) |
| bspline-jh_scalar | resultant, bspline | d=64, ff=128 | Recon + scalar MSE | 0.871 | 0.259 m | -6.43 | -0.03 | Scalar branch hurt both tasks |
| bspline-jh_stop | resultant, bspline | d=64, ff=128 | Recon + scalar (stop_grad) | 0.858 | 0.294 m | -7.94 | -0.01 | stop_gradient didn't help |
| bspline-jh_avgpool | resultant, bspline | d=64, ff=128 | Recon + scalar (avgpool) | 0.830 | 0.603 m | -26.2 | -0.63 | Global avg pooling worse |
| scalar-only-jh | resultant, bspline | d=64, ff=128 | Scalar MSE only | — | — | — | — | Scalar R²=0.20, encoder can't learn JH |
| mlp-raw-bspline-64 | raw→bspline | MLP h=64 | Reconstruction | 0.946 | 0.279 m | -5.40 | 0.44 | Raw input outperforms bspline input |
| **mlp-raw-bspline-128** | **raw→bspline** | **MLP h=128** | Reconstruction | **0.951** | 0.266 m | -6.11 | **0.52** | **Best PP R², ~8K params** |
| mlp-bspline-bspline-64 | bspline→bspline | MLP h=64 | Reconstruction | 0.942 | 0.229 m | -3.77 | 0.38 | B-spline input hurts performance |
| mlp-bspline-bspline-128 | bspline→bspline | MLP h=128 | Reconstruction | 0.951 | 0.262 m | -4.36 | 0.46 | Still worse than raw input |
| mlp-fpc-fpc-64 | fpc→fpc | MLP h=64 | Reconstruction | 0.958 | 0.053 m | 0.67 | 0.68 | 250 epochs |
| mlp-fpc-fpc-128 | fpc→fpc | MLP h=128 | Reconstruction | 0.960 | 0.051 m | 0.67±0.02 | 0.69±0.01 | 5-trial, resultant |
| **mlp-fpc-triaxial** | **fpc→fpc (triaxial)** | **MLP h=128** | **Reconstruction** | **0.971** | **0.035 m** | **0.82±0.03** | **0.80±0.03** | **5-trial mean±std** |
| mlp-fpc-triaxial-weighted | fpc→fpc (triaxial) | MLP h=128 | Reconstruction (weighted) | 0.971 | 0.035 m | 0.82±0.02 | 0.80±0.02 | 5-trial: no difference |
| mlp-fpc-fpc-256 | fpc→fpc | MLP h=256 | Reconstruction | 0.961 | 0.050 m | 0.69 | 0.70 | No improvement over h=128 |
| hybrid-residual | fpc→fpc (triaxial) | Hybrid+MLP | Reconstruction | 0.971 | **0.033 m** | 0.74±0.01 | 0.70±0.02 | 5-trial: lowest AE |
| hybrid-sequential | fpc→fpc (triaxial) | Hybrid+MLP | Reconstruction | 0.969 | 0.037 m | 0.76±0.02 | 0.73±0.03 | 5-trial: higher R² |
| mlp-fpc-eigenvalue | fpc→fpc | MLP h=128 | Eigenvalue-weighted | 0.949 | 0.063 m | 0.61 | 0.65 | Over-weights FPC1, hurts JH/PP |
| mlp-fpc-signal-space | fpc→fpc | MLP h=128 | Signal-space | 0.961 | 0.053 m | 0.67 | 0.68 | Unweighted |
| mlp-fpc-ss-weighted | fpc→fpc | MLP h=128 | Signal-space-weighted | 0.960 | 0.048 m | 0.67 | 0.69 | Jerk-weighted, best median errors |
| mlp-fpc-varimax | fpc→fpc (varimax) | MLP h=128 | Reconstruction | 0.960 | 0.050 m | 0.66 | 0.68 | Varimax hurts slightly |

---

## Detailed Experiment Notes

### Experiment 1: Baseline with Resultant Acceleration (test_run3)

**Configuration:**
- Input: Resultant acceleration √(x² + y² + z²)
- Model: d_model=64, num_heads=4, num_layers=3, d_ff=128
- Loss: MSE
- Epochs: 100 (early stopped at ~61)

**Results:**
- Signal RMSE: 0.126 BW, R² = 0.913
- Jump Height RMSE: 0.319 m, R² = -2.79
- Peak Power R² = 0.11

**Observation:** Good signal reconstruction but poor biomechanics metrics. The model predicts smooth GRF curves but misses critical features in the propulsion phase.

---

### Experiment 2: Triaxial Input (test_run4)

**Change:** Used raw triaxial (x, y, z) instead of resultant

**Results:**
- Signal R²: 0.913 → 0.925
- Jump Height RMSE: 0.343 m → 0.182 m (↓47%)
- Jump Height Bias: -0.240 m → -0.009 m (nearly eliminated)
- Peak Power R²: 0.26 → 0.29

**Inference:** Triaxial input preserves directional information critical for predicting vertical GRF. The vertical acceleration component (typically z-axis) provides direct physical correspondence to vertical force.

---

### Experiment 3: Larger Model (test_run5)

**Change:** Increased model capacity (d_model=64→128, d_ff=128→512)

**Results:**
- Training and validation losses matched (no overfitting)
- Signal metrics slightly worse due to outliers
- Biomechanics metrics showed high variance

**Observation:** Larger model generalizes well but a few extreme outliers skew the mean-based metrics. This motivated adding robust metrics (median AE).

---

### Experiment 4: Jump Height Loss Only (jh_loss)

**Change:** Loss = MSE(actual_JH, predicted_JH)

**Results:**
- Complete model collapse
- Predicted GRF: random noise with no structure
- Jump Height predictions: ~1.5-1.9 m (physically impossible)
- All metrics negative R²

**Inference:** Optimizing only for jump height creates a degenerate solution. The model learns that any signal producing high integrated values yields low loss, without learning actual GRF dynamics. There's no constraint to produce realistic force curves.

---

### Experiment 5: Combined Loss with Jump Height (comb_loss2)

**Configuration:**
- Loss = 1.0×MSE + 0.1×JH_loss + 0.0×PP_loss

**Results:**
- Signal R² = 0.868 (slightly lower)
- Jump Height Median AE = **0.094 m** (9.4 cm - excellent!)
- Jump Height RMSE = 0.250 m (pulled up by outliers)
- A few severe outliers predicting JH of -1.5 m

**Inference:** The combined loss maintains signal structure while improving typical-case jump height predictions. However, the jump height component may be destabilizing for edge cases, causing some predictions to go strongly negative.

---

### Experiment 6: Combined Loss with Peak Power (comb_loss3)

**Configuration:**
- Loss = 1.0×MSE + 0.0×JH_loss + 0.1×PP_loss

**Results:**
- Signal R² = **0.933** (best achieved)
- Signal RMSE = 0.108 BW (best achieved)
- Peak Power R² = **0.64** (major improvement)
- Peak Power Median AE = 4.98 W/kg
- Jump Height still poor (R² = -2.28)

**Observation:** Prediction curves visually closest to actual GRF. Even outliers showed reasonable signal shapes.

**Inference:** Peak power loss works better than jump height loss because:

1. **Localized feature**: Peak power depends on getting the maximum F×v correct at one point in time, not integrating over the entire signal.

2. **Stable gradients**: The max operation provides focused gradient signal to the peak location, rather than diffuse gradients across all timesteps.

3. **Less error accumulation**: Jump height requires double integration where errors compound; peak power only needs single integration plus max.

4. **Natural emphasis on propulsion**: The propulsion phase (where GRF > 1 BW and velocity is positive) is exactly where peak power occurs, so the loss naturally emphasizes this critical region.

---

### Experiment 7: Extended Input with Post-Takeoff ACC (extended_1000)

**Hypothesis:** Including 1000ms of accelerometer data after takeoff (flight phase + landing) might provide additional context for predicting pre-takeoff GRF. Flight time directly encodes jump height (h = ½gt²), and landing patterns may correlate with takeoff characteristics.

**Configuration:**
- Input: 750 samples (2000ms pre-takeoff + 1000ms post-takeoff)
- Output: 500 samples (2000ms pre-takeoff only, GRF=0 during flight)
- Model: d_model=128, d_ff=512 (same as comb_loss3)
- Loss: MSE + 0.1×PP (same as comb_loss3)

**Implementation:** The transformer encoder processes all 750 input samples with full self-attention, but the output is sliced to the first 500 positions before the final projection. Post-takeoff ACC provides context but no GRF predictions are made for that period.

**Results:**
- Signal R² = 0.884 (↓ from 0.933)
- Signal RMSE = 0.142 BW (↑ from 0.108 BW)
- Peak Power R² = 0.59 (↓ from 0.64)
- Jump Height R² = -13.9 (↓ from -2.28)

**Inference:** The post-takeoff extension degraded performance across all metrics. Possible reasons:

1. **Attention dilution**: Self-attention over 750 samples instead of 500 spreads attention too thin, diluting focus on the critical propulsion phase.

2. **Irrelevant context**: Post-takeoff ACC (freefall ~0g, landing impact) does not provide useful signal for reconstructing pre-takeoff GRF—it's essentially noise from the model's perspective.

3. **Causality**: GRF during pre-takeoff is determined by what happens *before* takeoff, not after. While flight time encodes jump height, the model cannot use future information to reconstruct the GRF curve that produced it.

4. **More parameters, same data**: Larger positional embeddings (750 vs 500) require more parameters without additional training signal.

**Conclusion:** The pre-takeoff ACC already contains all causally relevant information. Post-takeoff extension does not help and should not be used. Default `--post-takeoff-ms` remains 0.

---

### Experiment 8: Combined Loss with Both JH and PP (comb_loss4)

**Hypothesis:** Combining both jump height and peak power auxiliary losses might capture benefits of both—JH for reducing systematic bias observed in Bland-Altman plots, PP for maintaining good signal reconstruction.

**Configuration:**
- Loss = 1.0×MSE + 0.1×JH_loss + 0.1×PP_loss
- Model: d_model=128, d_ff=512

**Results:**
- Signal R² = 0.885 (↓ from comb_loss3's 0.933)
- Signal RMSE = 0.141 BW
- Jump Height Median AE = 0.126 m (between comb_loss2's 0.094 m and comb_loss3's 0.251 m)
- Jump Height R² = -1.62, Bias = -0.071 m
- Peak Power R² = 0.21 (↓↓ from comb_loss3's 0.64)
- Peak Power Bias = +6.86 W/kg (shifted to over-prediction)
- Extreme outliers: 2 samples with predicted JH < -0.4 m

**Observation:** The losses interfere with each other. Compared to comb_loss3 (PP only):
- Signal reconstruction degraded
- PP predictions shifted from well-centered to systematic over-prediction (+6.86 W/kg bias)
- JH predictions improved in clustering around identity line, but extreme negative outliers persist

**Inference:** Jump height loss dominates peak power loss at equal weights (0.1 each). This is likely because:

1. **Gradient magnitude**: JH loss involves double integration, producing larger gradients that propagate through more timesteps than PP's localized max operation.

2. **Conflicting objectives**: Optimizing for JH (which requires accurate integration over entire signal) may push the model in different directions than optimizing for PP (which focuses on the propulsion peak).

3. **Scale mismatch**: JH errors (in meters) and PP errors (in W/kg) may have inherently different gradient magnitudes even with equal weights.

**Next step:** Reduce JH weight to 0.01 to let PP loss remain dominant while still providing some JH guidance.

---

### Experiment 9: Reduced JH Weight (comb_loss5)

**Hypothesis:** Reducing JH weight from 0.1 to 0.01 (10× lower than PP) might allow PP loss to remain dominant while JH provides mild guidance.

**Configuration:**
- Loss = 1.0×MSE + 0.01×JH_loss + 0.1×PP_loss
- Model: d_model=128, d_ff=512

**Results:**
- Signal R² = 0.883 (similar to comb_loss4)
- Jump Height Median AE = 0.192 m (↑ worse than comb_loss4's 0.126 m)
- Jump Height R² = -8.01 (↓↓ much worse than comb_loss4's -1.62)
- Jump Height Bias = -0.279 m (stronger underprediction)
- Peak Power R² = 0.31 (↑ from comb_loss4's 0.21, but still ↓ from comb_loss3's 0.64)
- Peak Power Bias = +5.73 W/kg (still over-predicting)
- Extreme outliers worse: predictions down to -2.7 m

**Observation:** Reducing JH weight made things worse, not better. The JH outliers became more extreme (errors up to -2.8 m vs -1.6 m in comb_loss4), and overall JH R² degraded dramatically.

**Inference:** The jump height loss component is fundamentally problematic for this architecture:

1. **Unstable gradients**: Even at 0.01 weight, JH loss introduces instability. The double integration amplifies small errors, creating erratic gradients.

2. **Conflicting with PP**: When JH weight is low enough that PP can "win", the residual JH gradients act as noise rather than guidance, destabilizing training.

3. **No sweet spot**: At high weight (0.1), JH dominates and hurts PP. At low weight (0.01), JH destabilizes without benefit. There appears to be no weight where JH helps.

**Conclusion:** Jump height loss should be abandoned. The best configuration remains **comb_loss3** (MSE + 0.1×PP only) with Signal R² = 0.933 and PP R² = 0.64. Jump height prediction must rely on accurate signal reconstruction rather than direct optimization.

---

### Experiment 10: Temporally Weighted MSE (weighted_1)

**Hypothesis:** Weighting MSE by the second derivative of ACC (jerk) would emphasize biomechanically important regions (countermovement, propulsion) over quiet standing, potentially improving prediction of derived metrics.

**Configuration:**
- Loss = Temporally Weighted MSE (weights from |d²ACC/dt²|, globally averaged)
- Model: d_model=128, d_ff=512
- Training ran full 100 epochs without early stopping

**Results:**
- Signal R² = 0.908 (↓ from comb_loss3's 0.933)
- Signal RMSE = 0.127 BW
- Jump Height Median AE = 0.160 m (↓ better than comb_loss3's 0.251 m)
- Jump Height R² = -2.21, Bias = -0.145 m (underpredicting)
- Peak Power R² = -0.14 (↓↓ much worse than comb_loss3's 0.64)
- Peak Power Bias = +8.94 W/kg (strong over-prediction)

**Observation:** Bland-Altman and scatter plots showed systematic biases in both JH (underpredicting) and PP (overpredicting). Training continued for all 100 epochs without early stopping, suggesting a different loss landscape.

**Inference:** The temporal weighting did not help overall:

1. **PP degradation**: By de-emphasizing quiet standing regions, the model may have lost important baseline/offset information that affects velocity integration for peak power.

2. **Systematic biases**: The weighting scheme creates biases rather than reducing them, suggesting the jerk-based weights don't align well with what matters for biomechanics metrics.

3. **No early stopping**: The model kept improving on the weighted loss but this didn't translate to better biomechanics predictions—the weighted loss optimizes for something different than what we care about.

**Conclusion:** Temporally weighted MSE does not improve results. The standard MSE with PP auxiliary loss (comb_loss3) remains the best approach.

---

## Key Findings

### 1. Triaxial > Resultant
Using all three accelerometer axes provides directional information that improves vGRF prediction significantly.

### 2. Signal Metrics ≠ Biomechanics Metrics
A model can achieve R² > 0.9 for signal reconstruction while still poorly predicting derived metrics. The aggregate MSE doesn't capture whether critical features (propulsion peak, timing) are accurate.

### 3. Pure Biomechanics Loss Causes Collapse
Training only on jump height (or likely peak power alone) produces degenerate solutions. The model finds signals that minimize the metric without learning realistic GRF dynamics.

### 4. Combined Loss Works Best
Keeping MSE as the primary loss with a small biomechanics component (0.1 weight) balances signal reconstruction with metric optimization.

### 5. Peak Power > Jump Height as Auxiliary Loss
Peak power focuses on a localized feature (the propulsion peak) while jump height depends on accurate integration over time. The localized nature of peak power provides more stable training signal.

### 6. Jump Height Loss is Detrimental
Adding jump height loss at any weight (0.01 to 0.1) degrades performance. Even small JH weights introduce unstable gradients from double integration, conflicting with PP optimization. There is no beneficial weight for JH loss—it should be excluded entirely.

### 7. MLP Outperforms Transformer
A simple MLP (single hidden layer) achieves better results than the transformer with 10× fewer parameters. The attention mechanism provides no benefit for this mapping task.

### 8. FPC→FPC is the Winning Representation
FPC input/output with a simple MLP achieves the first positive JH R² (0.58) and best PP R² (0.62). The data-driven, variance-ordered FPC basis captures biomechanically relevant features that B-spline and raw representations miss.

---

## Recommended Configuration

### For Raw Signal Training (No FDA Transforms)

Best configuration for raw signal training uses triaxial input with PP auxiliary loss:

```bash
python src/train.py \
    --use-triaxial \
    --d-model 128 \
    --d-ff 512 \
    --loss combined \
    --mse-weight 1.0 \
    --jh-weight 0.0 \
    --pp-weight 0.1 \
    --epochs 100
```

**Results:** Signal R² = 0.93, PP R² = 0.64, JH R² = -2.28

### For FPC Transform Training (Current Best)

**IMPORTANT:** FPC transforms require **resultant** acceleration, NOT triaxial. Triaxial + FPC causes severe performance degradation (JH R² drops from 0.34 to 0.10).

```bash
python src/train.py \
    --input-transform fpc --output-transform fpc \
    --fixed-components --n-components 15 \
    --no-varimax \
    --loss eigenvalue_weighted \
    --epochs 100
```

**Results:** Signal R² = 0.91, JH R² = 0.34, PP R² = 0.33

### ~~Experimental: Weighted MSE~~ (Not Recommended)

The `--loss weighted` option was tested but degraded performance—see Experiment 10. It introduced biases and hurt peak power prediction. Use `--loss combined` with PP weight instead.

---

## Next Steps

1. ~~**Combine JH and PP losses**~~: Tested at multiple weights (0.1, 0.01)—JH loss is detrimental at any weight. Abandoned.

2. ~~**Temporally weighted MSE**~~: Tested `--loss weighted`—degraded PP prediction (R² -0.14) and introduced biases. Abandoned.

3. ~~**Triaxial with FPC**~~: Initially caused degradation with transformer, but **works excellently with MLP** (JH R² = 0.82). Triaxial + FPC + MLP is now the best configuration.

4. ~~**Velocity target representation**~~: Tested predicting velocity (single integration of GRF) instead of force, recovering GRF via differentiation. Velocity R² = 0.90 but differentiation destroyed GRF quality (BW R² = 0.47, PP R² = -0.98). Abandoned — coaches need accurate GRF curves.

5. ~~**Hybrid linear projection + MLP**~~: Implemented MATLAB-style eigenfunction projection with MLP refinement. Discovered eigenfunction inner products don't work (R² = -6); learned projection via Ridge regression achieves R² = 0.54. Hybrid sequential architecture improves to JH R² = 0.76, but still worse than simple MLP (0.82). Abandoned for performance, though useful for interpretability.

6. **Investigate outliers**: The outlier diagnostic plots may reveal common patterns in problematic samples.

7. **Sequence length**: Current 500 samples may truncate important context. Try 800 samples.

8. **Architecture variants**: Consider temporal convolutional networks (TCN) for comparison.

9. **Cross-validation**: Implement k-fold CV at subject level for more robust evaluation.

10. ~~**Custom discrete FPCA**~~: No longer needed — triaxial FPC with MLP achieves JH R² = 0.82, surpassing the original MATLAB results.

---

## FDA Transformation Experiments

Functional Data Analysis (FDA) approaches to address the parameter-to-sample ratio (~750K parameters, 896 training samples). These transformations enforce smoothness constraints appropriate for continuous biomechanical signals.

### Motivation

The current model treats each of 500 time points independently. FDA representations:
- Compress signals while enforcing smoothness
- Reduce effective dimensionality
- May improve generalization with limited training data

### ⚠️ Important: Triaxial + FPC Incompatibility

**Triaxial input does NOT work with FPC transforms.** All successful FPC experiments used resultant acceleration.

| Configuration | JH R² | PP R² | Notes |
|---------------|-------|-------|-------|
| FPC + Resultant | 0.23 | 0.24 | Baseline FPC |
| FPC + Triaxial | -0.11 | -0.13 | Complete failure |
| FPC + Resultant + Eigenvalue-weighted | **0.34** | **0.33** | Current best |
| FPC + Triaxial + Eigenvalue-weighted | 0.11 | 0.00 | Still fails |

**Why triaxial fails with FPC:**
- Triaxial creates 15 FPCs × 3 channels = 45 input features vs 15 for resultant
- FPCA treats each axis independently—model must learn which components from which channels matter
- Horizontal axes (x, y) contain less GRF-relevant information but still contribute FPCs

### Configurations Tested (Original Custom FPCA Implementation)

**⚠️ WARNING:** The results below were achieved with a custom FPCA implementation using discrete dot products. After refactoring to scikit-fda (which uses L² inner products), these results are **NOT reproducible**. See "scikit-fda Refactoring" section below.

| Run | Input | Output | Loss | Signal R² | JH Median AE | JH R² | PP R² | Notes |
|-----|-------|--------|------|-----------|--------------|-------|-------|-------|
| baseline | raw | raw | MSE | 0.919 | 0.230 m | -1.91 | 0.29 | Resultant ACC, d=64 |
| smooth_0.1 | raw | raw | Smooth λ=0.1 | 0.903 | 0.455 m | -7.23 | 0.40 | PP bias eliminated, JH worse |
| smooth_0.2 | raw | raw | Smooth λ=0.2 | 0.937 | **0.205 m** | -1.26 | **0.50** | Best JH & PP biomechanics |
| bspline_15 | bspline | bspline | MSE | 0.946 | 0.508 m | -10.48 | -0.22 | Too few basis functions |
| bspline_30 | bspline | bspline | MSE | 0.949 | 0.441 m | -6.84 | 0.33 | Best signal R², sweet spot |
| bspline_60 | bspline | bspline | MSE | 0.931 | 0.546 m | -12.04 | 0.02 | Too many basis functions |
| ~~fpc_15~~ | fpc | fpc | MSE | ~~0.949~~ | ~~**0.053 m**~~ | ~~**0.61**~~ | ~~**0.65**~~ | ~~Custom FPCA, NOT reproducible~~ |
| ~~fpc_15_novar~~ | fpc | fpc | MSE | ~~**0.953**~~ | ~~0.060 m~~ | ~~0.59~~ | ~~0.63~~ | ~~Custom FPCA, NOT reproducible~~ |
| ~~fpc_25_novar~~ | fpc | fpc | MSE | ~~0.948~~ | ~~0.068 m~~ | ~~0.56~~ | ~~0.61~~ | ~~Custom FPCA, NOT reproducible~~ |
| ~~fpc_15_large~~ | fpc | fpc | MSE | ~~0.955~~ | ~~0.050 m~~ | ~~0.62~~ | ~~0.66~~ | ~~Custom FPCA, NOT reproducible~~ |

### Current Reproducible FPC Results (scikit-fda)

| Run | Input | Loss | Signal R² | JH Median AE | JH R² | PP R² | Notes |
|-----|-------|------|-----------|--------------|-------|-------|-------|
| fpc_no_varimax | resultant | MSE | 0.86 | 0.105 m | 0.23 | 0.24 | Baseline with scikit-fda |
| eigenvalue_weighted | resultant | Eigenvalue MSE | **0.91** | **0.089 m** | **0.34** | **0.33** | Current best |
| signal_space | resultant | Signal-space MSE | 0.91 | 0.098 m | 0.18 | 0.18 | Did not help |

### Detailed Results

#### FDA Baseline (raw/raw, resultant)

**Configuration:**
- Input: Resultant acceleration (1 channel)
- Model: d_model=64, num_heads=4, num_layers=3, d_ff=128
- Loss: MSE
- Transforms: None (raw/raw)

**Results:**
```
Signal Metrics:
  RMSE: 0.2805 (normalized), 0.118 BW
  R²:   0.919

Reference (500ms curve vs full signal ground truth):
  JH RMSE: 0.035 m, R² = 0.948
  PP RMSE: 0.52 W/kg, R² = 0.998

Predicted vs Actual (from 500ms curves):
  JH: RMSE 0.280 m, Median AE 0.230 m, R² = -1.91, Bias = -0.196 m
  PP: RMSE 9.86 W/kg, Median AE 5.67 W/kg, R² = 0.29, Bias = +3.83 W/kg
  Valid samples: 238/240 (2 with negative JH)
```

**Observation:** Good signal reconstruction but systematic underprediction of jump height and overprediction of peak power. This baseline uses the smaller model (d=64) with resultant acceleration for a cleaner comparison of FDA effects.

---

## scikit-fda Refactoring Investigation

### ⚠️ Major Regression: Performance Never Recovered

The original FDA transformations used custom implementations that achieved excellent results (JH R² ≈ 0.61). After refactoring to use the scikit-fda library, **performance dropped to JH R² ≈ 0.34 and was never recovered** despite extensive investigation and normalization fixes.

This regression represents the single largest performance loss in the project. The best current FPC results are approximately 45% worse than what was achieved with the custom implementation.

### Motivation for Refactoring

- Use well-tested, maintained library code
- Benefit from scikit-fda's optimized implementations
- Easier maintenance and future extensions

### Implementation Changes

The refactoring replaced custom B-spline and FPCA implementations with scikit-fda equivalents:

```python
# Old: Custom basis matrix and penalty construction
# New: scikit-fda BSplineBasis, BasisSmoother, FPCA

from skfda import FDataGrid
from skfda.representation.basis import BSplineBasis, FDataBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.dim_reduction import FPCA
```

**Critical difference:** scikit-fda uses **L² inner products** (numerical integration over the functional domain) rather than **discrete dot products** (sum over sample points). This fundamentally changes how eigenfunctions are computed and weighted.

### Post-Refactoring Performance

After refactoring to scikit-fda, performance degraded significantly and **permanently**:

| Configuration | Signal R² | JH R² | JH Median AE | Notes |
|---------------|-----------|-------|--------------|-------|
| Original (custom FPCA) | 0.949 | 0.61 | 0.053 m | Pre-refactoring baseline |
| scikit-fda + varimax | 0.85 | -0.01 | 0.126 m | Initial refactoring |
| scikit-fda + no varimax | 0.91 | 0.30 | 0.092 m | Best post-refactoring |

### Investigation: Varimax Rotation

**Finding:** Varimax rotation hurts performance with scikit-fda.

| Configuration | Signal R² | JH R² | PP R² |
|---------------|-----------|-------|-------|
| With varimax | 0.85 | -0.01 | 0.18 |
| Without varimax | 0.91 | 0.30 | 0.33 |

**Hypothesis:** The varimax rotation interacts poorly with scikit-fda's L² inner product framework. The rotation is computed on loadings derived from L² eigenfunctions, but applying it to scores may not preserve the same properties as in the discrete case.

**Recommendation:** Disable varimax with `--no-varimax` when using scikit-fda FPCA.

### Investigation: ACC Outlier Filtering

Initial debugging revealed extreme ACC values (max 1144g in one sample) that appeared to be sensor artifacts. An outlier filter was implemented.

**Finding:** Aggressive outlier filtering (10g threshold) hurt performance.

| Configuration | Samples Excluded | Signal R² | JH R² |
|---------------|------------------|-----------|-------|
| No filtering | 0 | 0.91 | 0.30 |
| 10g threshold | 62 (5.5%) | 0.86 | 0.23 |

**Analysis:** The 10g threshold excluded valid samples that had brief high accelerations during landing/impact phases. These samples contained useful training information.

**Recommendation:** Keep outlier filtering disabled (default `--acc-max-threshold None`).

### Investigation: Normalization Pipeline

The data pipeline applies z-score normalization before FPCA:

```
Raw signals → Z-score normalize → FPCA → Model → Inverse FPCA → Denormalize
```

This was investigated because:
1. Standard FDA practice (Ramsay & Silverman) applies FPCA to raw signals
2. FPCA handles centering internally via mean function subtraction
3. Pre-normalization might distort the functional structure

**Experiment: FPCA on Raw (Unnormalized) Signals**

| Configuration | Signal R² | JH R² | JH Median AE |
|---------------|-----------|-------|--------------|
| Pre-normalized + FPCA | **0.91** | **0.30** | **0.092 m** |
| Raw + FPCA | 0.79 | 0.008 | 0.105 m |
| Raw + FPCA + score standardization | 0.80 | 0.02 | 0.105 m |

**Counterintuitive Finding:** Pre-normalizing signals before FPCA works significantly better than the theoretically "correct" approach of applying FPCA to raw signals.

**Analysis of Why Pre-Normalization Helps:**

1. **FPC Score Scale:**
   - With pre-normalization: scores have std ≈ 0.12, range [-0.96, 0.84]
   - Without pre-normalization: scores have std ≈ 0.05, range [-0.39, 0.34]
   - Smaller scores are harder for the model to learn with precision

2. **L² Inner Products:**
   scikit-fda's FPCA uses L² inner products computed via numerical integration:
   ```
   ⟨f, g⟩_L² = ∫ f(t) g(t) dt
   ```
   This weights all time points by the integration rule. The quiet standing phase (long duration, low variance) dominates the propulsion phase (short duration, high variance).

3. **Discrete vs. Continuous:**
   The original MATLAB implementation likely used discrete dot products:
   ```
   ⟨f, g⟩_discrete = Σ f(tᵢ) g(tᵢ)
   ```
   Pre-normalization makes L² inner products behave more like discrete dot products by equalizing variance across the signal.

4. **Mean Function Effect:**
   - Raw GRF has mean ≈ 1.1 BW across all time points (dominated by quiet standing)
   - After FPCA centering, most variance comes from deviations around 1.1 BW
   - Pre-normalization creates mean ≈ 0, making deviations more prominent

**Recommendation:** Keep pre-normalization in the pipeline. The current best configuration uses:
- Z-score normalization before FPCA
- No varimax rotation
- No outlier filtering

### Current Best Configuration (Post-Refactoring)

After extensive investigation, the best achievable results with scikit-fda use eigenvalue-weighted loss:

```bash
python src/train.py \
    --input-transform fpc --output-transform fpc \
    --fixed-components --n-components 15 \
    --no-varimax \
    --loss eigenvalue_weighted \
    --epochs 100
```

**Results:**
- Signal R² = 0.91
- JH R² = 0.34
- JH Median AE = 0.089 m
- PP R² = 0.33

**Note:** Do NOT use `--use-triaxial` with FPC transforms. It causes severe degradation.

### Performance Gap Analysis

The scikit-fda implementation achieves JH R² ≈ 0.34 vs. ~0.61 with the original custom code—a **45% regression**. The gap is due to:

1. **L² vs. Discrete Inner Products:** scikit-fda's continuous framework computes different eigenfunctions than discrete implementations. The L² inner product weights time points by the integration rule, not equally.

2. **Eigenfunction Shape:** Different numerical methods yield slightly different eigenfunctions that capture variance differently.

3. **Centering Approach:** scikit-fda centers using L² mean, while discrete implementations use pointwise mean.

4. **Score Scaling:** The resulting FPC scores have different scales and distributions, affecting what the neural network learns.

### Future Work

To potentially recover the original performance:

1. **Custom Discrete FPCA:** Re-implement FPCA using discrete dot products to match the original behavior. This is the most likely path to recovering performance.

2. **Weighted L² Inner Products:** Modify scikit-fda to use weighted integration that emphasizes the propulsion phase over quiet standing.

3. **Alternative Libraries:** Investigate other FDA libraries (R's fda package via rpy2, or custom NumPy implementation).

---

## Custom vs scikit-fda FPCA Comparison

### Investigation: What Actually Changed?

The original custom FPCA code was restored from git history (commit `07b7425`) and compared directly against the scikit-fda implementation using `src/compare_fpca.py`.

### Key Finding: Score Scaling, Not Algorithm

Surprisingly, the **eigenfunctions and variance explained are identical** between implementations:

| Property | Custom | scikit-fda | Difference |
|----------|--------|------------|------------|
| Variance explained (FPC1) | 44.49% | 44.49% | None |
| Variance explained (cumulative, 15 FPCs) | 98.0% | 98.0% | None |
| Eigenfunction correlation | 1.0 | 1.0 | Identical |
| Mean function | Identical | Identical | None |
| Reconstruction RMSE | 0.0435 | 0.0435 | None |

The fundamental difference is in **eigenfunction normalization**:

| Metric | Custom (Discrete) | scikit-fda (L²) | Ratio |
|--------|-------------------|-----------------|-------|
| Score std | 2.79 | 0.12 | **22x** |
| Score range | [-22, +19] | [-0.9, +1.0] | **22x** |
| FPC1 score std | 7.23 | 0.32 | **22x** |

### Root Cause: Normalization Convention

The ~22x difference (≈√500) comes from different eigenfunction normalization:

**Custom (Discrete Norm):**
```
Σᵢ |φ(tᵢ)|² = 1  (sum over 500 time points)
```

**scikit-fda (L² Norm):**
```
∫ |φ(t)|² dt = 1  (integral over domain [0,1])
```

Since numerical integration approximates `∫ f dt ≈ (1/n) Σ fᵢ` for n=500 points, the L² norm is 1/500 of the discrete norm. This makes scikit-fda eigenfunctions √500 ≈ 22x smaller, and correspondingly the scores are 22x larger in the custom version.

### Impact on Neural Network Training

This 22x difference in target score magnitudes fundamentally changes the training dynamics:

1. **Loss magnitude:** MSE loss is ~500x smaller with scikit-fda scores (22² ≈ 500)
2. **Gradient magnitude:** Gradients are proportionally smaller
3. **Learning rate sensitivity:** The effective learning rate is different
4. **Eigenvalue weighting:** The eigenvalue-weighted loss behaves differently when scores are scaled

### Implications

The performance regression may not be due to algorithmic differences (L² vs discrete inner products for covariance computation), but rather due to the **score scaling** affecting the neural network's ability to learn.

### Experimental Validation (January 2026)

Two approaches were tested to verify whether score scaling explains the performance difference:

**Option A: Scale scikit-fda scores by √500**
```bash
python src/train.py --input-transform fpc --output-transform fpc \
    --fixed-components --n-components 15 --no-varimax \
    --loss eigenvalue_weighted --score-scale auto --epochs 100
```

**Option C: Use custom FPCA directly**
```bash
python src/train.py --input-transform fpc --output-transform fpc \
    --fixed-components --n-components 15 --no-varimax \
    --loss eigenvalue_weighted --use-custom-fpca --epochs 100
```

**Results:**

| Configuration | JH R² | PP R² | Transformed R² |
|---------------|-------|-------|----------------|
| Baseline (scikit-fda) - Run 1 | 0.31 | 0.29 | 0.56 |
| Baseline (scikit-fda) - Run 2 | 0.22 | 0.28 | 0.51 |
| Option A (scaled scores) | **0.09** | 0.13 | 0.12 |
| Option C (custom FPCA) | **-0.16** | 0.06 | 0.09 |

**Conclusions:**

Both scaling approaches made performance **significantly worse**, not better:
- Option A degraded JH R² from ~0.27 to 0.09
- Option C produced negative JH R² (worse than predicting the mean)

This suggests the model hyperparameters (learning rate, architecture, loss weighting) were implicitly tuned for scikit-fda's smaller score magnitudes. The original "good" custom FPCA results (JH R² ≈ 0.61) were achieved with different hyperparameters.

**Important observations:**
1. Baseline runs show ~30% variability in JH R² between identical configurations
2. The score scaling hypothesis alone does not explain the performance regression
3. Further investigation needed: hyperparameter sensitivity, random seed effects

### Files

- `src/transformations_custom.py`: Restored custom FPCA implementation
- `src/compare_fpca.py`: Comparison script that produced these results

---

## Mean Function Normalization Investigation

The standard FDA approach (Ramsay & Silverman) centers data by subtracting the mean function (average curve across samples at each time point) rather than a global scalar mean. This section documents the implementation and evaluation of proper mean function normalization.

### Motivation

The original implementation used **scalar z-score normalization**:
```
normalized = (signal - global_mean) / global_std
```

This subtracts a single scalar value from all time points, which:
- Doesn't account for the characteristic shape of the signal
- May conflate centering with variance normalization in ways that affect FPCA

The FDA-standard approach uses **mean function normalization**:
```
normalized = (signal - mean_function) / global_std
```

Where `mean_function[t]` is the average across all samples at time point `t`.

### Implementation

Added robust mean function computation with outlier clipping to handle sensor artifacts:

```python
def _compute_robust_mean_function(data, clip_threshold=5.0):
    """Compute mean function by clipping outliers at each time point."""
    mean_function = np.zeros((seq_len, n_channels))
    for ch in range(n_channels):
        for t in range(seq_len):
            values = data[:, t, ch]
            mu, sigma = np.mean(values), np.std(values)
            clipped = np.clip(values, mu - 5*sigma, mu + 5*sigma)
            mean_function[t, ch] = np.mean(clipped)
    return mean_function
```

Key design choices:
- **5σ clipping**: Conservative threshold to remove only extreme outliers while retaining valid data
- **Pointwise clipping**: Applied at each time point separately, allowing different thresholds across the signal
- **All samples retained**: Outlier clipping affects mean computation only; all samples remain in training

### Results

| Configuration | Signal R² (BW) | JH R² | JH Median AE | JH Bias | PP R² |
|---------------|----------------|-------|--------------|---------|-------|
| Scalar normalization | 0.804 | 0.024 | 0.105 m | -0.012 m | -0.023 |
| Mean function normalization | 0.792 | 0.003 | 0.111 m | **0.002 m** | 0.0001 |

### Analysis

**What Improved:**
- **Bias nearly eliminated**: JH bias reduced from -0.012 m to 0.002 m (essentially zero)
- The model no longer systematically underpredicts jump height

**What Degraded:**
- Signal R² dropped slightly (0.804 → 0.792)
- JH R² dropped dramatically (0.024 → 0.003)
- PP R² dropped (essentially zero correlation)
- Normalized z-score Signal R² collapsed to 0.002

**Why Mean Function Normalization Hurt Performance:**

1. **FPCA Re-centering Conflict:**
   scikit-fda's FPCA performs its own mean function subtraction. With scalar normalization, the data has mean ≈ 0 globally but retains its characteristic shape. With mean function normalization, the data is already centered around zero at each time point, so FPCA's centering has little effect—but the eigenfunctions learned may be different.

2. **Information Removal:**
   The mean function contains useful information about the typical GRF profile. Subtracting it removes this "baseline shape" that the model may have been using to anchor predictions.

3. **Variance Structure Changed:**
   After mean function subtraction, variance is more uniform across time points. This changes which eigenfunctions capture the most variance, potentially emphasizing different signal features than the scalar-normalized version.

4. **L² Inner Product Interaction:**
   scikit-fda's L² inner products already down-weight high-variance regions through integration. Mean function normalization further homogenizes the signal, potentially removing the variance structure that helped the model distinguish between samples.

### Conclusion

Mean function normalization reduces systematic bias but degrades overall predictive accuracy with the current scikit-fda FPCA implementation. The scalar normalization approach, while theoretically "incorrect" for FDA, works better in practice because:

1. It preserves variance structure that helps distinguish samples
2. It interacts more favorably with scikit-fda's L² inner products
3. The model may implicitly learn the mean function through training

**Recommendation:** Retain scalar z-score normalization. The bias can potentially be addressed through:
- Post-hoc bias correction on predictions
- Including bias terms in the model architecture
- Alternative FPCA implementations that don't use L² inner products

---

## Robust Normalization Fix

### Problem: Corrupted Sample Destroying Normalization

Investigation revealed that **Sample 920** contained catastrophically corrupted ACC data:
- All values ranged from **-1130 to -1140g** instead of ~1g
- This single sample inflated the global std from 0.24 to **21.75** (90× larger)
- FPC scores were compressed to range [-0.01, 0.06] instead of [-1, 1]
- The original mean/std-based robust clipping failed because outliers corrupted the mean and std used for clipping bounds

### Solution: Robust Median/MAD Normalization

Implemented two fixes:

1. **Default `acc_max_threshold=100g`**: Automatically filters catastrophically corrupted samples while preserving legitimate high-impact data.

2. **Median/MAD-based robust statistics**:
   - Mean function computed using median and MAD (Median Absolute Deviation) instead of mean/std
   - MAD scaled by 1.4826 to estimate std for normal distributions
   - Global std also computed using MAD for robustness
   - Outliers identified as values beyond 5× robust std from median

### Results After Fix

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| ACC std | 21.75 | 0.12 |
| FPC score range | [-0.01, 0.06] | [-1.9, 2.4] |
| FPC score std | 0.009 | 0.46 |
| Resultant ACC at start | 2.0 ± 33.8 | 0.97 ± 0.02 |

The pipeline now correctly normalizes signals and produces FPC scores in the expected range.

---

## Varimax Rotation Analysis (Post-Fix)

With the robust normalization fix in place, re-evaluated varimax rotation impact.

### Configuration
```bash
python src/train.py \
    --use-triaxial \
    --input-transform fpc --output-transform fpc \
    --fixed-components --n-components 15 \
    [--no-varimax or default varimax] \
    --epochs 100
```

### Results Comparison

| Metric | With Varimax | Without Varimax |
|--------|-------------|-----------------|
| Transformed R² | 0.559 | **0.601** |
| Signal R² (BW) | 0.908 | **0.917** |
| JH R² | 0.080 | **0.216** |
| JH Median AE | 0.106 m | **0.097 m** |
| PP R² | -0.011 | **0.192** |
| PP Median AE | 9.09 W/kg | **7.06 W/kg** |

### Analysis: Why No-Varimax Performs Better

The MSE loss in FPC space treats all components equally, but their importance for reconstruction differs dramatically:

**Without varimax (standard FPCA):**
- FPC1 explains ~60-70% of variance → large scores → large contribution to MSE
- FPC2 explains ~15-20% → medium scores
- Later FPCs explain progressively less → small scores
- **MSE naturally weights important components more heavily**

**With varimax:**
- Variance redistributed more evenly across components
- All components contribute similarly to MSE
- Model "wastes effort" on components that barely affect reconstruction
- **Loss function misaligned with reconstruction importance**

### Recommendation

Use `--no-varimax` for FPC transforms. The standard FPCA ordering (by variance explained) creates implicit loss weighting that aligns with reconstruction quality.

---

## Eigenvalue-Weighted Loss Experiment

### Motivation

Standard MSE in FPC space treats all components equally, but earlier components explain more variance. Even without varimax, the implicit weighting from score magnitudes may not be optimal. Explicitly weighting by eigenvalues should focus the model on the most important components.

### Implementation

Added `--loss eigenvalue_weighted` option that weights each FPC component's squared error by its variance explained ratio (normalized to sum to 1):

```python
loss = Σ (eigenvalue[i] / Σeigenvalues) × (pred[i] - actual[i])²
```

For GRF with 15 components, FPC1 (~70% variance) gets ~0.70 weight, FPC2 (~15%) gets ~0.15, etc.

### Configuration

```bash
python src/train.py \
    --use-triaxial \
    --input-transform fpc --output-transform fpc \
    --fixed-components --n-components 15 \
    --no-varimax \
    --loss eigenvalue_weighted \
    --epochs 100
```

### Results Comparison

| Metric | Standard MSE | Eigenvalue-Weighted |
|--------|-------------|---------------------|
| Signal R² (BW) | **0.917** | 0.908 |
| JH R² | 0.216 | **0.343** |
| JH Median AE | 0.097 m | **0.089 m** |
| JH Bias | -0.018 m | **-0.005 m** |
| PP R² | 0.192 | **0.326** |
| PP Median AE | 7.06 W/kg | **6.50 W/kg** |
| PP Bias | -1.67 W/kg | **-0.75 W/kg** |

### Analysis

Eigenvalue-weighted loss substantially improves biomechanics metrics:
- **JH R² increased 59%**: 0.216 → 0.343
- **PP R² increased 70%**: 0.192 → 0.326
- **Bias nearly eliminated**: JH bias reduced from -18mm to -5mm

Trade-off: Signal R² slightly decreased (0.917 → 0.908), but this is acceptable given the biomechanics improvements.

### Why It Works

By explicitly weighting FPC1 at ~70% of the loss, the model focuses on accurately predicting the dominant mode of variation (overall GRF magnitude and shape). Errors in later components that contribute little to reconstruction quality are down-weighted.

### Remaining Gap

Despite improvements, JH R² of 0.34 and PP R² of 0.33 are still far from ideal. The reference comparison shows that even the **actual 500ms curves** achieve JH R² = 0.87 and PP R² = 0.99 against ground truth, indicating the model's signal predictions still miss critical features.

---

## Signal-Space Loss Experiment

### Motivation

Both standard MSE and eigenvalue-weighted loss operate in FPC score space. Even with eigenvalue weighting, errors in FPC scores may not correspond directly to reconstruction errors. Signal-space loss computes MSE on the reconstructed signals after inverse FPCA transform, directly optimizing for signal quality.

### Implementation

Added `--loss signal_space` option that:
1. Inverse transforms predicted FPC scores through the FPCA pipeline (varimax reversal, score unstandardization, eigenfunction reconstruction)
2. Inverse transforms actual FPC scores the same way
3. Computes MSE on the reconstructed signals

The inverse transform is implemented in TensorFlow to maintain gradient flow:
```python
# Reconstruct signal from FPC scores
signal = mean_function + Σ score[i] × eigenfunction[i]
```

### Configuration

```bash
python src/train.py \
    --use-triaxial \
    --input-transform fpc --output-transform fpc \
    --fixed-components --n-components 15 \
    --no-varimax \
    --loss signal_space \
    --epochs 100
```

### Results Comparison

| Metric | Standard MSE | Eigenvalue-Weighted | Signal-Space |
|--------|-------------|---------------------|--------------|
| Transformed R² | 0.601 | 0.587 | **0.587** |
| Signal R² (BW) | **0.917** | 0.908 | 0.914 |
| JH R² | 0.216 | **0.343** | 0.179 |
| JH Median AE | 0.097 m | **0.089 m** | 0.098 m |
| JH Bias | -0.018 m | **-0.005 m** | -0.019 m |
| PP R² | 0.192 | **0.326** | 0.180 |
| PP Median AE | 7.06 W/kg | **6.50 W/kg** | 7.38 W/kg |
| PP Bias | -1.67 W/kg | **-0.75 W/kg** | -1.78 W/kg |

### Analysis

Signal-space loss performed **worse** than eigenvalue-weighted loss and similar to (or slightly worse than) standard MSE:
- JH R² = 0.179 (vs 0.343 eigenvalue-weighted, 0.216 standard)
- PP R² = 0.180 (vs 0.326 eigenvalue-weighted, 0.192 standard)

### Why Signal-Space Loss Didn't Help

1. **Gradient Dilution**: When computing loss on 500 time points instead of 15 FPC scores, gradients are spread across more dimensions. The model receives weaker signal about which FPC components to improve.

2. **Eigenfunction Dominance**: FPC1 contributes ~70% of the reconstructed signal. Signal-space MSE is dominated by FPC1 reconstruction errors, effectively just another form of implicit weighting—but with more numerical instability.

3. **Uniform Time Weighting**: Signal-space MSE weights all time points equally. The quiet standing phase (300+ samples) dominates the propulsion phase (~50 samples). This is the opposite of what we need for biomechanics.

4. **Loss Landscape**: The mapping from FPC scores to signal space is linear, so signal-space loss is a quadratic form in FPC scores. This is mathematically equivalent to a weighted MSE in FPC space, but with weights determined by eigenfunction magnitudes—which may not align with biomechanical importance.

### Conclusion

**Eigenvalue-weighted loss remains the best approach** for FPC-space training. The explicit weighting by variance explained focuses the model on the most important components without the gradient dilution of signal-space loss.

The improvement from eigenvalue weighting (JH R² 0.22 → 0.34) shows that loss function design matters, but further gains likely require:
- Better representation of the propulsion phase
- Alternative architectures that capture temporal dependencies differently
- Hybrid losses that combine FPC-space and biomechanics objectives

---

## Simple Normalization Discovery (January 2026)

### The Problem

Training with raw signals or B-spline transforms produced terrible results (JH R² = -4.7, Signal R² = 0.65). Investigation revealed the MAD-based normalization was creating extreme values.

### Root Cause

The robust normalization pipeline used:
1. Mean function centering (median across samples at each time point)
2. MAD-based std estimation on the centered residuals

This produced `grf_std = 0.05` (very small), resulting in normalized values ranging from **-27 to +52** instead of the typical ±3 range for z-scores.

The MAD was small because most time points are quiet standing (near zero after centering), but propulsion has large deviations that MAD treats as "outliers."

### Solution

Added `--simple-normalization` flag that uses global z-score:
```python
mean = np.mean(data)   # scalar
std = np.std(data)     # ~0.42 for GRF
normalized = (data - mean) / std  # range: ~[-3, +6]
```

### Results

| Configuration | Signal R² | JH R² | PP R² |
|---------------|-----------|-------|-------|
| Raw + MAD norm | 0.65 | -4.74 | 0.32 |
| Raw + simple norm | 0.93 | -1.69 | - |
| B-spline + simple norm | 0.93 | -2.05 | 0.28 |

Simple normalization dramatically improved signal prediction, though JH R² remained negative.

---

## ReconstructionLoss Experiment (January 2026)

### Motivation

With transforms (B-spline, FPC), the model predicts coefficients, not signals. Computing loss in coefficient space may not optimize signal reconstruction quality. ReconstructionLoss computes MSE after inverse-transforming predictions back to signal space.

### Implementation

```python
class ReconstructionLoss:
    def __init__(self, reconstruction_matrix, mean_function=None, temporal_weights=None):
        # For B-spline: signal = basis_matrix @ coefficients
        # For FPC: signal = mean + eigenfunctions @ scores

    def call(self, y_true, y_pred):
        signal_true = self._reconstruct(y_true)
        signal_pred = self._reconstruct(y_pred)
        error = (signal_true - signal_pred) ** 2
        if temporal_weights:
            error = error * temporal_weights
        return mean(error)
```

### Temporal Weighting

Jerk-based weights derived from ACC second derivative:
- Quiet standing: low jerk → low weight (~0.3)
- Propulsion phase: high jerk → high weight (~4.0)
- Weights normalized to sum to 1.0

**Key finding**: Propulsion-only weighting made results worse because errors in early signal compound through integration. Jerk-based weights performed better.

### Results

| Configuration | Signal R² | JH R² | PP R² |
|---------------|-----------|-------|-------|
| B-spline + MSE | 0.91 | -2.05 | 0.28 |
| B-spline + reconstruction (no weights) | 0.93 | -1.92 | - |
| B-spline + reconstruction + jerk weights | **0.94** | -2.27 | **0.49** |

### Analysis

ReconstructionLoss with jerk-based temporal weighting achieved:
- **PP R² = 0.49** - significant improvement over all previous configurations
- Signal R² = 0.94 - excellent signal reconstruction
- JH R² still negative - jump height remains challenging

The disconnect between Signal R² (0.94) and JH R² (-2.27) highlights that jump height depends on subtle signal features that affect takeoff velocity. Small errors in the propulsion phase compound through double integration.

### Files

- `src/losses.py`: ReconstructionLoss class
- `src/transformations.py`: get_reconstruction_components() for B-spline and FPC
- `src/data_loader.py`: compute_temporal_weights() with jerk-based option

---

## Velocity Target Representation (January 2026)

### Motivation

Jump height requires double integration of GRF, compounding small prediction errors. By predicting velocity (single integration of GRF) as the model's target, the learning signal should be smoother, and JH is only one integration step away from the predicted representation.

### Pipeline

```
Force mode (existing):
  GRF(BW) → normalize → [B-spline] → model targets
  model output → [inverse B-spline] → denormalize → GRF(BW) → biomechanics

Velocity mode (new):
  GRF(BW) → velocity(m/s) → normalize → [B-spline] → model targets
  model output → [inverse B-spline] → denormalize → velocity(m/s) → differentiate → GRF(BW) → biomechanics
```

Velocity computed via impulse-momentum theorem: `v(t) = g × cumsum(GRF/BW - 1) × dt`

GRF recovered via differentiation: `GRF/BW = (dv/dt) / g + 1.0`

### Configuration

```bash
python -m src.train \
    --target-representation velocity \
    --input-transform bspline --output-transform bspline \
    --loss reconstruction \
    --simple-normalization \
    --epochs 100
```

### Results

| Metric | Force mode | Velocity mode |
|--------|-----------|---------------|
| Signal R² (normalized space) | 0.94 (GRF) | 0.90 (velocity) |
| Signal R² (BW, after recovery) | 0.94 | 0.47 |
| JH R² | -2.27 | -1.12 |
| JH RMSE | — | 0.24 m |
| JH Bias | — | +0.16 m |
| PP R² | 0.49 | -0.98 |
| PP RMSE | — | 16.81 W/kg |
| PP Bias | — | +6.93 W/kg |

Velocity normalization stats: mean = 0.042 m/s, std = 0.718 m/s.

### Analysis

The velocity prediction itself was accurate (R² = 0.90), but **differentiation destroyed the recovered GRF quality**. This is the mirror image of the original problem:

1. **Integration smooths, differentiation amplifies.** A velocity curve that looks good at R² = 0.90 still has errors that, once differentiated, produce GRF curves that don't match the true shape. The information needed for accurate GRF is not preserved through the velocity→differentiate path.

2. **JH improved but PP collapsed.** JH R² improved from -2.27 to -1.12 (still negative), but PP R² collapsed from 0.49 to -0.98. The propulsion peak in the GRF — which determines peak power — is particularly sensitive to differentiation errors.

3. **No convergence beyond 5 epochs.** JH RMSE was 0.24m at both 5 epochs and 100 epochs, despite velocity R² improving from 0.84 to 0.90. The differentiation step is the bottleneck, not model accuracy.

4. **This is not a numerical issue.** Using analytical B-spline derivatives instead of `np.gradient` would not help — the errors are in the predicted coefficients, not in the differentiation method. Even small coefficient errors produce GRF curves that look wrong to coaches.

### Conclusion

**Abandoned.** The velocity target representation is fundamentally unsuitable when the GRF curve itself must be accurate, which is a requirement for coaching applications. The approach trades GRF accuracy for velocity accuracy, but coaches need to see the GRF curve. The code was reverted to avoid unnecessary complexity.

The underlying insight remains valid — JH depends on double integration and is inherently harder to predict from GRF — but the solution must work within the force domain rather than changing the target representation.

---

## Scalar-Conditioned Transformer (January 2026)

### Motivation

Jump height prediction from GRF curves remains the project's core challenge (JH R² consistently negative). A dual-branch architecture was designed to provide the GRF decoder with global context about the jump by predicting jump height as a scalar side-output, then conditioning the curve decoder with that prediction.

### Architecture

```
ACC input → Input Projection → Positional Encoding → Encoder Blocks
                                                          │
                                    ┌─────────────────────┤
                                    ↓                     ↓
                              Scalar Branch         Truncate to output_seq_len
                            x[:, -1, :]                   │
                           Dense(d_model//2, relu)        │
                           Dense(1) → JH pred             │
                                    │                     │
                                    │     Conditioning    │
                                    └──→ Dense(d_model) ──┤
                                         expand + add  ───┘
                                                          │
                                                   Output Projection
                                                   Dense(output_dim)
                                                          │
                                                     GRF curve
```

The scalar branch takes the last encoder time step (`x[:, -1, :]`), predicts jump height via two dense layers, then projects the scalar back to `d_model` and adds it to the encoder output before the output projection. Both curve and scalar losses train the shared encoder; no stop_gradient is applied.

### Experiment 11a: Baseline (No Scalar Branch)

**Configuration:**
```bash
python src/train.py --input-transform bspline --output-transform bspline \
    --loss reconstruction --simple-normalization \
    --run-name bspline-jh_branch --epochs 100
```

**Results:**
- Signal R² (BW): 0.933
- JH R²: -3.60, Median AE: 0.222 m, Bias: +0.173 m
- PP R²: 0.08

### Experiment 11b: Scalar-Conditioned JH Branch

**Configuration:**
```bash
python src/train.py --input-transform bspline --output-transform bspline \
    --loss reconstruction --simple-normalization \
    --scalar-prediction jump_height --scalar-loss-weight 1.0 \
    --run-name bspline-jh_scalar --epochs 100
```

**Results:**
- Signal R² (BW): 0.871 (↓ from 0.933)
- JH R²: -6.43, Median AE: 0.259 m, Bias: -0.144 m (↓ worse, bias flipped)
- PP R²: -0.03 (↓ from 0.08)
- **Scalar branch JH prediction:** RMSE 0.140 m, MAE 0.104 m, R² 0.18

### Analysis

The scalar branch degraded both tasks:

1. **Curve reconstruction worsened:** Signal R² dropped from 0.933 to 0.871. The scalar conditioning injected a poorly-learned signal into the encoder output, corrupting the GRF decoder.

2. **Scalar prediction was poor:** R² = 0.18 means the branch captures almost none of the JH variance. The RMSE of 0.14 m against a GT range of [0.003, 0.664] m is only marginally better than predicting the mean.

3. **Wrong pooling for coefficient space:** The architecture takes `x[:, -1, :]` (last time step), which was designed for raw temporal sequences where the last position corresponds to takeoff. In B-spline coefficient space, the last position is the 30th B-spline coefficient — it has no privileged temporal meaning. Global average pooling would be more appropriate for coefficient representations.

4. **Conflicting gradients:** Both losses train the shared encoder without stop_gradient. The scalar loss pushes the encoder toward features useful for JH prediction; the curve loss pushes toward features useful for GRF reconstruction. With the scalar performing poorly, its gradients act as noise on the shared encoder, degrading curve quality.

### Experiment 11c: Stop-Gradient on Conditioning

**Hypothesis:** Applying `tf.stop_gradient()` before the conditioning projection would prevent the curve loss from corrupting the scalar branch, isolating training so only the scalar MSE loss updates the scalar layers.

**Results:**
- Signal R² (BW): 0.858 (↓ from 0.871 without stop_grad)
- Scalar branch JH R²: 0.232 (↑ slightly from 0.18)
- JH R² from curves: -7.94 (↓ worse)
- PP R²: -0.01 (similar)

**Analysis:** Stop-gradient helped the scalar prediction marginally (R² 0.18 → 0.23) but curve reconstruction degraded further. The problem isn't gradient conflict — it's that conditioning with a poorly-predicted scalar (R² = 0.23) injects noise regardless of gradient flow.

### Experiment 11d: Global Average Pooling

**Hypothesis:** The `x[:, -1, :]` pooling takes the last B-spline coefficient, which has no privileged meaning. Global average pooling (`tf.reduce_mean(x, axis=1)`) might capture more relevant information for scalar prediction.

**Configuration:**
```bash
python src/train.py --input-transform bspline --output-transform bspline \
    --loss reconstruction --simple-normalization \
    --scalar-prediction jump_height --scalar-loss-weight 1.0 \
    --run-name bspline-jh_avgpool --epochs 100
```

**Results:**
- Signal R² (BW): 0.830 (↓↓ from 0.858)
- Scalar branch JH R²: 0.245 (essentially unchanged from 0.232)
- JH R² from curves: -26.2 (↓↓↓ catastrophic)
- PP R²: -0.63 (↓↓ worse)

**Analysis:** Global average pooling did not improve scalar prediction (R² 0.23 → 0.24). The curve reconstruction collapsed, with JH R² falling from -7.94 to -26.2. The scalar branch fundamentally cannot predict JH accurately from this encoder's representations.

### Conclusion

**Abandoned.** The scalar-conditioned architecture fails regardless of:
- Gradient flow control (stop_gradient)
- Pooling strategy (last position vs global average)

The scalar branch achieves only R² ≈ 0.24 for JH prediction — barely better than predicting the mean. This is consistent with the earlier finding that JH loss at any weight is detrimental (Experiments 4, 5, 8, 9). The encoder simply doesn't learn features that predict JH well, and conditioning the curve decoder with a poor prediction actively harms GRF reconstruction.

### Experiment 11e: Increased Scalar Loss Weight

**Hypothesis:** The scalar loss contributes smaller gradients than the curve loss (single value vs 500 time points). Increasing scalar weight might give the branch more training signal.

**Results (scalar_loss_weight = 100):**
- Signal R² (BW): 0.531 (↓↓↓ collapsed from 0.93)
- Scalar branch JH R²: 0.19 (unchanged)
- JH R² from curves: -172.9 (catastrophic)
- PP R²: -9.2 (catastrophic)

Weights 10, 100, 1000 all showed the same pattern: scalar R² plateaus at ~0.19-0.24, curve reconstruction collapses.

### Experiment 11f: Scalar-Only Mode (No Curve Prediction)

**Hypothesis:** Perhaps the curve reconstruction task competes with JH learning. Training the encoder solely for JH prediction (no curve output) might allow it to learn JH-relevant features.

**Configuration:**
```bash
python src/train.py --input-transform bspline --output-transform bspline \
    --simple-normalization --scalar-only \
    --run-name scalar-only-jh --epochs 100
```

**Results:**
- Scalar JH R²: 0.204
- RMSE: 0.138 m
- MAE: 0.115 m
- Bias: -0.012 m

**Analysis:** Same R² ≈ 0.20 as dual-branch experiments. The encoder cannot learn JH-predictive features regardless of whether it's also reconstructing curves. Task competition is not the problem.

### Final Conclusion

**Abandoned.** The scalar-conditioned architecture fails because the transformer encoder fundamentally cannot learn JH-predictive features from B-spline ACC coefficients:

1. Scalar R² ≈ 0.20 regardless of: pooling strategy, gradient flow, loss weight, or training objective
2. This is consistent with JH loss experiments (Experiments 4, 5, 8, 9) where JH optimization at any weight was detrimental
3. Even when JH is the *only* objective (scalar-only mode), the encoder achieves R² = 0.20 — barely better than predicting the mean

The fundamental problem is architectural: a transformer encoder processing B-spline coefficients does not naturally extract the subtle temporal features that determine takeoff velocity. JH depends on precise integration of the propulsion phase — information that may be distributed across coefficients in ways the attention mechanism cannot capture.

**Possible future directions:**
- Different architecture (1D CNN, LSTM) that preserves temporal locality
- Raw signal input instead of B-spline coefficients
- Direct ACC → JH regression without intermediate GRF prediction
- Physics-informed features (e.g., explicit velocity integration in the architecture)

**Recommendation:** Abandon auxiliary JH prediction. Focus on improving GRF curve reconstruction; JH must be computed post-hoc from predicted curves.

---

## MLP Baseline Model (January 2026)

### Motivation

The transformer architecture (~750K parameters) may be overparameterized for 896 training samples. A simple MLP baseline replicates the approach used successfully in MATLAB: a single hidden layer mapping B-spline coefficients directly.

### Architecture

```
Input → Flatten → Dense(hidden, relu) → Dropout(0.1) → Dense(output) → Reshape
```

| Input Transform | Hidden | Input Size | Parameters |
|-----------------|--------|------------|------------|
| raw | 128 | 500 | ~68K |
| bspline | 64 | 30 | ~4K |
| bspline | 128 | 30 | ~8K |

Compare to transformer: ~750K parameters

### Best Configuration (raw→bspline)

```bash
python src/train.py --model-type mlp --mlp-hidden 128 \
    --input-transform raw --output-transform bspline \
    --loss reconstruction --simple-normalization \
    --run-name mlp-raw-bspline --epochs 100
```

### Results Comparison

| Config | Input | Hidden | Params | Signal R² | PP R² |
|--------|-------|--------|--------|-----------|-------|
| Transformer | bspline | — | ~750K | 0.94 | 0.49 |
| MLP | raw | 64 | ~36K | 0.946 | 0.44 |
| **MLP** | **raw** | **128** | **~68K** | **0.951** | **0.52** |
| MLP | bspline | 64 | ~4K | 0.942 | 0.38 |
| MLP | bspline | 128 | ~8K | 0.951 | 0.46 |

**Key finding:** Raw input outperforms B-spline input. The ACC signal contains information useful for GRF prediction that is lost during B-spline compression.

### Detailed Results: MLP raw→bspline (h=128)

```
Body Weight Units:
  RMSE: 0.0919 BW
  MAE:  0.0586 BW
  R²:   0.9511

Jump Height:
  RMSE:       0.4365 m
  Median AE:  0.2656 m
  Bias:       -0.0544 m
  R²:        -6.1058

Peak Power:
  RMSE:       7.89 W/kg
  Median AE:  4.56 W/kg
  Bias:       -2.12 W/kg
  R²:         0.5219
```

### Detailed Results: MLP bspline→bspline (h=128)

```
Body Weight Units:
  RMSE: 0.0923 BW
  MAE:  0.0603 BW
  R²:   0.9507

Jump Height:
  RMSE:       0.3792 m
  Median AE:  0.2618 m
  Bias:       0.0399 m
  R²:        -4.3611

Peak Power:
  RMSE:       8.38 W/kg
  Median AE:  4.81 W/kg
  Bias:       -0.80 W/kg
  R²:         0.4613
```

### 🎯 BREAKTHROUGH: MLP fpc→fpc

This configuration achieves the first positive JH R² and best overall results:

```bash
python src/train.py --model-type mlp --mlp-hidden 128 \
    --input-transform fpc --output-transform fpc \
    --loss reconstruction --simple-normalization \
    --run-name fpc-both-mlp --epochs 200
```

**Best Results (h=128, 149 epochs):**
```
Body Weight Units:
  RMSE: 0.0827 BW
  MAE:  0.0476 BW
  R²:   0.9604

Jump Height:
  RMSE:       0.0813 m
  Median AE:  0.0490 m
  Bias:       -0.0146 m
  R²:         0.6854
  Invalid:    0 samples (no negative JH)

Peak Power:
  RMSE:       6.30 W/kg
  Median AE:  3.91 W/kg
  Bias:       -1.16 W/kg
  R²:         0.7038
```

| Hidden | Epochs | Params | JH R² | PP R² |
|--------|--------|--------|-------|-------|
| 64 | 100 | ~2K | 0.58 | 0.62 |
| 64 | 250 | ~2K | 0.67 | 0.68 |
| **128** | 149 | **~4K** | **0.69** | **0.70** |
| 256 | — | ~8K | 0.69 | 0.70 |

**h=128 is the sweet spot.** Larger hidden layers provide no benefit — the ceiling is the FPC representation (15 components), not model capacity.

### Loss Function Comparison (FPC-MLP h=128)

| Loss | Signal R² | JH R² | JH Median | PP R² | PP Median | Notes |
|------|-----------|-------|-----------|-------|-----------|-------|
| **reconstruction** | **0.960** | **0.69** | 4.9 cm | **0.70** | 3.9 W/kg | **Best R²** |
| signal_space_weighted | 0.960 | 0.67 | **4.8 cm** | 0.69 | **3.7 W/kg** | Best median errors |
| signal_space | 0.961 | 0.67 | 5.3 cm | 0.68 | 4.2 W/kg | Unweighted |
| eigenvalue_weighted | 0.949 | 0.61 | 6.3 cm | 0.65 | 4.7 W/kg | Over-weights FPC1 |

**Reconstruction loss is optimal for R².** Signal_space_weighted achieves slightly better median errors but marginally worse R². Both use jerk-based temporal weighting (propulsion phase ~10× more weighted than quiet standing). Eigenvalue weighting focuses too much on FPC1 (~70% of loss) at the expense of later components.

**Note:** A bug in temporal weight computation was fixed — weights were previously all 1.0 due to an aggressive min_weight floor. After fix, weights range [0.4, 4.1].

**Why FPC works where B-spline failed:**

1. **Mean function captures the template.** The FPC mean function encodes the typical CMJ shape: quiet standing at ~1 BW, countermovement dip, propulsion rise, return toward baseline. This "template" is baked in — the model doesn't need to learn it.

2. **Scores model deviations only.** FPC scores capture how each jump deviates from the template: deeper countermovement, steeper propulsion, etc. The model's job is simplified to predicting variation, not reconstructing the entire curve.

3. **Quiet standing is error-free.** Since quiet standing (~1 BW) is in the mean function, the model doesn't introduce errors there. This is critical for JH because errors during quiet standing compound through double integration. B-spline coefficients contribute to the whole curve including flat portions — more opportunities for integration-destroying errors.

4. **Variance-ordered components.** FPCs are ordered by variance explained. The first few components capture the dominant modes of variation that determine biomechanical outcomes.

5. **Data-driven basis.** FPCs are learned from the training data, capturing the actual patterns of variation in CMJ signals. B-splines are generic smooth basis functions with no task-specific structure.

6. **Dimensionality.** 15 FPCs vs 30 B-spline coefficients — fewer parameters, less overfitting, and each coefficient is more meaningful.

### Analysis

The MLP outperforms the transformer with 10× fewer parameters. Key findings:

1. **Raw input beats B-spline input.** PP R² improves from 0.46 to 0.52 when using raw ACC instead of B-spline coefficients as input. The B-spline compression loses information useful for GRF prediction — likely high-frequency content or transient features.

2. **MLP beats transformer.** A single hidden layer (128 neurons, ~68K params) outperforms a 3-layer transformer with multi-head attention (~750K params). The attention mechanism adds no value for this mapping task.

3. **The mapping is learnable but not linear.** The hidden layer provides necessary nonlinearity — h=128 outperforms h=64 — but doesn't require the complexity of attention.

4. **Derived metric errors remain structural.** Both architectures achieve excellent curve reconstruction (R² > 0.95) but poor JH prediction (R² negative). This confirms the problem is how small curve errors propagate through impulse integration, not model capacity.

### Implications

- **FPC→FPC with MLP is the recommended configuration** — matches MATLAB success
- **Representation matters more than architecture** — a 2K parameter MLP beats a 750K transformer
- **FPCs capture biomechanically relevant features** that B-spline and raw representations miss
- **The JH problem is solved** with the right representation — R² 0.58, median error 6.2 cm

---

## Hybrid Linear Projection + MLP Investigation (January 2026)

### Motivation

The MATLAB implementation (from the PhD work) used a "functional projection" approach where FPC scores were projected through a matrix computed from eigenfunction inner products:

```
P(i,j) = ∫[φ_ACC_i(t) × φ_GRF_j(t)] dt / ∫[φ_GRF_j(t)²] dt
Ŝ_GRF = rescale × S_ACC × P
```

This approach theoretically provides an interpretable, physics-informed baseline for mapping ACC FPCs to GRF FPCs. The goal was to implement this projection as a hybrid model: linear projection provides a baseline prediction, and an MLP learns nonlinear corrections.

### Implementation

Three hybrid architectures were implemented in `src/models.py`:

1. **Residual**: `output = rescale × (P @ x) + MLP(x)` — MLP adds corrections
2. **Sequential**: `output = MLP(P @ x)` — MLP refines projection output
3. **Parallel**: `output = α × (P @ x) + (1-α) × MLP(x)` — weighted combination

The projection matrix can be initialized two ways:
- **Computed**: From eigenfunction inner products (MATLAB-style)
- **Learned**: Via Ridge regression on training data

### Critical Discovery: Eigenfunction Inner Products Don't Work

Testing the computed projection matrix (eigenfunction inner products) produced catastrophic results:

| Configuration | JH R² | PP R² | Signal R² |
|---------------|-------|-------|-----------|
| Eigenfunction projection only | -5.94 | - | - |
| Hybrid (residual) with computed P | -0.14 | - | - |

**Root cause investigation:**

1. Examined MATLAB code at `/Users/markgewhite/.../FPC Mapping/`
2. Discovered that `mapFPC.m` uses `pca_projection_series`, not `pca_projection`
3. `pca_projection_series` actually uses a **neural network** to learn the mapping, not eigenfunction inner products!

The pure eigenfunction inner product approach fails because:
- ACC and GRF eigenfunctions are computed independently on different signal types
- Their "overlap" (inner product) doesn't naturally align
- Despite F=ma connecting the signals, the eigenfunction bases learned from each are fundamentally different

### Solution: Learned Projection Matrix

Replaced eigenfunction inner products with Ridge regression to learn the projection from data:

```python
def learn_fpc_projection_matrix(X_train, y_train, alpha=1.0):
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(X_train, y_train)
    return model.coef_.T  # shape: (input_features, output_features)
```

**Learned projection performance:**

| Configuration | JH R² | Notes |
|---------------|-------|-------|
| Learned projection only (Ridge) | 0.54 | Linear baseline |
| Hybrid residual | 0.70 | MLP adds to projection |
| **Hybrid sequential** | **0.76** | MLP refines projection |

### 5-Trial Architecture Comparison

Both hybrid architectures were evaluated with 5 trials (500 epochs each):

```bash
python src/train.py \
    --model-type hybrid --hybrid-architecture [residual|sequential] \
    --use-triaxial \
    --input-transform fpc --output-transform fpc \
    --loss reconstruction \
    --simple-normalization \
    --n-trials 5 --seed 42 \
    --epochs 500
```

| Metric | Hybrid Residual | Hybrid Sequential | Winner |
|--------|-----------------|-------------------|--------|
| **JH R²** | 0.7354 ± 0.0132 | **0.7625 ± 0.0219** | Sequential (+0.027) |
| **JH Median AE** | **0.0326 ± 0.0007 m** | 0.0375 ± 0.0017 m | Residual (-4.9 mm) |
| **PP R²** | 0.7021 ± 0.0202 | **0.7300 ± 0.0290** | Sequential (+0.028) |
| **PP Median AE** | **2.52 ± 0.15 W/kg** | 2.79 ± 0.10 W/kg | Residual (-0.28 W/kg) |
| **Signal R² (BW)** | **0.9712 ± 0.0009** | 0.9691 ± 0.0011 | Residual (+0.002) |

**Comparison with MLP baseline:**

| Metric | Hybrid Residual | Hybrid Sequential | MLP (Best) |
|--------|-----------------|-------------------|------------|
| JH R² | 0.735 ± 0.013 | 0.763 ± 0.022 | **0.823 ± 0.030** |
| JH Median AE | **0.033 ± 0.001 m** | 0.037 ± 0.002 m | 0.035 ± 0.002 m |
| PP R² | 0.702 ± 0.020 | 0.730 ± 0.029 | **0.798 ± 0.029** |
| PP Median AE | **2.52 ± 0.15 W/kg** | 2.79 ± 0.10 W/kg | 2.66 ± 0.07 W/kg |
| Signal R² (BW) | **0.971 ± 0.001** | 0.969 ± 0.001 | **0.971 ± 0.001** |

### Analysis: R² vs Absolute Error Trade-off

A nuanced pattern emerges from the comparison:

1. **Sequential achieves higher R²** — explains more variance in the population, better at distinguishing high jumpers from low jumpers.

2. **Residual achieves lower absolute errors** — individual predictions are closer to ground truth on average. JH median error is 3.3 cm vs 3.7 cm (11% improvement).

3. **Residual may "regress toward the mean"** — more conservative predictions that don't fully capture extremes, resulting in lower R² but better typical-case accuracy.

**Practical implications:**
- **Athlete monitoring over time**: Residual's lower AE may be preferable (detecting small changes)
- **Comparing athletes or population studies**: Sequential's higher R² preserves rankings better
- **Maximum performance**: MLP still wins on R² metrics

### Why Neither Hybrid Beats MLP

1. **Linear projection bottleneck**: Even with MLP refinement, starting from a linear projection (R² = 0.54) limits achievable performance.

2. **MLP learns projection implicitly**: A direct MLP with sufficient hidden units learns any beneficial linear structure plus nonlinear corrections without constraints.

3. **Interpretability vs performance**: The hybrid approach offers interpretability (the P matrix shows which ACC FPCs contribute to which GRF FPCs) but at the cost of R² reduction.

### Key Insights

1. **MATLAB's "projection" approach actually used neural networks** — the eigenfunction inner product formula alone doesn't work for ACC→GRF mapping.

2. **Learned projection provides a reasonable baseline** (R² = 0.54) that can be refined by an MLP.

3. **Simple MLP remains the best approach** — the added complexity of hybrid architecture doesn't improve R² performance.

4. **Residual vs Sequential trade-off:**
   - **Sequential** achieves higher R² (better variance explanation)
   - **Residual** achieves lower absolute errors (better individual prediction accuracy)
   - Choice depends on application: monitoring (residual) vs population comparison (sequential)

### Files Added

| File | Purpose |
|------|---------|
| `src/transformations.py` | Added `learn_fpc_projection_matrix()`, `compute_fpc_projection_matrix()` |
| `src/models.py` | Added `HybridProjectionMLP` class |
| `src/train.py` | Added `--model-type hybrid`, `--hybrid-architecture`, `--projection-init` options |
| `scripts/analyze_projection.py` | Projection matrix visualization and analysis |
| `scripts/debug_projection.py` | Eigenfunction debugging script |
| `scripts/test_learned_projection.py` | Quick test of learned projection performance |

### Conclusion

**The hybrid linear projection + MLP approach does not improve upon the simple MLP for R² metrics.** The investigation revealed:

- MATLAB's success came from neural network refinement, not pure eigenfunction inner products
- A learned projection provides an interpretable but suboptimal baseline
- The simple MLP (JH R² = 0.82) remains the best configuration for variance explanation

**However, hybrid residual offers a compelling trade-off:**
- **Lowest absolute errors** of all architectures (JH median AE = 3.3 cm)
- Better signal reconstruction (Signal R² = 0.971)
- More conservative predictions that may be preferable for athlete monitoring applications

The choice between architectures depends on the use case:
- **Maximum R²**: Use simple MLP
- **Minimum absolute error**: Use hybrid residual
- **Interpretability**: Hybrid approaches expose the projection matrix structure

---

## Rigorous B-Spline Reference Evaluation

### Motivation

Previous evaluations compared FPC-based predictions against FPC reconstructions. This creates an unfair comparison:

- **FPC reconstruction** (99% variance threshold) discards 1% of signal variance
- **B-spline reconstruction** captures closer to 100% of smoothed signal variance

When evaluating FPC output against FPC-reconstructed ground truth, the "hardest" 1% of variance is removed from the target, artificially inflating R² values.

### Implementation

Added `--use-bspline-reference` flag that:
1. Computes B-spline reconstruction of normalized GRF **before** applying output transforms
2. Uses this consistent 500-point smoothed reference as ground truth for all evaluations
3. Ensures fair apples-to-apples comparison across B-spline and FPC output transforms

### Results with Rigorous Evaluation

**Hybrid Sequential (5 trials):**
```bash
python src/train.py --model-type hybrid --hybrid-architecture sequential \
    --use-bspline-reference --n-trials 5
```

| Trial | JH R² | PP R² | Signal R² |
|-------|-------|-------|-----------|
| 0 | 0.539 | 0.720 | 0.971 |
| 1 | 0.496 | 0.680 | 0.969 |
| 2 | 0.596 | 0.773 | 0.972 |
| 3 | 0.518 | 0.678 | 0.970 |
| 4 | 0.516 | 0.670 | 0.970 |
| **Mean ± Std** | **0.533 ± 0.034** | **0.704 ± 0.039** | **0.970 ± 0.001** |

**Hybrid Residual (5 trials):**
```bash
python src/train.py --model-type hybrid --hybrid-architecture residual \
    --use-bspline-reference --n-trials 5
```

| Trial | JH R² | PP R² | Signal R² |
|-------|-------|-------|-----------|
| 0 | 0.555 | 0.691 | 0.970 |
| 1 | 0.494 | 0.666 | 0.970 |
| 2 | 0.449 | 0.656 | 0.970 |
| 3 | 0.563 | 0.740 | 0.971 |
| 4 | 0.423 | 0.569 | 0.967 |
| **Mean ± Std** | **0.497 ± 0.056** | **0.664 ± 0.056** | **0.970 ± 0.001** |

### Comparison: Standard vs Rigorous Evaluation

| Metric | Standard Eval | Rigorous Eval | Difference |
|--------|---------------|---------------|------------|
| **Hybrid Sequential** | | | |
| JH R² | 0.763 ± 0.022 | 0.533 ± 0.034 | -0.230 |
| PP R² | 0.730 ± 0.029 | 0.704 ± 0.039 | -0.026 |
| Signal R² | 0.969 ± 0.001 | 0.970 ± 0.001 | +0.001 |
| **Hybrid Residual** | | | |
| JH R² | 0.735 ± 0.013 | 0.497 ± 0.056 | -0.238 |
| PP R² | 0.702 ± 0.020 | 0.664 ± 0.056 | -0.038 |
| Signal R² | 0.971 ± 0.001 | 0.970 ± 0.001 | -0.001 |

### Analysis

1. **JH R² drops significantly** (~0.23) — the 1% "missing variance" in FPC reconstruction was disproportionately in the high-frequency features that affect jump height calculation.

2. **PP R² drops moderately** (~0.03) — peak power is less sensitive to the missing variance.

3. **Signal R² unchanged** — the overall curve shape is well-captured regardless of evaluation mode.

4. **MLP outperforms hybrid architectures on R² metrics** — the simple MLP achieves best variance explanation:

| Metric | Hybrid Residual | Hybrid Sequential | MLP | Winner |
|--------|-----------------|-------------------|-----|--------|
| JH R² | 0.497 ± 0.056 | 0.533 ± 0.034 | **0.639 ± 0.033** | MLP |
| JH Median AE | 0.052 ± 0.004 m | **0.048 ± 0.002 m** | 0.051 ± 0.001 m | Sequential |
| PP R² | 0.664 ± 0.056 | 0.704 ± 0.039 | **0.803 ± 0.030** | MLP |
| PP Median AE | 2.79 ± 0.15 W/kg | **2.63 ± 0.14 W/kg** | 2.65 ± 0.32 W/kg | Sequential |
| Signal R² | 0.970 ± 0.001 | 0.970 ± 0.001 | **0.971 ± 0.001** | MLP |

**Key observations:**
- **MLP dominates R² metrics** — JH R² improved by 0.11 over sequential, PP R² by 0.10
- **Sequential achieves lowest median absolute errors** — but differences are small
- **MLP has higher PP Median AE variance** (0.32 vs 0.14 W/kg) — less consistent across trials
- **Signal R² essentially identical** — all architectures capture the curve shape equally well

### The Interpretability Trade-off

Despite MLP's superior R² metrics, **hybrid residual offers an interpretability advantage** that may justify the ~0.14 JH R² sacrifice:

**Residual architecture:** `output = P @ x + MLP(x)`
- The projection matrix P **directly contributes** to the output
- P coefficients are interpretable: "ACC FPC₃ contributes weight 0.42 to GRF FPC₁"
- The MLP learns additive corrections — P remains the "explainable backbone"
- You can analyze P to understand the biomechanical ACC→GRF mapping

**Sequential architecture:** `output = MLP(P @ x)`
- P's contribution is **transformed through nonlinear layers**
- Cannot easily interpret what P "means" after MLP transformation
- The interpretability is lost

**Scientific value of interpretability:**

The projection matrix P could reveal insights such as:
- Which ACC eigenfunctions (movement patterns) predict which GRF eigenfunctions (force characteristics)
- Whether propulsion-phase acceleration maps to braking-phase force production
- The relative importance of different movement features for jump performance

For biomechanics research or scientific publication, being able to explain *why* the model makes predictions may be worth more than a few percentage points of R².

### Conclusion

Rigorous B-spline reference evaluation provides a fairer comparison across output transforms and reveals:

1. Biomechanical metrics (JH, PP) are more sensitive to the "missing" 1% variance in FPC reconstruction than previously apparent
2. **Simple MLP achieves best R² metrics** — JH R² = 0.64, PP R² = 0.80 (vs 0.53/0.70 for sequential hybrid)
3. **Hybrid sequential achieves lowest median absolute errors** — though differences are small
4. **Hybrid residual's interpretable projection matrix offers scientific value** that may outweigh its R² disadvantage

**Recommendation:**
- For **maximum R² (variance explanation)**: Use simple MLP
- For **lowest absolute errors**: Use hybrid sequential
- For **interpretable biomechanical insights**: Use hybrid residual (accepting ~0.14 JH R² trade-off)
- For **consistent cross-method comparisons**: Always use `--use-bspline-reference`

---

## Appendix: Biomechanics Calculations

### Jump Height (Impulse-Momentum Method)
```
net_force = GRF - 1.0  (in BW units)
acceleration = net_force × g
velocity = ∫ acceleration dt
position = ∫ velocity dt
jump_height = position_final + velocity_final² / (2g)
```

### Peak Power
```
velocity = g × ∫(GRF - 1.0) dt
power = GRF × velocity  (instantaneous)
peak_power = g × max(power)  (in W/kg)
```

Both calculations use 250 Hz sampling rate after GRF downsampling.
