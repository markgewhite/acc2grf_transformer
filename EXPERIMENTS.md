# Experiment Log: ACC → GRF Transformer

This document records the experiments conducted to develop a transformer model that predicts vertical ground reaction force (vGRF) from lower-back accelerometer data during countermovement jumps.

## Objective

Train a sequence-to-sequence transformer to map accelerometer signals to vGRF, with the ultimate goal of accurately predicting biomechanical metrics (jump height, peak power) from the predicted GRF.

---

## Current Status (January 2026)

**Best reproducible configuration:**
```bash
python src/train.py \
    --input-transform fpc --output-transform fpc \
    --fixed-components --n-components 15 \
    --no-varimax \
    --loss eigenvalue_weighted \
    --epochs 100
```

**Results:**
- Signal R² (BW): 0.91
- JH R²: 0.34
- PP R²: 0.33

**Key lessons learned:**

1. **Triaxial vs Resultant depends on transform type:**
   - Raw signals: Triaxial > Resultant (preserves directional information)
   - FPC transforms: Resultant > Triaxial (triaxial causes dimensionality explosion)

2. **scikit-fda adoption caused permanent performance regression:**
   - Original custom FPCA: JH R² ≈ 0.61
   - After scikit-fda refactoring: JH R² ≈ 0.34 (best case)
   - The L² inner products in scikit-fda behave differently from discrete dot products

3. **Normalization pipeline required extensive fixes:**
   - Sample 920 had corrupted ACC data (-1130g) that destroyed global normalization
   - Required robust median/MAD statistics instead of mean/std
   - Multiple iterations to get stable training

4. **Loss function design matters but has limits:**
   - Eigenvalue-weighted MSE improved JH R² from 0.22 to 0.34
   - Signal-space loss did not help (gradient dilution)
   - Jump height in loss always hurts (unstable double-integration gradients)

---

## Data Overview

- **Source**: `processedjumpdata.mat` (858 MB)
- **Subjects**: 73
- **Valid jumps**: 1136 total, ~240 in validation set
- **ACC sampling rate**: 250 Hz (lower back sensor, triaxial)
- **GRF sampling rate**: 1000 Hz → downsampled to 250 Hz
- **Sequence length**: Padded/truncated to 500 samples
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

3. ~~**Triaxial with FPC**~~: Tested—causes severe degradation. Triaxial creates 45 input features (15 FPCs × 3 channels) vs 15 for resultant. The model cannot effectively learn which components matter. Abandoned.

4. **Investigate outliers**: The outlier diagnostic plots may reveal common patterns in problematic samples.

5. **Sequence length**: Current 500 samples may truncate important context. Try 800 samples.

6. **Architecture variants**: Consider temporal convolutional networks (TCN) for comparison.

7. **Cross-validation**: Implement k-fold CV at subject level for more robust evaluation.

8. **Custom discrete FPCA**: Re-implement FPCA with discrete dot products to recover original performance (JH R² ≈ 0.61 vs current 0.34).

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
