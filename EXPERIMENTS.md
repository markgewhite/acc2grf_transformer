# Experiment Log: ACC → GRF Transformer

This document records the experiments conducted to develop a transformer model that predicts vertical ground reaction force (vGRF) from lower-back accelerometer data during countermovement jumps.

## Objective

Train a sequence-to-sequence transformer to map triaxial accelerometer signals to vGRF, with the ultimate goal of accurately predicting biomechanical metrics (jump height, peak power) from the predicted GRF.

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

Based on experiments, the best configuration is:

```bash
python -m src.train \
    --use-triaxial \
    --d-model 128 \
    --d-ff 512 \
    --loss combined \
    --mse-weight 1.0 \
    --jh-weight 0.0 \
    --pp-weight 0.1 \
    --epochs 100
```

### Experimental: Weighted MSE

To try temporally weighted MSE (emphasizes biomechanically important regions):

```bash
python -m src.train \
    --use-triaxial \
    --d-model 128 \
    --d-ff 512 \
    --loss weighted \
    --epochs 100
```

The `weighted` loss uses temporal weights derived from the global average of |d²ACC/dt²| across training samples, emphasizing countermovement and propulsion phases over quiet standing.

---

## Next Steps

1. ~~**Combine JH and PP losses**~~: Tested at multiple weights (0.1, 0.01)—JH loss is detrimental at any weight. Abandoned.

2. **Temporally weighted MSE**: Try `--loss weighted` to emphasize high-jerk regions (countermovement, propulsion) over quiet standing. Weights derived from second derivative of ACC, averaged globally over training set.

3. **Investigate outliers**: The outlier diagnostic plots may reveal common patterns in problematic samples.

4. **Sequence length**: Current 500 samples may truncate important context. Try 800 samples.

5. **Architecture variants**: Consider temporal convolutional networks (TCN) for comparison.

6. **Cross-validation**: Implement k-fold CV at subject level for more robust evaluation.

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
