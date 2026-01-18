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

---

## Next Steps

1. **Combine both biomechanics losses**: Try `--jh-weight 0.05 --pp-weight 0.1` to get benefits of both.

2. **Investigate outliers**: The outlier diagnostic plots may reveal common patterns in problematic samples.

3. **Sequence length**: Current 500 samples may truncate important context. Try 800 samples.

4. **Architecture variants**: Consider temporal convolutional networks (TCN) for comparison.

5. **Cross-validation**: Implement k-fold CV at subject level for more robust evaluation.

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
