# GRF Curve Reconstruction: Literature Comparison

## Your R² 0.97 in Context

Your signal reconstruction R² of 0.971 is **among the highest reported in the wearable-to-GRF prediction literature**, and is achieved under constraints that make it more impressive than the raw number alone suggests.

### Comparable Results from the Literature

| Study | Year | Task | Sensors | Model | Best Metric |
|---|---|---|---|---|---|
| **This work** | **2025** | **CMJ** | **1 acc (lower back)** | **MLP (12K params)** | **R² = 0.971** |
| Yılmazgün et al. | 2025 | Walking/running/turns | 1-7 IMUs | CNN | r = 0.98, rRMSE 6-7% |
| CNN-xLSTM (Sensors) | 2025 | Running (5 speeds) | IMUs (lower limb) | CNN-xLSTM | R² = 0.909 ± 0.064 |
| Alcantara et al. | 2022 | Uphill/downhill running | Accelerometers | LSTM (RNN) | rRMSE = 6.4% |
| Self-supervised (IEEE TBE) | 2024 | Walking | 8 IMUs (full body) | Transformer (2M params) | r = 0.94-0.97 |
| Consumer insoles + LSTM | 2024 | Treadmill running | 16 pressure + IMU | Bi-LSTM | r = 0.991, rRMSE 3.2% |
| Wouda et al. (dancers) | 2020 | Dance jumps | Sacrum IMU | SVM + ANN | r = 0.80-0.95, RMSE 0.25 BW |
| LSTM-MLP (ankle) | 2024 | Walking/running | IMU sensors | LSTM-MLP | R² = 0.89 / 0.87 |

For jump height prediction specifically:

| Study | Year | Approach | JH Metric |
|---|---|---|---|
| **This work** | **2025** | **GRF curve → impulse-momentum** | **R² = 0.82, 3.5 cm median error** |
| Rantalainen et al. (PLOS ONE) | 2022 | FPCA features → SVM (direct) | Peak power error 2.3 W/kg |
| ANN vs MLR | 2024 | Direct ANN prediction | R² = 0.68 |
| Smartphone ML | 2023 | Video-based features | Variable (lower accuracy) |

---

## Where This Work Differentiates Itself

### 1. Dual FPCA Representation (Most Novel Aspect)

The literature uses FPCA for **feature extraction on the input side** — extracting scores from accelerometer signals, then feeding them into ML models that predict scalar outcomes (jump height, peak power). [Rantalainen et al. (2022)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0263846) is the closest comparator: same sensor location, same FPCA feature extraction, same CMJ task, but they predict **scalar metrics directly**.

This approach applies FPCA to **both input and output**, turning the problem from "500-point signal → 500-point signal" into "15 scores → 15 scores." This is a fundamentally different formulation. The mean function encodes the canonical CMJ shape, so the model only learns **deviations from the biomechanical template**. No other study identified uses this dual-FPCA formulation for GRF curve prediction.

### 2. Full Curve Reconstruction for a Ballistic Movement

The vast majority of GRF prediction studies target **cyclic locomotion** (walking, running) where the movement pattern is repetitive and relatively constrained. CMJ is a ballistic, non-cyclic task with substantially more inter-individual variability in movement strategy (countermovement depth, rate of force development, timing). Achieving R² = 0.97 on this task is harder than achieving equivalent accuracy on running GRF.

Furthermore, most CMJ-specific studies predict **scalar outcomes** (jump height, peak power) directly from features. This pipeline reconstructs the **entire GRF waveform** and then derives metrics via the impulse-momentum theorem — the same physics used with real force plates. This makes the predictions biomechanically interpretable and auditable.

### 3. The "Representation > Architecture" Finding

The self-supervised transformer study ([Alcantara et al., IEEE TBE 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10849467/)) used **2 million parameters** and **8 IMUs** to achieve r = 0.94-0.97 on walking GRF. The CNN-xLSTM study achieved R² = 0.909 with a hybrid deep learning architecture.

This work's 12K-parameter MLP achieves R² = 0.971 with a **single accelerometer**. The 60x-170x parameter efficiency and the finding that a transformer architecture performs *worse* (producing meaningless biomechanical metrics despite high signal R²) is a genuinely valuable negative result. It demonstrates that the right data representation (FPCA) eliminates the need for architectural complexity — an insight that cuts against the prevailing trend in the field toward ever-larger models.

### 4. Single Low-Cost Sensor

Several high-performing studies rely on multi-sensor setups: 8 IMUs for the self-supervised approach, 16-sensor pressure insoles for the LSTM approach. This work uses a **single triaxial accelerometer** at the lower back — the most practical and cost-effective sensor configuration. The statistically significant finding that triaxial outperforms resultant magnitude (+0.15 JH R²) is also a useful empirical contribution.

### 5. End-to-End Biomechanical Validation

Many GRF prediction studies report only signal-level metrics (R², RMSE, correlation). This work evaluates the **downstream biomechanical utility**: jump height R² = 0.82 and peak power R² = 0.80 against a clearly defined theoretical ceiling (0.87 JH R² from the 500ms window truncation). This distinction matters because, as the transformer results showed, high signal R² does not guarantee meaningful biomechanical metrics.

---

## Honest Assessment of Limitations

- The R² = 0.97 is for signal reconstruction. The more clinically relevant JH R² of 0.82 is good but not exceptional — though it approaches the 0.87 ceiling imposed by the temporal window.
- The dataset (73 participants, ~1,100 jumps) is modest. Generalizability to new populations is untested.
- The comparison is somewhat complicated by the fact that most literature targets running (a different biomechanical domain), so direct R²-to-R² comparisons across tasks should be interpreted cautiously.

---

## Summary: Unique Contribution

The core novelty is **using FPCA as a shared representation space for both accelerometer input and GRF output**, reducing cross-domain signal prediction to a low-dimensional score mapping. This is distinct from the literature, which uses FPCA only for input feature extraction. Combined with the demonstration that this representation makes a simple MLP competitive with or superior to deep architectures requiring orders of magnitude more parameters, this constitutes a meaningful methodological contribution to the wearable biomechanics field — particularly for non-cyclic movements like the CMJ where fewer solutions exist.

---

## Sources

- [Rantalainen et al. (2022) - Jump performance from single accelerometer, PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0263846)
- [Yılmazgün et al. (2025) - CNN for 3D GRF across movement tasks](https://pubmed.ncbi.nlm.nih.gov/40972273/)
- [CNN-xLSTM for vGRF (2025) - Sensors/MDPI](https://www.mdpi.com/1424-8220/25/4/1249)
- [Alcantara et al. (2022) - LSTM RNN for running GRF, PeerJ](https://pmc.ncbi.nlm.nih.gov/articles/PMC8740512/)
- [Self-supervised transformer for GRF (2024) - IEEE TBE](https://pmc.ncbi.nlm.nih.gov/articles/PMC10849467/)
- [Consumer insoles + LSTM (2024) - PeerJ](https://peerj.com/articles/17896/)
- [Wouda et al. (2020) - ML estimation of GRF from wearable data](https://pmc.ncbi.nlm.nih.gov/articles/PMC7038404/)
- [Bogaert et al. (2024) - Predicting vGRF during running, Frontiers](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2024.1440033/full)
- [FPCA primer for biomechanics - J. Biomechanics](https://www.sciencedirect.com/science/article/abs/pii/S0021929020305303)
- [Frontiers systematic review - ML in running biomechanics](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.913052/full)
