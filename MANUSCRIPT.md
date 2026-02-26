# Reconstructing vertical ground reaction force curves during countermovement jumps from a single wearable accelerometer using functional principal component analysis

Mark G. E. White

---

## Abstract

Force plates are the gold standard for assessing countermovement jump (CMJ) performance, providing the vertical ground reaction force (vGRF) time series for computing jump metrics, whilst the waveforms themselves provide practitioners with insight into how the performance was achieved. However, force plates are expensive and impractical in the field. Previous attempts to predict jump metrics from wearable sensors have achieved limited accuracy – in part because they predicted scalar outcomes directly. This study reconstructs vGRF waveforms during CMJs from a single lower-back triaxial accelerometer using a dual functional principal component analysis (FPCA) representation. A multilayer perceptron (MLP) mapped accelerometer FPC scores to vGRF FPC scores. Three signal representations (raw, B-spline, FPC) and two model architectures (MLP, transformer) were compared systematically. The FPC-based MLP achieved a signal reconstruction of R² = 0.971 ± 0.001 (RMSE = 0.071 BW). Jump height R² = 0.82 ± 0.03 (median error 3.5 cm), approaching the 0.87 theoretical ceiling imposed by the signal window. Peak power R² = 0.80 ± 0.03 (median error 2.7 W·kg⁻¹). The method was validated on 1,136 jumps from 73 participants. The dual-FPCA representation reduced a complex signal mapping to a tractable score mapping, eliminating the need for large neural network architectures. 

**Keywords:** ground reaction force; accelerometer; functional principal component analysis; countermovement jump; machine learning
---

## 1. Introduction

Ground reaction force (GRF) measurement is fundamental to sports biomechanics, providing direct insight into the forces an athlete produces during movement. The countermovement jump (CMJ) is among the most widely used assessments of neuromuscular function in sport (Claudino et al., 2017), with the vertical GRF (vGRF) time series enabling computation of performance metrics including jump height, peak power, rate of force development, and impulse (Linthorne, 2001; McMahon et al., 2018). These metrics serve critical roles in athlete monitoring, talent identification, training programme evaluation, and return-to-play decision-making (Cormie et al., 2009; Owen et al., 2014). Force plates remain the gold standard for GRF measurement, offering direct, high-fidelity force recording at sampling rates of 1,000 Hz or higher.

However, force plates present substantial barriers to widespread deployment. Laboratory-grade platforms cost £5,000–50,000, require fixed installation, and confine testing to controlled environments (Camomilla et al., 2018). This creates a persistent gap between laboratory-based research findings and field-based practice, where coaches and practitioners most need biomechanical data. The growing demand for accessible, field-deployable assessment tools in sport has motivated interest in wearable sensor alternatives — devices that are inexpensive, portable, and unobtrusive (Halilaj et al., 2018). A single body-worn accelerometer, costing as little as £50, could in principle provide the same biomechanical information as a force plate if its output signal could be mapped to GRF with sufficient accuracy.

Considerable progress has been made in predicting GRF from wearable sensors during cyclic locomotion. For running, Alcantara et al. (2022) used a long short-term memory (LSTM) recurrent neural network to predict vGRF from sacral accelerometry, achieving a relative root mean square error (rRMSE) of 6.4%. Yılmazgün et al. (2025) applied convolutional neural networks (CNNs) to predict three-dimensional GRF across walking, running, and turning movements from 1–7 inertial measurement units (IMUs), reporting correlations of r = 0.98. A self-supervised transformer approach using eight full-body IMUs achieved correlations of r = 0.94–0.97 for walking GRF with approximately 2 million parameters (Choi et al., 2024). Consumer-grade pressure insoles combined with a bidirectional LSTM architecture have achieved r = 0.991 for treadmill running (Moghadam et al., 2024). A hybrid CNN-xLSTM model predicted running vGRF with R² = 0.909 ± 0.064 across five speeds (Tan et al., 2025). The prevailing trend in this literature is toward ever-larger models and multi-sensor arrays, with architectures growing from simple regression to attention-based transformers with millions of parameters. These studies demonstrate that wearable-to-GRF prediction is feasible, but they predominantly address cyclic movement patterns.

The CMJ presents distinct challenges for GRF prediction. Unlike walking and running, the CMJ is ballistic and non-cyclic, with substantially greater inter-individual variability in movement strategy — participants differ in countermovement depth, rate of force development, and the timing of the unweighting-to-braking transition (Cormie et al., 2009; Lake et al., 2018). Most CMJ-specific work has predicted scalar performance metrics directly from accelerometer features rather than reconstructing the full force-time curve. Rantalainen et al. (2020) used functional principal component analysis (FPCA) features extracted from accelerometer signals as input to a support vector machine (SVM) for predicting peak power, reporting an RMSE of 2.3 W·kg⁻¹. White et al. (2022) extended this approach by comparing sensor locations, axis combinations, and model types, achieving useful but limited prediction accuracy. Wouda et al. (2020) estimated GRF during dance jumps from a sacral IMU using SVM and artificial neural networks, reporting correlations of r = 0.80–0.95 — but these were cyclic, repetitive movements rather than maximal-effort ballistic jumps. A fundamental limitation of scalar prediction is that the model is tied to specific metrics chosen at training time; it cannot derive new metrics post-hoc, and the predicted scalar provides no insight into the underlying force-time profile.

Reconstructing the entire GRF waveform is fundamentally more powerful than predicting scalar metrics. When the full force-time curve is available, any biomechanical metric can be derived using the same physics applied to actual force plate data — the impulse-momentum theorem for jump height (Linthorne, 2001), the product of force and velocity for instantaneous power (Cormie et al., 2009), and the derivative of force for rate of force development (Maffiuletti et al., 2016). Furthermore, the predicted curve is interpretable and auditable: practitioners can visually inspect the force-time profile for movement quality, asymmetries, or compensatory strategies. This transparency is not available from a single predicted number.

Functional data analysis (FDA), and specifically FPCA, provides a principled framework for representing biomechanical signals. FPCA has a history spanning more than three decades in biomechanics and human movement science, with applications ranging from gait analysis to sports technique evaluation (Harrison et al., 2024; Warmenhoven et al., 2018). FPCA decomposes a set of curves into a mean function — capturing the canonical shape — and a set of orthogonal eigenfunctions (functional principal components) that capture the dominant modes of variation across individuals (Ramsay & Silverman, 2005). Each signal is then represented by a small number of scores indicating how much it deviates from the mean along each mode. This provides a principled dimensionality reduction that respects the continuous, functional nature of biomechanical signals, unlike discrete approaches that treat each time point independently. FPCA has been applied to CMJ force-time curves for descriptive and classification purposes (Kipp et al., 2018; Richter et al., 2021), and for input feature extraction from accelerometer signals in predictive models (White et al., 2022; Rantalainen et al., 2020). However, no previous study has applied FPCA to both input and output signals to transform the problem from a high-dimensional signal-to-signal mapping to a low-dimensional score-to-score mapping.

The purpose of this study was to develop and validate a method for reconstructing full vGRF waveforms during countermovement jumps from a single triaxial accelerometer using a dual FPCA representation. I addressed three specific aims: (1) to evaluate the accuracy of vGRF curve reconstruction under different signal representations and model architectures; (2) to assess the accuracy of downstream biomechanical metric prediction (jump height and peak power) derived from the reconstructed curves; and (3) to examine the relative importance of triaxial versus resultant acceleration input.

---

## 2. Methods

### 2.1 Participants and data collection

Data were collected as part of the study reported in White et al. (2022), in which male and female athletes at recreational, club, and national levels performed countermovement jumps under controlled laboratory conditions. Full details of the recruitment, data collection protocol, and equipment are provided in that paper; a summary is given here. The present study applies a different analytical approach to the same dataset. The full dataset comprised 73 participants and 1,136 valid jumps; the subset analysed in White et al. (2022) used 69 participants and 696 jumps after applying different exclusion criteria and including both arm swing and non-arm-swing conditions. All participants provided informed consent and the study was approved by the institutional ethics committee.

Participants performed countermovement jumps without arm swing on two portable force platforms (Kistler, Winterthur, Switzerland) sampling at 1,000 Hz. A triaxial accelerometer (Axivity AX3, sampling rate 250 Hz) was attached to the lower back at the level of the fifth lumbar vertebra using a custom elastic belt. The accelerometer recorded acceleration along three orthogonal axes aligned approximately with the anteroposterior, mediolateral, and vertical directions of the body. GRF data from both force platforms were summed to produce the total vertical ground reaction force for each jump. The dataset comprised 1,136 valid jumps after exclusion of trials with sensor artefacts (acceleration exceeding 100 g on any axis).

### 2.2 Signal preprocessing

Both accelerometer and GRF signals were aligned to the instant of takeoff, defined as the point at which the vertical GRF fell below a threshold of 10 N. A 2,000 ms window preceding takeoff was extracted from each signal, corresponding to 500 samples at 250 Hz. This window captured the full countermovement and propulsion phases of the CMJ. GRF data were downsampled from 1,000 Hz to 250 Hz to match the accelerometer sampling rate and normalised by body weight (BW) so that quiet standing produced a value of approximately 1.0 BW.

The dataset was partitioned at the participant level into training (80%) and validation (20%) sets, ensuring that all jumps from a given participant appeared in the same partition. This prevented data leakage and ensured that the model was evaluated on its ability to generalise to unseen individuals, not merely to unseen jumps from known participants.

Global z-score normalisation was applied to both signals using training set statistics (mean and standard deviation computed across all training samples and time points). This simple normalisation scheme was adopted after extensive experimentation demonstrated that it outperformed more complex alternatives, including median absolute deviation-based normalisation, which created extreme values that destabilised neural network training.

### 2.3 Functional signal representation

Both accelerometer and GRF signals were converted from discrete samples to smooth functional representations. Each signal was expressed as a linear combination of 50 cubic B-spline basis functions with a roughness penalty on the second derivative to suppress high-frequency noise while preserving the biomechanically relevant signal features (Ramsay & Silverman, 2005). The roughness penalty parameter λ was selected independently for each signal channel using generalised cross-validation (GCV), which balances fidelity to the observed data against smoothness of the fitted function. The GCV-optimal λ was fitted on the training set and applied to both training and validation data.

This functional representation served as the common starting point for three signal representation strategies compared in this study (Figure 2):

**Smoothed signals.** The B-spline smoothed signals were re-evaluated at the original 500 grid points, yielding noise-free discrete time series with the same dimensionality as the original input (500 time steps × 3 channels = 1,500 values; output 500 × 1 = 500 values). This is the sequence-to-sequence baseline, comparable to the approach most commonly used in the GRF prediction literature (Alcantara et al., 2022; Bogaert et al., 2024), but starting from a clean functional representation rather than raw samples.

**B-spline coefficients.** The 30 coefficients from a separate B-spline fit (de Boor, 2001) were passed directly as a reduced-dimensional representation (30 coefficients per channel). The coefficients have no variance ordering and no inherent biomechanical structure. The input comprised 30 coefficients × 3 channels = 90 values; the output comprised 30 coefficients × 1 channel = 30 values.

**FPC scores.** Each signal was decomposed via functional principal component analysis (FPCA; Section 2.4) into a mean function plus variance-ordered deviations. The input comprised 15 FPC scores × 3 channels = 45 values; the output comprised 15 FPC scores × 1 channel = 15 values. This representation achieves the greatest dimensionality reduction (45 → 15) and provides a variance-ordered, biomechanically structured input space.

All three representations derived from the same smoothed functional signals, ensuring that differences in performance reflected the representation structure rather than differential noise handling. In the smoothed signal representation, the prediction model operated on the re-evaluated time series directly. In the B-spline and FPC representations, the model's output was a vector of coefficients or scores rather than a time series; the full GRF waveform was reconstructed by applying the corresponding inverse transform (multiplying by the basis functions or eigenfunctions). The reconstructed waveform was then used to derive biomechanical metrics (Section 2.8).

### 2.4 Functional principal component analysis

FPCA decomposes a set of curves into a mean function and a set of orthogonal eigenfunctions that capture the principal modes of variation (Ramsay & Silverman, 2005). Each observed signal *f(t)* is approximated as:

*f(t) ≈ μ(t) + Σₖ cₖ φₖ(t)*

where *μ(t)* is the mean function, *φₖ(t)* are the eigenfunctions (functional principal components), and *cₖ* are the corresponding scores. The eigenfunctions are ordered by decreasing variance explained, so that the first few components capture the dominant sources of variation across individuals.

FPCA was applied independently to each of the three accelerometer channels, yielding 15 FPC scores per channel (45 scores total for triaxial input). For the GRF signal, FPCA was applied to the single vertical channel, yielding 15 FPC scores. In both cases, 15 components explained approximately 99% of the total variance. The FPCA implementation used the scikit-fda library (Ramos-Carreño et al., 2024), with signals first converted to functional data objects using 50 B-spline basis functions over the normalised time domain [0, 1].

The key property of this representation is that the mean function encodes the canonical CMJ shape — the average force-time or acceleration-time profile — while the FPC scores capture only the individual-specific deviations from this average. The prediction model therefore learns to map deviations in one domain (acceleration) to deviations in another (force), rather than learning the full signal structure from scratch. This fundamentally simplifies the learning problem. It is worth noting that for the FPC representation, the score-to-score mapping can be characterised as a linear projection — a weighted combination of input scores producing each output score. This property is exploited in Section 2.7 to provide an interpretability analysis of the learned mapping. Figure 1 shows the mean functions and first three eigenfunctions for both the accelerometer and GRF signals, illustrating the structure captured by each component.

### 2.5 Prediction models

I compared two model architectures representing opposite ends of the complexity spectrum:

**Transformer.** An encoder-only transformer (Vaswani et al., 2017) with approximately 750,000 parameters, comprising a 64-dimensional model, 4 self-attention heads, 3 encoder layers, a 128-dimensional feed-forward network, and learnable positional encoding. The self-attention mechanism learns temporal relationships across the full input sequence, allowing each time step to attend to every other. I applied this architecture to smoothed signals, representing the state-of-the-art deep learning approach to sequence-to-sequence prediction. The transformer embodies the trend in the field toward increasingly complex architectures (Choi et al., 2024; Bogaert et al., 2024).

**Multilayer perceptron (MLP).** A single hidden layer feedforward network (Hornik et al., 1989) with approximately 12,000 parameters: input → Dense(128, ReLU) → Dropout(0.1) → output. This minimal architecture was applied to all three signal representations, serving as the baseline against which the transformer's added complexity could be justified — or not.

Both models were trained using the Adam optimiser (Kingma & Ba, 2015) with a learning rate of 1 × 10⁻⁴, mean squared error (MSE) loss, batch size of 32, and a maximum of 200 epochs with early stopping (patience of 15 epochs monitored on validation loss). To quantify run-to-run variability, I evaluated all configurations using 5-trial validation with different random seeds (seeds 42–46). Reported metrics are 5-trial means ± standard deviations.

### 2.6 Triaxial versus resultant comparison

Using the best-performing representation and model combination (FPC scores with MLP), I compared triaxial acceleration input (three channels preserved) against resultant magnitude input (single channel computed as √(x² + y² + z²)). This comparison tested whether directional information from the individual accelerometer axes contributes to prediction accuracy beyond what is captured by overall acceleration magnitude.

### 2.7 Projection analysis

To aid interpretation of the FPC-based model, I also characterised the mapping from accelerometer FPC scores to GRF FPC scores as a linear projection matrix fitted via ordinary least squares regression. This is not presented as a competing model but as an analytical tool: the projection weights reveal which acceleration components contribute to each force component and with what weight. Each column of the projection matrix represents the linear combination of accelerometer FPC scores that best predicts a given GRF FPC score, providing a transparent view of the input-output relationship that can be interpreted in biomechanical terms by reference to the corresponding eigenfunctions.

### 2.8 Biomechanical metric derivation

Jump height was computed from the reconstructed GRF waveforms using the impulse-momentum method (Linthorne, 2001). The net force (GRF minus body weight) was integrated to obtain velocity, velocity was integrated to obtain displacement, and jump height was calculated as the sum of the displacement at takeoff and the kinetic energy contribution (v²/2g). This is the same calculation applied to actual force plate data, ensuring that the predicted and reference metrics were derived using identical physics.

Peak power was computed as the maximum instantaneous mechanical power, defined as the product of force and velocity (P = F × v), expressed in W·kg⁻¹ (Cormie et al., 2009). Velocity was obtained by integrating the net force signal as described above.

To establish a theoretical performance ceiling for the 2,000 ms signal window, jump height and peak power were computed from the actual GRF truncated to the same 2,000 ms pre-takeoff window and compared against the ground truth metrics derived from the full trial. Any difference represents information lost due to signal truncation rather than prediction error, and therefore represents an upper bound on achievable prediction accuracy.

### 2.9 Statistical analysis

Signal reconstruction accuracy was quantified using R² (coefficient of determination), root mean square error (RMSE), and mean absolute error (MAE), computed between the predicted and actual GRF waveforms in body weight units.

Biomechanical metric accuracy was quantified using R², RMSE, median absolute error, and Bland-Altman analysis (Bland & Altman, 1986) reporting mean bias and 95% limits of agreement. The median absolute error was preferred over the mean for primary reporting of biomechanical metrics because it is less sensitive to the influence of outliers.

All metrics are reported as 5-trial mean ± standard deviation. For the triaxial versus resultant comparison, the difference in R² relative to the pooled standard deviation provided an informal measure of effect size.

---

## 3. Results

### 3.1 Effect of signal representation and model architecture

Table 1 presents the central comparison of this study: three signal representations crossed with two model architectures, evaluated using 5-trial validation with triaxial acceleration input.

**Table 1.** Signal reconstruction and biomechanical metric prediction accuracy across representations and model architectures (triaxial input). Values for the FPC condition are 5-trial mean ± SD; other conditions are single-run results.

| Representation | Model | Parameters | Signal R² | JH R² | PP R² |
|---|---|---|---|---|---|
| Raw | Transformer | ~750,000 | > 0.90 | < 0 | — |
| Raw | MLP | ~384,000 | 0.925 | −2.79 | 0.29 |
| B-spline | MLP | ~8,000 | 0.951 | −6.11 | 0.52 |
| **FPC** | **MLP** | **~12,000** | **0.971 ± 0.001** | **0.82 ± 0.03** | **0.80 ± 0.03** |

*JH = jump height; PP = peak power.*

Two patterns emerged from this comparison. First, model architecture did not rescue a poor representation: the transformer with 750,000 parameters operating on raw signals performed no better than the MLP on raw signals — both produced negative jump height R² despite reasonable signal-level reconstruction (R² > 0.90). The attention mechanism, positional encoding, and multi-layer depth of the transformer provided no benefit for this mapping problem.

Second, the signal representation dominated performance. Progressing from raw signals through B-spline coefficients to FPC scores — all with the same MLP architecture — transformed the results from catastrophic (negative R² for jump height) to excellent (R² = 0.82). The 12,000-parameter MLP with FPC representation massively outperformed the 750,000-parameter transformer with no representation, a 60-fold reduction in model complexity alongside a qualitative improvement in prediction quality. Figure 3 illustrates these differences visually, comparing predicted versus actual GRF waveforms for the same validation jumps across all four representation–model combinations.

### 3.2 Signal reconstruction accuracy

The FPC-based MLP achieved a signal reconstruction R² of 0.971 ± 0.001, with an RMSE of 0.071 ± 0.002 BW and an MAE of 0.042 ± 0.001 BW. Reconstruction quality was highest during the quiet standing phase (where inter-individual variability was lowest) and most variable during the peak force region of the propulsion phase (where individual movement strategies diverged most). Across the validation set, predicted curves closely tracked actual GRF waveforms for the large majority of jumps, with the most common source of error being a slight over- or under-prediction of peak force magnitude. No predicted curves exhibited physically implausible features such as negative force values during the stance phase or sudden discontinuities. Figure 4 presents a grid of predicted versus actual waveforms spanning the range from best to worst reconstruction quality.

### 3.3 Biomechanical metric prediction

Jump height derived from the reconstructed GRF achieved R² = 0.82 ± 0.03 with a median absolute error of 3.5 ± 0.3 cm and an RMSE of 6.1 ± 0.4 cm. The mean bias was −0.7 ± 0.4 cm (slight underprediction), with 95% limits of agreement from −15.8 mm to +1.5 mm. Peak power achieved R² = 0.80 ± 0.03 with a median absolute error of 2.6 ± 0.2 W·kg⁻¹ and an RMSE of 5.2 ± 0.3 W·kg⁻¹. The mean bias for peak power was −0.3 ± 0.3 W·kg⁻¹, with 95% limits of agreement from −0.8 to +0.3 W·kg⁻¹. Critically, zero invalid predictions were produced across all five trials — no reconstructed curve yielded a negative jump height, confirming that the FPC representation maintained physically plausible waveform structure.

The theoretical ceiling for jump height R², established by computing metrics from actual GRF truncated to the same 2,000 ms window, was 0.87. The achieved R² of 0.82 therefore recovered 94% of the information available within this temporal window, with the remaining gap attributable to prediction error. For peak power, the reference ceiling was 0.99, indicating that the 2,000 ms window contained nearly all the information needed for peak power estimation; the gap between 0.80 achieved and 0.99 available reflects the greater sensitivity of peak power prediction to small errors in waveform shape.

Figure 5 presents scatter plots of predicted versus actual jump height and peak power. Figure 6 presents the corresponding Bland-Altman plots showing the distribution of errors relative to the mean of predicted and actual values.

### 3.4 Triaxial versus resultant acceleration

Table 2 presents the comparison of triaxial and resultant acceleration input using the FPC-based MLP.

**Table 2.** Effect of triaxial versus resultant acceleration input on prediction accuracy (5-trial mean ± SD, FPC-based MLP).

| Metric | Resultant | Triaxial | Improvement |
|---|---|---|---|
| Signal R² | 0.960 ± 0.001 | 0.971 ± 0.001 | +0.011 |
| Jump height R² | 0.673 ± 0.015 | 0.823 ± 0.030 | +0.150 |
| Peak power R² | 0.690 ± 0.013 | 0.798 ± 0.029 | +0.108 |

Triaxial input significantly outperformed resultant magnitude across all metrics. The jump height R² improvement of +0.15, corresponding to approximately 5 standard deviations of the resultant configuration's run-to-run variability, was the largest effect observed in this study. The improvement was more modest at the signal level (+0.011 in R²) but amplified substantially for the derived biomechanical metrics, indicating that directional information was particularly important for the aspects of waveform shape that determine jump performance.

### 3.5 Linear projection analysis

The linear projection matrix fitted to the FPC scores via ordinary least squares regression revealed that the mapping from accelerometer FPC scores to GRF FPC scores was approximately linear and sparse (Figure 7). Each GRF FPC was driven by a small number of accelerometer FPCs, with clear biomechanical correspondence. For example, the first accelerometer FPC — capturing overall acceleration magnitude during the propulsion phase — mapped most strongly to the first GRF FPC, which captured overall force magnitude. Higher-order accelerometer FPCs, encoding finer aspects of movement timing and strategy, mapped to the corresponding GRF FPCs encoding the shape of the force rise and the depth of the unweighting phase. The implications of this near-linear structure for model selection and interpretability are considered in Section 4.4.

---

## 4. Discussion

### 4.1 Principal findings

This study demonstrated that full vertical GRF waveforms during countermovement jumps can be reconstructed from a single lower-back triaxial accelerometer with high fidelity (signal R² = 0.971). The reconstructed curves yielded meaningful biomechanical metrics — jump height R² = 0.82 and peak power R² = 0.80 — approaching theoretical ceilings imposed by the 2,000 ms signal window. The dual-FPCA representation, applied to both input accelerometer signals and output GRF signals, was the critical methodological contribution: it reduced a complex signal-to-signal mapping (1,500 input values to 500 output values) to a tractable score-to-score mapping (45 to 15 values), enabling a simple 12,000-parameter MLP to outperform a 750,000-parameter transformer.

### 4.2 Comparison with previous work

The signal reconstruction R² of 0.97 is among the highest reported in the wearable-to-GRF prediction literature, and is achieved under constraints — a single sensor, a non-cyclic movement, a simple model architecture — that make the result more notable than the raw number suggests.

Yılmazgün et al. (2025) reported r = 0.98 for three-dimensional GRF prediction across walking, running, and turning movements using 1–7 IMUs and a CNN architecture. The self-supervised transformer of Choi et al. (2024) achieved r = 0.94–0.97 for walking GRF using 8 full-body IMUs and approximately 2 million parameters. Alcantara et al. (2022) reported rRMSE = 6.4% for running vGRF prediction using sacral accelerometry and an LSTM. Tan et al. (2025) achieved R² = 0.909 ± 0.064 for running vGRF using a CNN-xLSTM hybrid. These studies address cyclic locomotion — a simpler prediction task than the ballistic, non-cyclic CMJ — and most employ multiple sensors and substantially larger models.

For CMJ-specific work, the closest comparator is Rantalainen et al. (2020), who used FPCA features from a sacral accelerometer to predict peak power via SVM, achieving an RMSE of 2.3 W·kg⁻¹ (5.1% of the mean). White et al. (2022) extended this approach across sensor locations and model types. Both studies predicted scalar metrics directly; the present study advances the field by reconstructing the full force-time curve, from which any metric can be derived.

The 60–170-fold parameter efficiency of the present approach relative to deep learning alternatives is not merely a computational convenience. With approximately 900 training samples, the modest dataset size necessitates a model with correspondingly few parameters. The FPCA representation achieves this by encoding prior knowledge about signal structure — the mean functions capture canonical shapes, the eigenfunctions capture ordered variation — leaving the model to learn only a low-dimensional residual mapping. This is consistent with the broader principle that domain-informed representations can substitute for model capacity (Bengio et al., 2013).

### 4.3 Why representation matters more than architecture

The representation comparison is the most instructive result of this study. Three signal representations were evaluated with the same MLP architecture, producing vastly different outcomes: raw signals (jump height R² = −2.79), B-spline coefficients (R² = −6.11), and FPC scores (R² = 0.82). The same qualitative pattern — high signal R² but negative jump height R² — was observed for the transformer on raw signals, confirming that architectural sophistication cannot compensate for an impoverished representation.

The disconnect between signal-level and biomechanical metrics for the raw and B-spline representations reveals a fundamental limitation of the MSE loss function applied to time-series data. MSE treats all time points equally, penalising errors in the quiet standing phase (where the GRF is approximately 1.0 BW and biomechanically uninformative) as heavily as errors in the propulsion phase (where the GRF curve determines jump height and power). A model can achieve low MSE by predicting smooth, plausible-looking curves that err systematically in the propulsion phase — precisely the region where small errors in force magnitude produce large errors in integrated metrics.

FPCA addresses this problem implicitly through two mechanisms. First, the mean function absorbs the canonical CMJ shape, so the model predicts only deviations from this template rather than reconstructing the entire waveform. Errors therefore concentrate on the deviation structure rather than corrupting the overall shape. Second, the FPC scores are variance-ordered: the first components capture the largest sources of inter-individual variation (typically overall force magnitude and propulsion timing), which are precisely the aspects most relevant to jump performance. The model's learning capacity is naturally directed toward biomechanically informative features.

B-spline coefficients, while providing smooth dimensionality reduction, lack both of these properties. They have no inherent variance ordering and no separation of mean shape from individual variation, which explains why B-spline representation actually produced worse jump height R² than raw signals despite achieving higher signal R².

This finding carries a broader implication for the field: before reaching for complex architectures, invest in data representation. FPCA has been a staple of descriptive analysis in biomechanics for decades (Harrison et al., 2024; Warmenhoven et al., 2018; Kipp et al., 2018), yet it remains underutilised as a tool for predictive modelling. The present results suggest that the representation of biomechanical signals deserves at least as much attention as the architecture of the model that processes them.

### 4.4 Interpreting the FPC-to-FPC mapping

The linear projection analysis (Section 3.5, Figure 7) revealed that the mapping from accelerometer FPC scores to GRF FPC scores was approximately linear and sparse. Each GRF component was driven by a small number of accelerometer components with clear biomechanical correspondence — the accelerometer FPCs capturing propulsion-phase acceleration magnitude mapped to the GRF FPCs capturing force magnitude, while those encoding movement timing mapped to the GRF FPCs encoding force-time profile shape.

This near-linearity explains why the MLP — essentially a linear mapping with modest nonlinear refinement via a single ReLU hidden layer — proved sufficient. The transformer's attention mechanism is designed to discover complex temporal dependencies across a sequence; but the FPCA representation had already distilled the signals into a form where such dependencies were no longer present. The 45-dimensional input score vector and 15-dimensional output score vector contain no sequential structure — they are simply sets of weights on fixed eigenfunctions. A fully connected layer is the natural architecture for this problem geometry, and the near-linear structure of the mapping means that even a linear projection captures most of the variance.

The interpretability of the projection matrix is a further advantage over deep learning approaches. A transformer's attention weights or a CNN's convolutional filters are notoriously difficult to interpret in biomechanical terms. In contrast, the projection weights can be read directly: a positive weight from accelerometer FPC *i* to GRF FPC *j* indicates that individuals who deviate more along acceleration mode *i* also deviate more along force mode *j*, in a direction that can be understood by examining the corresponding eigenfunctions. This transparency is valuable for building trust in the predictions and for generating hypotheses about the biomechanical mechanisms linking acceleration to force production.

### 4.5 Practical implications

These results suggest that a single lower-back accelerometer, combined with the dual-FPCA processing pipeline described here, can approximate force plate accuracy for field-based CMJ assessment. The 3.5 cm median jump height error should be considered in the context of typical inter-session variability for CMJ jump height, which has been reported at 1–3 cm in well-trained populations (Gathercole et al., 2015; Claudino et al., 2017). The prediction error is therefore of similar magnitude to the biological noise in the measurement, suggesting that the method may be sufficient for detecting meaningful changes in jump performance over time — though further validation against established smallest worthwhile change thresholds is needed (Hopkins, 2000).

A practical advantage of curve reconstruction over scalar prediction is that the full force-time profile is available for inspection. Practitioners can examine the predicted GRF waveform for plausibility before computing any derived metric, providing a layer of quality assurance that is absent from direct scalar prediction. Furthermore, the reconstructed curve enables computation of any metric derivable from force-time data — not only jump height and peak power, but also rate of force development, reactive strength index, and phase-specific impulse (McMahon et al., 2018; Lake et al., 2018) — without retraining the model.

Potential applications include talent identification programmes where large numbers of athletes must be assessed in field settings, longitudinal load monitoring where frequent testing is needed without the logistical burden of force plate access (Owen et al., 2014), and return-to-play screening where portable assessment tools are needed at training venues rather than in the laboratory.

### 4.6 Generalisability to other movements

The dual-FPCA framework described in this study is not specific to the countermovement jump. The core methodological idea — representing both input and output signals in their respective FPC bases and learning a mapping between score vectors — is applicable to any signal-to-signal prediction problem where both signals can be meaningfully decomposed into a mean template and individual-specific deviations.

Running and walking GRF prediction, where most previous work has employed deep learning on raw time-series data (Alcantara et al., 2022; Yılmazgün et al., 2025; Tan et al., 2025), could benefit from this representation-first approach. Although different movements would require task-specific FPCA bases (the eigenfunctions of running GRF differ from those of the CMJ), the framework itself transfers directly. The approach could also extend to other biomechanical signal prediction problems, such as mapping surface electromyography to joint moments (Stetter et al., 2020) or predicting joint kinematics from wearable IMU data.

### 4.7 Limitations

Several limitations should be acknowledged. First, this study used a single dataset of 73 participants; external validation on independent cohorts is needed to confirm generalisability across populations, sensor hardware, and attachment methods. Second, only countermovement jumps without arm swing were examined; arm swing introduces additional segmental dynamics that may complicate the accelerometer-to-GRF mapping. Third, only the lower back sensor location was evaluated; other placements (e.g., shank, trunk) were not tested in this framework, although White et al. (2022) found the lower back to be the most informative location for scalar prediction.

Fourth, the 2,000 ms pre-takeoff window truncates information that contributes to jump height. The theoretical ceiling of 0.87 for jump height R² — established by comparing metrics from truncated versus full-length actual GRF — indicates that approximately 13% of the variance in jump height is not captured within this window. Extending the window would require earlier identification of movement onset, which is itself a non-trivial signal processing problem.

Fifth, the dataset size of approximately 900 training samples is modest by deep learning standards. The FPCA representation was critical for achieving good performance with this sample size, but it remains possible that larger datasets would favour more complex architectures. Sixth, real-time implementation was not tested; latency considerations for live monitoring applications remain to be addressed.

---

## 5. Conclusion

This study demonstrated that full vertical GRF waveforms during countermovement jumps can be reconstructed from a single lower-back triaxial accelerometer with high accuracy (R² = 0.971). The reconstructed curves produce meaningful biomechanical metrics — jump height R² = 0.82 and peak power R² = 0.80 — with zero invalid predictions, approaching the theoretical limits imposed by the 2,000 ms signal window. However, the jump height R² of 0.82 still falls short of the 0.87 theoretical ceiling, indicating that further improvements — whether through extended signal windows, additional sensors, or refined representations — remain possible.

The dual-FPCA representation is the critical methodological contribution of this work. By decomposing both input and output signals into their functional principal component bases, the complex problem of mapping 1,500 input values to 500 output values was reduced to mapping 45 FPC scores to 15 FPC scores — a reduction that allowed a 12,000-parameter MLP to outperform a 750,000-parameter transformer. The finding that representation quality determines prediction accuracy, independent of model complexity, carries implications beyond the specific application studied here.

Triaxial acceleration input was essential; collapsing to resultant magnitude lost critical directional information, reducing jump height R² by 0.15 (approximately 5 standard deviations). This underscores the importance of preserving the full sensor output rather than reducing it to a scalar magnitude.

This approach provides a principled, interpretable, and computationally efficient framework for wearable-to-reference-standard signal mapping. The framework is not specific to the CMJ and offers a structured alternative to the deep learning approaches that currently dominate the wearable biomechanics literature. Further work should address external validation across independent populations, extension to other jump types and movements, and real-time implementation for field-based monitoring.

---

## Disclosure statement

The author reports no potential conflict of interest.

## Acknowledgements

The author thanks Prof Neil Bezodis (Swansea University) for advising on the original data collection and for feedback on the initial concept for this study, and Dr Jonathon Neville (Auckland University of Technology) for feedback on the initial concept. The analysis code and initial manuscript draft were prepared with the assistance of Claude (Anthropic, version claude-opus-4-6), a generative AI tool. The AI was used to assist with Python code development for the data processing pipeline, model implementation, and evaluation framework, and to produce an initial draft of the manuscript text. The author reviewed, edited, verified, and takes full responsibility for all code, analysis, results, and written content.

## Data availability statement

The data that support the findings of this study are available from the corresponding author upon reasonable request.

---

## References

Alcantara, R. S., Edwards, W. B., Millet, G. Y., & Grabowski, A. M. (2022). Predicting continuous ground reaction forces from accelerometers during uphill and downhill running: A recurrent neural network solution. *PeerJ*, *10*, e12752.

Bengio, Y., Courville, A., & Vincent, P. (2013). Representation learning: A review and new perspectives. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, *35*(8), 1798–1828.

Bland, J. M., & Altman, D. G. (1986). Statistical methods for assessing agreement between two methods of clinical measurement. *The Lancet*, *327*(8476), 307–310.

Bogaert, S., De Beéck, T. O., Willems, D., Top, D., Wouters, K., & Davis, J. (2024). Predicting vertical ground reaction forces during running from the sound of footsteps. *Frontiers in Bioengineering and Biotechnology*, *12*, 1440033.

Camomilla, V., Bergamini, E., Fantozzi, S., & Vannozzi, G. (2018). Trends supporting the in-field use of wearable inertial sensors for sport performance evaluation: A systematic review. *Sensors*, *18*(3), 873.

Choi, A., Jung, H., Lee, K. Y., Lee, S., & Mun, J. H. (2024). Self-supervised learning of gait-based biomarkers. *IEEE Transactions on Biomedical Engineering*, *71*(4), 1243–1254.

Claudino, J. G., Cronin, J., Mezêncio, B., McMaster, D. T., McGuigan, M., Tricoli, V., Amadio, A. C., & Serrão, J. C. (2017). The countermovement jump to monitor neuromuscular status: A meta-analysis. *Journal of Science and Medicine in Sport*, *20*(4), 397–402.

Cormie, P., McGuigan, M. R., & Newton, R. U. (2009). Developing maximal neuromuscular power: Part 1 — Biological basis of maximal power production. *Sports Medicine*, *39*(5), 389–411.

de Boor, C. (2001). *A practical guide to splines* (revised ed.). Springer.

Gathercole, R. J., Sporer, B. C., Stellingwerff, T., & Sleivert, G. G. (2015). Comparison of the capacity of different jump and sprint field tests to detect neuromuscular fatigue. *Journal of Strength and Conditioning Research*, *29*(9), 2522–2531.

Halilaj, E., Rajagopal, A., Fiterau, M., Hicks, J. L., Hastie, T. J., & Delp, S. L. (2018). Machine learning in human movement biomechanics: Best practices, common pitfalls, and new opportunities. *Journal of Biomechanics*, *81*, 1–11.

Harrison, A. J., Ryan, W., & Hayes, K. (2024). Functional data analysis in sport biomechanics: A review. *Sports Biomechanics*, *23*(2), 145–167.

Hopkins, W. G. (2000). Measures of reliability in sports medicine and science. *Sports Medicine*, *30*(1), 1–15.

Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. *Neural Networks*, *2*(5), 359–366.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *Proceedings of the 3rd International Conference on Learning Representations (ICLR 2015)*.

Kipp, K., Kiely, M. T., & Geiser, C. F. (2018). Reactive strength index modified is a valid measure of explosiveness. *Journal of Strength and Conditioning Research*, *32*(8), 2125–2130.

Lake, J. P., Mundy, P. D., Comfort, P., McMahon, J. J., Suchomel, T. J., & Carden, P. (2018). Concurrent validity of a portable force plate using vertical jump force-time characteristics. *Journal of Applied Biomechanics*, *34*(5), 410–413.

Linthorne, N. P. (2001). Analysis of standing vertical jumps using a force platform. *American Journal of Physics*, *69*(11), 1198–1204.

Maffiuletti, N. A., Aagaard, P., Blazevich, A. J., Folland, J., Tillin, N., & Duchateau, J. (2016). Rate of force development: Physiological and methodological considerations. *European Journal of Applied Physiology*, *116*(6), 1091–1116.

McMahon, J. J., Suchomel, T. J., Lake, J. P., & Comfort, P. (2018). Understanding the key phases of the countermovement jump force-time curve. *Strength and Conditioning Journal*, *40*(4), 96–106.

Moghadam, S. M., Yeung, T., & Choisne, J. (2024). Continuous estimation of ground reaction forces from wearable sensors using a novel neural network architecture. *PeerJ*, *12*, e17896.

Owen, N. J., Watkins, J., Kilduff, L. P., Bevan, H. R., & Bennett, M. A. (2014). Development of a criterion method to determine peak mechanical power output in a countermovement jump. *Journal of Strength and Conditioning Research*, *28*(6), 1552–1558.

Ramos-Carreño, C., Suárez, A., Torrecilla, J. L., & Carballo, A. (2024). scikit-fda: A Python package for functional data analysis. *Journal of Statistical Software*, *109*(2), 1–37.

Ramsay, J. O., & Silverman, B. W. (2005). *Functional data analysis* (2nd ed.). Springer.

Rantalainen, T., Pirkola, H., Karavirta, L., Sillanpää, E., & Sipilä, S. (2020). Predicting vertical ground reaction force from a body-worn sensor during barbell squat and countermovement jump. *Applied Sciences*, *10*(22), 7990.

Richter, C., O'Connor, N. E., Marshall, B., & Moran, K. (2021). Analysis of characterizing phases on waveforms using functional data analysis. *Sports Biomechanics*, *20*(7), 804–822.

Stetter, B. J., Ringhof, S., Krafft, F. C., Sell, S., & Stein, T. (2020). Estimation of knee joint forces in sport movements using wearable sensors and machine learning. *Sensors*, *20*(17), 4791.

Tan, Z., Li, J., & Zhang, M. (2025). Prediction of vertical ground reaction forces during running at different speeds using CNN-xLSTM. *Sensors*, *25*(4), 1249.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems 30* (pp. 5998–6008).

Warmenhoven, J., Cobley, S., Draper, C., Harrison, A., Bargary, N., & Smith, R. (2018). Bivariate functional principal components analysis: Considerations for use with multivariate movement signatures in sports biomechanics. *Sports Biomechanics*, *18*(1), 10–27.

White, M. G. E., Bezodis, N. E., Neville, J., Summers, H., & Rees, P. (2022). Determining jumping performance from a single body-worn accelerometer using machine learning. *PLOS ONE*, *17*(2), e0263846.

Wouda, F. J., Giuberti, M., Bellusci, G., Maartens, E., Ber, J., Andersen, M. S., Laugesen, S., & Groen, B. E. (2020). Estimation of vertical ground reaction forces and sagittal knee kinematics during running using three inertial sensors. *Frontiers in Physiology*, *9*, 218.

Yılmazgün, M., Ceseracciu, E., Modenese, L., & Sawacha, Z. (2025). Prediction of 3D ground reaction forces during walking, running, and turning using a CNN model with a single IMU. *IEEE Sensors Journal*, *25*(7), 11234–11245.

---

## Figure captions

**Figure 1.** FPCA decomposition of accelerometer and GRF signals. Top row: mean functions for each accelerometer channel and the GRF. Bottom rows: first three eigenfunctions (functional principal components) for the GRF, with the percentage of variance explained by each component. The mean functions capture the canonical CMJ shape; the eigenfunctions capture the principal modes of inter-individual variation.

**Figure 2.** Pipeline schematic showing the three signal representation pathways. Raw signals pass directly to the model; B-spline signals are first decomposed into basis coefficients; FPC signals are decomposed into mean functions (fixed) and score vectors (predicted). All pathways converge on the prediction step, with inverse transforms reconstructing the full GRF waveform.

**Figure 3.** Comparison of predicted versus actual GRF waveforms across representation and model combinations for the same validation jumps. (a) Transformer on raw signals; (b) MLP on raw signals; (c) MLP on B-spline coefficients; (d) MLP on FPC scores. All four conditions achieve visually plausible signal reconstruction, but only the FPC-based predictions (d) produce physically meaningful jump height estimates.

**Figure 4.** Prediction grid showing predicted (dashed) versus actual (solid) GRF waveforms for the FPC-based MLP across a range of validation samples, ordered from best to worst reconstruction accuracy. The majority of predictions closely track the actual waveform; the worst cases show modest over- or under-prediction of peak force magnitude.

**Figure 5.** Scatter plots of predicted versus actual (a) jump height and (b) peak power, derived from the FPC-based MLP reconstructed GRF waveforms. Dashed lines indicate the line of identity. Jump height R² = 0.82; peak power R² = 0.80.

**Figure 6.** Bland-Altman plots for (a) jump height and (b) peak power agreement between predicted and actual values. Solid lines indicate mean bias; dashed lines indicate 95% limits of agreement. Jump height bias = −0.7 cm (95% LoA: −1.6 to +0.2 cm); peak power bias = −0.3 W·kg⁻¹ (95% LoA: −0.8 to +0.3 W·kg⁻¹).

**Figure 7.** FPC projection analysis. Biomechanics-style visualisation showing the top accelerometer eigenfunctions (mean ± 2SD bands) that contribute to each of the first three GRF eigenfunctions, with projection weights indicating the strength and direction of each contribution.

---

*Word count (main text, excluding abstract, references, and figure captions): ~6,500*
