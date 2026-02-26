# Data Quality Exclusions

This document records accelerometer data quality issues identified through post-hoc inspection of the CMJ dataset, and the resulting trial exclusions.

## Background

The dataset was collected as part of the study reported in White et al. (2022). Triaxial accelerometer data were recorded from Delsys Trigno sensors (250 Hz) attached to the participant's lower back. Two categories of sensor artefact were identified by examining peak resultant acceleration magnitudes and checking for runs of consecutive identical axis values.

The inspection was performed using `scripts/inspect_data_quality.py`.

## Issues Identified

### 1. Participant 39 — Suspected Sensor Miscalibration (16 trials excluded)

All 16 trials from Participant 39 show systematically low peak resultant acceleration (2.64–3.08 g) despite jump heights of 0.43–0.55 m. Participants with comparable jump heights typically produce peak resultant accelerations of 5–8 g — roughly double the values recorded for this participant.

The most likely explanation is an error in the manual analogue-to-digital calibration of the sensor unit used for this participant, resulting in acceleration values approximately half their true magnitude. Because the scaling factor is unknown, these trials cannot be corrected and are excluded.

**Trials excluded:** All 16 (8 arms, 8 noarms)

### 2. Participants 4 and 19 — ADC Clipping on X-axis (12 trials excluded)

Several trials from Participants 4 and 19 show runs of 21–31 consecutive identical values on the X-axis accelerometer channel, locking at approximately −2.18 g. This is a classic analogue-to-digital converter (ADC) saturation artefact: the true acceleration exceeded the sensor's measurement range and was truncated at the ADC rail. The affected portions of the signal do not represent the true acceleration, which would have been more negative than the recorded value.

**Participant 4:** Clipping occurs in all 4 arms trials but not in the 4 noarms trials. The arm swing condition likely produced higher accelerations that exceeded the sensor's range. The 4 clean noarms trials are retained.

**Participant 19:** Clipping occurs in all 8 trials (both conditions), suggesting a persistent issue — possibly the sensor orientation placed the gravity vector more heavily on the X-axis, or the sensor unit had a lower dynamic range.

**Trials excluded:** 4 (Participant 4, arms) + 8 (Participant 19, all) = 12

### 3. Participant 75 — Investigated, Retained

Participant 75's noarms trials (peak resultant 3.1–4.6 g) are notably lower than their arms trials (5.9–6.8 g). This was investigated but attributed to genuine biomechanical differences: jump order was randomised, no sensor reattachment occurred between conditions, and some participants land more softly without the arm swing. This participant's data are retained.

## Summary

| Issue | Participant | Trials Excluded | Reason |
|-------|-------------|-----------------|--------|
| Miscalibration | 39 | 16 | Peak ACC ~50% of expected |
| ADC clipping | 4 (arms only) | 4 | X-axis saturation at −2.18 g |
| ADC clipping | 19 (all) | 8 | X-axis saturation at −2.18 g |
| **Total** | | **28** | |

After exclusion: **691 − 28 = 663 trials** from **67 participants** (Participant 4 retained with noarms trials only).

## Reproduction

To verify these findings:

```bash
python scripts/inspect_data_quality.py --data-path data/cmj_dataset_both.npz
```

To generate a cleaned dataset with these exclusions applied (default behaviour):

```bash
python scripts/prepare_dataset.py --conditions both
```

To generate a dataset without quality exclusions (original behaviour):

```bash
python scripts/prepare_dataset.py --conditions both --no-exclude-quality
```
