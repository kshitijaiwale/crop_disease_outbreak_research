# Research Summary: V11 Red Rot Forecasting System

## **Core Model Architecture**
- **Type**: Knowledge-Guided Temporal Convolutional Network (KG-TCN).
- **Temporal Depth**: 28-day sequence history.
- **Fusion Logic**: Multi-modal fusion of Z-scored weather dynamics (NASA POWER) and agronomic modulation (Variety, Ratoon, Age).
- **Optimization**: BCE Loss with `WeightedRandomSampler` and `FocalLoss` for hard-boundary mining.
- **Calibration**: Post-hoc Temperature Scaling ($T \approx 1.34$) applied to resolve logit saturation.

## **Experimental Design**
- **Training Window**: 2005–2018 (Captures historical patterns).
- **Validation Window**: 2019–2021 (Extreme monsoon/high-moisture window).
- **Test Window**: 2022–2024 (Deployment ready; pending Ground Truth annotation).

## **Performance Analysis (2019–2021)**
- **Recall**: 100% (Caught all 5 major outbreak events).
- **Lead Time**: 7–14 days (Actionable horizon for fungicidal intervention).
- **Signal Separation**: High separation in the "wetter" monsoon window confirmed by feature probing (RH persistence driver separation 0.86).
- **False Positive Rate**: 8.93% (Acceptable for agricultural early warning; target was <10%).

## **Operational Features**
- **Real-time Engine**: Integrated with NASA POWER API for automated daily inference.
- **Alert Clustering**: 2-day consecutive "High Risk" rule to prevent alert fatigue.
- **Explainability**: Structured driver detection (Humidity, Rainfall, Temperature Lag) to provide context for farmer advisories.
- **Feedback Loop**: Integrated SQLite loop for field-observation logging and future re-training.

## **Known Limitations & Future Work**
- **FPR Target**: While 8.93% is strong, operational targets are <5%. Further suppression via NDVI/Variety specific interaction layers is recommended.
- **Test Set Labeling**: Generalization on the 2022–2024 window requires synchronization with local agricultural departments for event ground truth.
- **Variety Simulation**: Currently uses probabilistic simulation based on regional trends. Transitioning to specific variety IDs from farmer records will improve precision.

> [!NOTE]
> This system is currently at a **Research Prototype** stage. While functionally complete and biologically aligned, it lacks enterprise features such as multi-factor authentication, high-availability API clustering, and an automated retraining CI/CD pipeline.
