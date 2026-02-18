# Adaptive Density-Based Anomaly Detection for Non-Stationary Server Monitoring

## Overview

This project investigates density-based anomaly detection methods for real-world server CPU monitoring data. The objective is to systematically compare classical nonparametric methods (KDE), temporal sequence modeling, time-indexed density estimation, and neural conditional density models under a cost-sensitive evaluation framework.

The study emphasizes:

- Bias–variance tradeoff
- Temporal dependency modeling
- Contextual conditioning
- Cost-based threshold optimization
- Practical industrial considerations

This project was developed for:
- **AI in Industry**
- **AI in Industry Project**

---

## Dataset

We use the **Numenta Anomaly Benchmark (NAB)** dataset:

- Dataset: `realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv`
- Source: https://github.com/numenta/NAB
- Sampling rate: 5 minutes
- Duration: ~14 days
- Total observations: ~4,032
- Labeled anomaly windows: Provided via `combined_windows.json`

### Dataset Characteristics

- Clear daily periodicity
- Strong short-term temporal autocorrelation
- Anomalies occur in sustained windows (regime shifts), not isolated spikes
- Real AWS EC2 server monitoring data

---

## Problem Formulation

We model anomaly detection as **density estimation**:

An observation \( x_t \) is anomalous if:

\[
- \log p(x_t) > \tau
\]

Where:
- \( p(\cdot) \) is the estimated probability density
- \( \tau \) is a threshold optimized under a cost model

---

## Cost Model

We evaluate detection performance using a cost-sensitive framework:

\[
\text{Cost} = c_{alarm} \cdot FP + c_{missed} \cdot FN
\]

Where:
- FP = false positives
- FN = false negatives
- \( c_{alarm} \) = cost of false alarm
- \( c_{missed} \) = cost of missed anomaly

Primary evaluation uses:
# Adaptive Density-Based Anomaly Detection for Non-Stationary Server Monitoring

## Overview

This project investigates density-based anomaly detection methods for real-world server CPU monitoring data. The objective is to systematically compare classical nonparametric methods (KDE), temporal sequence modeling, time-indexed density estimation, and neural conditional density models under a cost-sensitive evaluation framework.

The study emphasizes:

- Bias–variance tradeoff
- Temporal dependency modeling
- Contextual conditioning
- Cost-based threshold optimization
- Practical industrial considerations

This project was developed for:
- **AI in Industry**
- **AI in Industry Project**

---

## Dataset

We use the **Numenta Anomaly Benchmark (NAB)** dataset:

- Dataset: `realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv`
- Source: https://github.com/numenta/NAB
- Sampling rate: 5 minutes
- Duration: ~14 days
- Total observations: ~4,032
- Labeled anomaly windows: Provided via `combined_windows.json`

### Dataset Characteristics

- Clear daily periodicity
- Strong short-term temporal autocorrelation
- Anomalies occur in sustained windows (regime shifts), not isolated spikes
- Real AWS EC2 server monitoring data

---

## Problem Formulation

We model anomaly detection as **density estimation**:

An observation \( x_t \) is anomalous if:

\[
- \log p(x_t) > \tau
\]

Where:
- \( p(\cdot) \) is the estimated probability density
- \( \tau \) is a threshold optimized under a cost model

---

## Cost Model

We evaluate detection performance using a cost-sensitive framework:

\[
\text{Cost} = c_{alarm} \cdot FP + c_{missed} \cdot FN
\]

Where:
- FP = false positives
- FN = false negatives
- \( c_{alarm} \) = cost of false alarm
- \( c_{missed} \) = cost of missed anomaly

Primary evaluation uses:
c_alarm = 1
c_missed = 2


Thresholds are optimized on the validation set.

---

## Implemented Models

### 1. Univariate KDE

Models:

\[
p(x_t)
\]

- Gaussian kernel
- Bandwidth manually selected
- Baseline method

---

### 2. Sequence-Based KDE (Sliding Window)

Models:

\[
p(x_{t-w+1}, ..., x_t)
\]

Window sizes tested:
- w = 12 (1 hour)
- w = 24 (2 hours)

Captures short-term temporal dependencies.

---

### 3. Time-Indexed KDE

Models:

\[
p_{slot}(x_t)
\]

Where `slot` = 5-minute interval of day (0–287).

Captures daily periodic structure explicitly.

---

### 4. Neural Conditional Density Model

Implements a neural network with a **Gaussian distribution head**:

\[
p(x_t \mid \text{time-of-day}, \text{day-of-week})
\]

Architecture:
- Fully connected neural network
- Outputs mean and log standard deviation
- Trained via Negative Log Likelihood (NLL)

---

### 5. Neural + Short Sequence Model

Extends neural model with lag features:

\[
p(x_t \mid x_{t-1}, x_{t-2}, x_{t-3}, \text{context})
\]

Combines contextual conditioning and short-term memory.

---

## Experimental Results (Validation Set)

Using cost parameters:
c_alarm = 1
c_missed = 2


| Model              | FP  | FN  | Cost |
|--------------------|-----|-----|------|
| Univariate KDE     | 13  | 189 | 391  |
| Time-Indexed KDE   | 39  | 162 | 363  |
| Sequence (w=12)    | 63  | 120 | 303  |
| Sequence (w=24)    | 84  | 105 | 294  |
| Neural Conditional | 32  | 170 | 372  |
| Neural + 3 Lags    | 18  | 182 | 382  |

---

## Key Insights

1. Sequence-based KDE significantly outperforms univariate and time-indexed models.
2. Neural conditional modeling improves over simple KDE by capturing contextual interpolation.
3. However, in this dataset, short-term temporal dependencies dominate anomaly detection performance.
4. Parametric neural models underperform high-dimensional KDE when anomaly structure is driven by sustained temporal pattern shifts.
5. Increasing model flexibility improves detection but increases variance and computational cost.

---

## Technical Stack

- Python
- NumPy
- Pandas
- Scikit-learn
- PyTorch
- Matplotlib

---

---

## Conclusion

This project demonstrates that:

- Nonparametric sequence modeling is highly effective for detecting sustained anomalies in server monitoring data.
- Neural conditional density estimation provides smooth contextual modeling but requires sufficient temporal features to compete with sequence-based methods.
- Cost-sensitive threshold optimization is essential for industrial anomaly detection.

The comparative framework provides a structured evaluation of density-based methods in a real industrial monitoring scenario.

---

## Future Work

- Conditional Normalizing Flows
- Longer neural lag windows
- Drift-adaptive training
- Multi-server joint modeling
- Real-time deployment evaluation

---

## Author

Alireza  Shahidiani




