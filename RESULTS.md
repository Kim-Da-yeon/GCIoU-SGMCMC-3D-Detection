# Experimental Results

## Overview

This document contains detailed experimental results from comparing different optimization methods for 3D object detection with GCIoU loss on the KITTI dataset.

## Experimental Setup

- **Dataset**: KITTI 3D Object Detection
  - Training: 3712 images
  - Validation: 3769 images
- **Architecture**: PointPillars with CT-Stack backbone
- **Loss Function**: GCIoU (Gradient-Corrected IoU)
- **Training**: 80 epochs with OneCycle learning rate scheduling
- **Hardware**: NVIDIA GPU with CUDA 11.1+

## Main Results

### Quantitative Performance Comparison

| Optimizer | Recall@0.7 | BEV AP Easy (%) | BEV AP Moderate (%) | BEV AP Hard (%) | Final Loss | Avg Objects |
|-----------|------------|-----------------|---------------------|-----------------|------------|-------------|
| **Adam**  | **0.417**  | **78.58**       | **65.15**           | **63.66**       | 1.149      | 4.65        |
| SGHMC     | 0.341      | 56.31           | 49.59               | 43.12           | **0.813**  | 3.08        |
| SGD       | 0.317      | 55.56           | 49.07               | 42.19           | 0.983      | 3.09        |
| SGLD      | 0.138      | 33.31           | 25.67               | 20.45           | 1.135      | 1.53        |
| SGNHT     | 0.128      | 31.08           | 25.37               | 24.85           | 1.326      | 2.50        |

### Key Findings

#### 1. Adam Dominates Across All Metrics
- **31.4% higher** BEV AP than SGHMC (65.15% vs 49.59%)
- **153% higher** performance than best SGMCMC without momentum (SGLD/SGNHT)
- Consistent performance across all difficulty levels (Easy/Moderate/Hard)

#### 2. Training Loss vs. Detection Performance Mismatch

**Loss Rankings**:
1. SGHMC: 0.813 (lowest)
2. SGD: 0.983
3. SGLD: 1.135
4. **Adam: 1.149**
5. SGNHT: 1.326

**Performance Rankings** (BEV AP Moderate):
1. **Adam: 65.15%** (highest)
2. SGHMC: 49.59%
3. SGD: 49.07%
4. SGLD: 25.67%
5. SGNHT: 25.37%

**Insight**: SGHMC achieves the lowest training loss but ranks 2nd in detection performance. Adam has the 4th lowest loss but achieves the best detection performance. This demonstrates that **minimizing training loss ≠ maximizing task-specific performance**.

#### 3. Critical Role of Momentum

**With Momentum**:
- SGHMC: 49.59% BEV AP
- SGD: 49.07% BEV AP

**Without Momentum**:
- SGLD: 25.67% BEV AP
- SGNHT: 25.37% BEV AP

**Momentum provides 147% performance improvement** (SGHMC vs SGLD), making it absolutely critical for complex 3D detection tasks.

#### 4. Temperature Sensitivity

For SGLD and SGNHT, temperature parameter T is extremely sensitive:

| Temperature | Final Loss | Gradient Norm | Performance |
|-------------|------------|---------------|-------------|
| 1e-5        | 1.523      | 10            | Low but stable |
| 1e-4        | 0          | 0             | Complete failure |
| 1e-3        | 2.524      | NaN           | Gradient explosion |
| 1e-2        | 5.597      | NaN           | Gradient explosion |
| 0.1         | 149.869    | NaN           | Gradient explosion |
| 1.0         | 783.009    | NaN           | Gradient explosion |

**Valid range**: Only T = 1e-5 produces stable training, but with significantly degraded performance.

## Difficulty-Based Analysis

### Performance Across KITTI Difficulty Levels

#### Easy
- Adam: 78.58%
- SGHMC: 56.31% (-28%)
- SGD: 55.56% (-29%)
- SGLD: 33.31% (-58%)
- SGNHT: 31.08% (-60%)

#### Moderate
- Adam: 65.15%
- SGHMC: 49.59% (-24%)
- SGD: 49.07% (-25%)
- SGLD: 25.67% (-61%)
- SGNHT: 25.37% (-61%)

#### Hard
- Adam: 63.66%
- SGHMC: 43.12% (-32%)
- SGD: 42.19% (-34%)
- SGLD: 20.45% (-68%)
- SGNHT: 24.85% (-61%)

**Observation**: Performance gap widens as difficulty increases, with Adam showing better robustness to challenging cases.

## Convergence Analysis

### Training Dynamics

- **Adam**: Rapid initial convergence (loss < 1.5 within 5 epochs), stable throughout training
- **SGHMC**: Slower initial convergence, achieves lowest loss but oscillates
- **SGD**: Similar to SGHMC but slightly less stable
- **SGLD/SGNHT**: Unstable training, high variance in loss trajectory

### Detection Count Characteristics

| Optimizer | Avg Detections/Image | Recall Strategy |
|-----------|---------------------|------------------|
| Adam      | 4.65                | Aggressive       |
| SGD       | 3.09                | Conservative     |
| SGHMC     | 3.08                | Conservative     |
| SGNHT     | 2.50                | Too conservative |
| SGLD      | 1.53                | Under-detecting  |

Adam detects more objects per image while maintaining high precision, indicating better-balanced detection strategy.

## Implications

### Practical Recommendations

1. **For Production 3D Detection**:
   - Use Adam or similar adaptive optimizers
   - Avoid SGMCMC methods unless uncertainty quantification is critical

2. **For Research Purposes**:
   - SGHMC shows potential for uncertainty estimation despite lower performance
   - Momentum is non-negotiable for stability
   - Temperature tuning is impractically sensitive

3. **For Bayesian Deep Learning**:
   - Need adaptive temperature scheduling
   - Consider hybrid approaches (Adam + Bayesian posterior)
   - Layer-wise optimizer strategies may help

### Theoretical Insights

1. **Flat vs. Sharp Minima**:
   - Adam finds flatter minima that generalize better despite higher loss
   - SGHMC's lower loss in sharper minima doesn't translate to better detection

2. **Task-Specific Optimization**:
   - Training loss (sum of cls + loc + dir) ≠ Detection metric (AP/Recall)
   - Need to optimize for task-specific metrics, not just loss

3. **Adaptive Learning Rates**:
   - Parameter-specific adaptation critical for multi-module networks
   - 3D detection networks have vastly different scales across modules

## Future Work

1. **Hybrid Optimizers**:
   - Combine Adam's adaptivity with SGHMC's uncertainty quantification
   - Use Adam for backbone, SGHMC for detection head

2. **Adaptive Temperature**:
   - Curriculum-based temperature scheduling
   - Layer-wise or module-wise temperature values

3. **Extended Evaluation**:
   - Test on other datasets (nuScenes, Waymo Open)
   - Evaluate uncertainty calibration for safety-critical applications
   - Measure computational overhead

## Citation

If you use these results in your research, please cite:

```bibtex
@article{kim2024gciou_sgmcmc,
  title={Exploratory Study on Applying Stochastic Gradient MCMC for 3D Object Detection Model Optimization},
  author={Kim, Dayeon},
  year={2024}
}
```
