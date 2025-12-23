# 3D 객체 탐지 모델 최적화를 위한 확률적 경사 MCMC 기법 적용에 대한 탐색적 연구

**Exploratory Study on Applying Stochastic Gradient MCMC for 3D Object Detection Model Optimization**

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper/research_paper.pdf)
[![KITTI](https://img.shields.io/badge/Dataset-KITTI-blue)](http://www.cvlibs.net/datasets/kitti/)
[![OpenPCDet](https://img.shields.io/badge/Framework-OpenPCDet-green)](https://github.com/open-mmlab/OpenPCDet)

## 📋 개요 (Overview)

본 연구는 자율주행을 위한 3D 객체 검출에서 **GCIoU (Generalized Complete IoU) 손실**을 사용하는 모델에 **Stochastic Gradient MCMC (SGMCMC)** 기법을 적용하여, 기존 Adam optimizer와의 성능을 비교 분석합니다.

### 주요 특징
- **Dataset**: KITTI 3D Object Detection Benchmark
- **Framework**: OpenPCDet with PointPillars architecture
- **Loss Function**: GCIoU (Gradient-Corrected IoU)
- **Optimizers Compared**: Adam, SGD, SGHMC, SGLD, SGNHT

## 🎯 연구 목적

3D 객체 탐지에서 IoU 기반 회귀 손실의 gradient vanishing 문제를 해결한 GCIoU 손실이 Adam optimizer와 함께 제안되었습니다. 본 연구는 동일한 에너지 함수를 유지하면서 SGMCMC 기법들을 적용하여 **Bayesian sampling 기반 최적화의 실용성**을 체계적으로 평가합니다.

## 🔬 실험 설정

### 하이퍼파라미터
- **Training Epochs**: 80
- **Learning Rate Schedule**: OneCycle (초기 학습률: 0.00035)
- **Batch Size**: 8
- **Weight Decay**: 0.01
- **Architecture**: CT-Stack backbone with PointPillars

### 평가 지표
- Bird's Eye View Average Precision (BEV AP)
- Recall@IoU=0.7
- KITTI 난이도별 평가 (Easy/Moderate/Hard)

## 📊 주요 실험 결과

### 정량적 성능 비교

| Optimizer | Recall@0.7 | BEV AP Easy (%) | BEV AP Moderate (%) | BEV AP Hard (%) | Final Loss |
|-----------|------------|-----------------|---------------------|-----------------|------------|
| **Adam**  | **0.417**  | **78.58**       | **65.15**           | **63.66**       | 1.149      |
| SGHMC     | 0.341      | 56.31           | 49.59               | 43.12           | **0.813**  |
| SGD       | 0.317      | 55.56           | 49.07               | 42.19           | 0.983      |
| SGLD      | 0.138      | 33.31           | 25.67               | 20.45           | 1.135      |
| SGNHT     | 0.128      | 31.08           | 25.37               | 24.85           | 1.326      |

### 핵심 발견사항

1. **Adaptive Learning Rate의 압도적 우위**
   - Adam이 BEV AP Moderate 65.15%로 최고 성능 달성
   - SGHMC 대비 +31.4% 성능 향상
   - 복잡한 3D 검출 문제에서 파라미터별 적응형 학습률이 필수적

2. **Training Loss와 Detection 성능의 불일치**
   - SGHMC: 최저 loss (0.813) 달성했으나 성능은 2위 (49.59%)
   - Adam: 4번째 loss (1.149)이지만 최고 성능 (65.15%)
   - **Loss minimization ≠ Task performance** 입증

3. **Momentum의 결정적 역할**
   - SGHMC (49.59%) vs SGLD (25.67%): **147% 성능 차이**
   - Momentum이 stochastic optimization의 안정성을 근본적으로 좌우
   - Noise injection만으로는 복잡한 3D 검출 문제 해결 불가

4. **Temperature Parameter의 극심한 민감도**
   - T ≥ 1e-3: Gradient explosion 발생
   - T = 1e-4: 학습 완전 실패 (성능 0 수렴)
   - T = 1e-5: 유일한 유효 범위이나 성능 저하
   - Bayesian method의 실용적 적용 제한

## 🏗️ 프로젝트 구조

```
GCIoU-SGMCMC-3D-Detection/
├── README.md
├── paper/
│   └── research_paper.pdf          # 연구 논문 전문
├── configs/                         # 실험 설정 파일
│   ├── pointpillars_adam.yaml
│   ├── pointpillars_sgd.yaml
│   ├── pointpillars_sghmc.yaml
│   ├── pointpillars_sgld.yaml
│   └── pointpillars_sgnht.yaml
├── optimizers/                      # SGMCMC 옵티마이저 구현
│   ├── sgld.py
│   ├── sghmc.py
│   └── sgnht.py
├── loss/                           # GCIoU 손실 함수
│   └── gciou_loss.py
├── experiments/                    # 실험 결과 및 로그
│   ├── results/
│   └── logs/
└── requirements.txt
```

## 🔧 설치 및 사용법

### 요구사항
```bash
# Python 3.8+
# PyTorch 1.10+
# CUDA 11.1+

pip install -r requirements.txt
```

### OpenPCDet 설치
```bash
git clone https://github.com/open-mmlab/OpenPCDet.git
cd OpenPCDet
pip install -r requirements.txt
python setup.py develop
```

### KITTI 데이터셋 준비
1. [KITTI 3D Object Detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) 다운로드
2. 데이터 구조 설정:
```
data/kitti/
├── training/
│   ├── calib/
│   ├── label_2/
│   ├── velodyne/
│   └── image_2/
└── testing/
    ├── calib/
    ├── velodyne/
    └── image_2/
```

### 학습 실행

#### Adam Optimizer (Baseline)
```bash
python train.py --cfg_file configs/pointpillars_adam.yaml
```

#### SGHMC
```bash
python train.py --cfg_file configs/pointpillars_sghmc.yaml \
    --optimizer sghmc \
    --temperature 1e-5 \
    --friction 0.1
```

#### SGLD
```bash
python train.py --cfg_file configs/pointpillars_sgld.yaml \
    --optimizer sgld \
    --temperature 1e-5
```

#### SGNHT
```bash
python train.py --cfg_file configs/pointpillars_sgnht.yaml \
    --optimizer sgnht \
    --temperature 1e-5 \
    --thermostat_mass 1.0
```

## 📈 SGMCMC 구현 세부사항

### SGLD (Stochastic Gradient Langevin Dynamics)
```python
θ_{t+1} = θ_t - η·∇E(θ_t) + √(2ηT)·N(0,I)
```
- 가장 단순한 SGMCMC 기법
- Random walk 특성으로 샘플 효율 낮음

### SGHMC (Stochastic Gradient Hamiltonian Monte Carlo)
```python
r_{t+1/2} = r_t - (η/2)·∇E(θ_t) - γr_t + N(0, 2γηT)
θ_{t+1} = θ_t + η·r_{t+1/2}
r_{t+1} = r_{t+1/2} - (η/2)·∇E(θ_{t+1}) - γr_{t+1/2}
```
- Momentum 변수 도입으로 더 나은 탐색
- Adam 대비 24% 낮은 성능이나 SGLD보다 147% 향상

### SGNHT (Stochastic Gradient Nosé-Hoover Thermostat)
```python
dθ = M^{-1}r·dt
dr = -∇E(θ)dt - ξr·dt
dξ = (1/Q)(||r||² - dT)dt + √(2ϵT/Q)·dW_t
```
- Thermostat 변수로 온도 자동 조절
- 추가 복잡성으로 인해 SGLD보다도 낮은 성능

## 💡 주요 기여

1. **실증적 비교 분석**
   - 복잡한 3D vision task에서 다양한 최적화 기법의 체계적 비교
   - Adaptive learning rate > Stochastic exploration 입증

2. **Flat Minima 이론의 실용적 검증**
   - Training loss minimization ≠ Task performance
   - Task-specific metric 최적화의 중요성

3. **Momentum과 Temperature의 역할 규명**
   - Momentum의 결정적 중요성 (147% 성능 차이)
   - Temperature tuning의 극심한 어려움

4. **실무적 가이드라인 제시**
   - 복잡한 vision task에서는 Adam과 같은 adaptive optimizer 권장
   - Bayesian method는 uncertainty quantification이 필수적인 경우에만 제한적 사용

## 🔮 향후 연구 방향

### Hybrid Approaches
- Adam의 adaptive mechanism + SGHMC의 uncertainty quantification 결합
- 층별 optimizer 전략 (backbone: Adam, detection head: SGHMC)

### Temperature Scheduling
- Curriculum 기반 noise scheduling
- Adaptive temperature adjustment

### Architecture-Specific Tuning
- 다양한 3D 검출 아키텍처 (SECOND, PV-RCNN) 적용
- Transformer 기반 3D detector 실험

## 📚 참고문헌

### 주요 논문
- **GCIoU Loss**: Ming et al. (2023). "Deep dive into gradients: Better optimization for 3D object detection with gradient-corrected IoU supervision." CVPR.
- **SGHMC**: Chen et al. (2014). "Stochastic gradient Hamiltonian Monte Carlo." ICML.
- **SGLD**: Welling & Teh (2011). "Bayesian learning via stochastic gradient Langevin dynamics." ICML.
- **SGNHT**: Ma et al. (2015). "A complete recipe for stochastic gradient MCMC." NeurIPS.

### 데이터셋 및 프레임워크
- **KITTI**: Geiger et al. (2012). "Are we ready for autonomous driving? The KITTI vision benchmark suite." CVPR.
- **PointPillars**: Lang et al. (2019). "PointPillars: Fast encoders for object detection from point clouds." CVPR.
- **OpenPCDet**: [GitHub Repository](https://github.com/open-mmlab/OpenPCDet)

## 📧 연락처

- **저자**: 김다연
- **소속**: [Your Institution]
- **이메일**: dayun0405@gmail.com

## 📄 라이선스

이 프로젝트는 학술 연구 목적으로 공개되었습니다. 상업적 사용 시 저자에게 문의해주시기 바랍니다.

## 🙏 감사의 말

본 연구는 다음의 오픈소스 프로젝트를 기반으로 수행되었습니다:
- OpenPCDet
- KITTI Dataset
- PyTorch

---

**Note**: 본 연구는 탐색적 연구로서, SGMCMC 기법이 복잡한 3D 객체 검출 문제에서 현재 실무적 한계를 가짐을 보여줍니다. Uncertainty quantification이 필수적인 특수한 경우가 아니라면, Adam과 같은 adaptive optimizer를 사용하는 것을 권장합니다.
