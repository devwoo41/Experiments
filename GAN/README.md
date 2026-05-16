# Generative Adversarial Nets — 재현 실험

Goodfellow et al., *Generative Adversarial Nets* (NeurIPS 2014, [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)) 의 핵심 — Algorithm 1, two-player minimax 게임의 fixed point ($D^* = 1/2$), 그리고 GAN 학습 특유의 병리들 (saturating gradient, mode collapse) — 을 MNIST 위에서 직접 재현한 기록.

노트북: [gan_experiment.ipynb](gan_experiment.ipynb)

---

## 1. 실험 셋업

| 항목 | 값 |
|---|---|
| 데이터셋 | MNIST (50,000 train + 10,000 test, 픽셀 [0, 1]) |
| Generator | MLP — z(100) → 1200 ReLU → 1200 ReLU → 784 sigmoid |
| z 분포 | $z_i \sim \mathcal{U}(-1, 1)$ |
| Discriminator | MLP with **Maxout** (240 units, 5 pieces) + dropout 0.5 + sigmoid out |
| G loss (default) | **Non-saturating**: $-\log D(G(z))$ |
| D loss | $-(\log D(x) + \log(1 - D(G(z))))$ |
| Optimizer | Adam(lr=2e-4, β=(0.5, 0.999)) — 논문은 SGD+momentum, 안정성 위해 DCGAN 셋업 사용 |
| Batch / k | 100 / 1 (논문) |
| Epochs (vanilla) | 60 |
| Hardware | Colab T4 / A100 — 1회 약 15\~20분 |

---

## 2. Vanilla GAN — 정상 학습 (60 epochs)

### 2.1 학습 곡선

**G/D loss**:
- G loss 가 학습 초반 (epoch 3\~4 부근) 에 약 6 까지 치솟았다가 점차 감소해 epoch 60 에서 **~0.9** 로 수렴.
- D loss 는 epoch 5 부근에서 최저점 (~0.2) 을 찍고, G 가 개선되면서 점차 상승해 **~1.3** 로 수렴.
- 두 곡선이 epoch 25 부근에서 교차한 후 분리되는 패턴 — GAN 학습이 건강할 때의 전형적 형태로, G 가 D 를 따라잡아 D 의 작업이 점점 어려워지고 있다는 신호.

**Discriminator confidence** (Algorithm 1 의 fixed point 진단):
- $D(x)$ 가 시작 시 0.8 부근에서 점차 **0.55** 로 하강.
- $D(G(z))$ 가 시작 시 0.2 부근에서 점차 **0.45** 로 상승.
- **두 값이 모두 0.5 (= 논문 Prop. 2 의 $D^* = 1/2$) 로 수렴** — 게임이 nash equilibrium 부근에 도달했음을 의미. 이는 vanilla 셋업이 이론대로 작동했다는 가장 직접적인 증거.

### 2.2 epoch 별 샘플 진화

| epoch | 관찰 |
|---|---|
| 1 | 형체 없는 회색 blob. G 가 아직 sigmoid 출력의 분포 자체를 못 배운 상태. |
| 10 | 거의 모두 1-처럼 생긴 수직 스트로크 — G 가 우선 가장 단순한 모드부터 학습. |
| 20 | 다양한 숫자 형태가 등장하기 시작하지만 흐릿함. |
| 35 | 0, 1, 3, 7, 8, 9 등 대부분의 숫자가 식별 가능. |
| 45\~60 | 안정된 품질. 일부 샘플은 진짜 MNIST 와 구별이 어려울 정도. |

### 2.3 Parzen window log-likelihood (논문 Table 1 metric)

10,000개 생성 샘플 위에서 isotropic Gaussian Parzen window. σ 는 1,000 sample validation set 으로 탐색:

| σ | val mean log p (nats) |
|---:|---:|
| 0.10 | −429.52 |
| 0.15 | 88.77 |
| 0.17 | 138.79 |
| **0.20** | **155.90 ← best** |
| 0.23 | 138.09 |
| 0.25 | 116.43 |
| 0.30 | 47.09 |

**MNIST test set 평균 log-likelihood = 153.36 ± 1.71 nats**
(논문 Table 1 보고치: **225 ± 2 nats**)

**Δ = −72 nats**. 차이 원인 추정:
1. **Training epoch 수**. 논문은 훨씬 더 길게 학습 (수백 epoch). 60 epoch 만 돌렸으므로 G 가 아직 분포를 완전히 fit 하지 못함.
2. **Optimizer 차이**. 논문 SGD+momentum vs 본 실험 Adam — 같은 fixed point 라도 수렴 trajectory 가 다르므로 같은 step 수에서 도달 지점이 다를 수 있음.
3. **Parzen window 의 noise**. 이 metric 자체가 고차원에서 매우 노이지하다는 게 후속 연구 (Theis et al. 2016) 에서 밝혀짐. 시각적 품질 차이만큼 정량 차이가 크게 보일 수 있음.

153 nats 라는 절댓값은 random 베이스라인 (~−1000 nats 이하) 대비 훨씬 위쪽이고, 시각적 샘플 품질과 일관됨 — **방향성은 재현**.

---

## 3. 학습 불안정성의 의도적 재현

논문 §3 이 경고한 두 가지 병리를 강제 셋업으로 재현해서, GAN 학습이 *왜* 까다로운지 정량/시각으로 확인.

### 3.1 Saturating G loss + 강한 D — 그래디언트 소실 (catastrophic)

**셋업**: G 의 loss 를 논문 Eq. 1 원형 ($\min_G \log(1 - D(G(z)))$, saturating) 으로 두고, D 의 학습률을 G 보다 5배 강하게 (1e-3 vs 2e-4).

**관찰**:
- **Epoch 1 안에 모두 0 으로 붕괴**. D loss = 0.000, G loss = 0.000.
- $D(x) = 1.000$, $D(G(z)) = 0.000$ — D 가 *완벽하게* 진짜와 가짜를 구분.
- 이 상태에서 saturating loss 는 $\log(1 - D(G(z))) \to \log(1 - 0) = 0$ 에서 평평해져 **G 의 gradient 가 정확히 0**. G 는 학습 신호를 받지 못하고 epoch 30 까지 첫 epoch 의 노이즈 blob 그대로 멈춰 있음.
- 생성 샘플: epoch 1, 5, 10, 15, 20, 30 이 거의 픽셀 단위로 동일.

**이게 바로 논문 §3 이 non-saturating ($\max_G \log D(G(z))$) 을 권장한 정확한 이유** — saturating 형식은 fixed point 가 같아도 학습 초기 dynamics 가 deadlock 으로 끝남.

### 3.2 k=5 (D over-training) — Mode collapse

**셋업**: 모든 것 vanilla 와 같되, D 를 G 보다 5번 더 자주 업데이트 (Algorithm 1 의 $k=5$). 논문이 $k=1$ 을 쓴 이유를 보기 위함.

**관찰**:
- G loss 가 epoch 0 의 ~8 에서 epoch 30 의 ~2 까지 계속 떨어지긴 하지만, **vanilla 의 0.9 와 비교하면 한참 높음** — G 가 D 의 강력함을 따라가지 못함.
- $D(x) \approx 0.8\text{–}1.0$, $D(G(z)) \approx 0\text{–}0.2$ — D 가 일방적으로 우세, 두 값 모두 0.5 와는 멀어 nash equilibrium 미달성.
- 샘플 시각화: epoch 1\~15 거의 전부 수직 스트로크 (= "1") 만 생성. epoch 20\~30 에서 일부 다른 모드가 등장하지만 여전히 1 이 다수.

### 3.3 Mode coverage 정량 비교 (분류기 기반)

별도로 학습한 MNIST 분류기 (LeNet 비슷, test acc ~99%) 로 각 셋업의 generator 가 만든 10,000장의 예측 클래스 분포를 측정.

| Digit | vanilla | saturating | k=5 |
|---:|---:|---:|---:|
| 0 | 8.87% | 0.00% | 0.86% |
| 1 | 11.74% | 0.00% | **53.72%** |
| 2 | 8.71% | 0.00% | 1.19% |
| 3 | 10.12% | **100.00%** | 10.62% |
| 4 | 11.67% | 0.00% | 8.57% |
| 5 | 8.11% | 0.00% | 4.87% |
| 6 | 9.48% | 0.00% | 4.21% |
| 7 | 9.78% | 0.00% | 4.77% |
| 8 | 10.07% | 0.00% | 5.12% |
| 9 | 11.45% | 0.00% | 6.07% |

**해석**:
- **Vanilla**: 모든 클래스가 8\~12% 사이 — uniform 10% 라인 부근. 10개 모드 다 커버 ✓
- **Saturating**: 100% 가 "3" 으로 분류 (실제로는 무의미한 노이지 blob 인데 분류기 입장에서 3 에 가장 가까운 모양). **단일 모드로 완전 붕괴**.
- **k=5**: **53.72% 가 "1"**, 0/2 는 거의 0%, 나머지 클래스는 4\~10% 로 짜부라짐. 1번 모드 위주의 **부분적 mode collapse**.

---

## 4. 종합 판단

### 4.1 성공한 것 ✓

1. **Algorithm 1 코드화** — Algorithm 1 의 한 step 한 step (k D updates, 1 G update, non-saturating G loss) 을 그대로 옮긴 학습 루프가 60 epoch 안에 인식 가능한 MNIST 숫자 생성.
2. **Fixed point 진단 ($D \to 1/2$)** — vanilla 셋업에서 $D(x), D(G(z))$ 가 모두 0.5 근처로 수렴, Prop. 2 의 이론적 fixed point 와 일치하는 동작.
3. **Parzen log-likelihood 의 방향성 재현** — 153.36 nats. 절댓값은 논문보다 낮지만 random 베이스라인 대비 수백 nats 위. 학습 epoch 늘리면 더 올라올 여지.
4. **학습 병리의 정량 재현** — 논문 §3 이 *말로만* 설명한 saturating loss 의 deadlock 과 k 과잉의 mode collapse 를, *수치와 그래프* 로 직접 보임.

### 4.2 절반-성공 / 한계 △

1. **Parzen log-likelihood 절댓값 갭** (153 vs 225). epoch 수 / SGD vs Adam / Parzen 자체의 noise 가 섞여 있어 어디에 책임이 가장 큰지는 본 실험으로는 분리 못함.
2. **Vanilla 의 최종 샘플** 도 일부는 식별 불가능하거나 깨진 형태 — 60 epoch 가 충분히 길지 않을 수 있음.

### 4.3 한 줄 결론

> Goodfellow 2014 의 **알고리즘과 이론 (특히 fixed point 와 saturating loss 의 함정) 은 충실히 재현**. **Parzen 정량 metric 의 정확한 수치 매칭은 부분적**.

---

## 5. 다음에 해 볼 만한 것

- **SGD + momentum 으로 재현** — 논문 그대로 SGD 로 돌리고 lr/momentum 스케줄을 따라가 보기. Adam 과의 Parzen log-lik 차이가 얼마나 되는지.
- **Training 더 길게** (200\~300 epoch) — Parzen 153 → 225 까지 좁힐 수 있는지.
- **현대 metric (FID, IS)** 로 §3 의 collapse 정도를 정량 비교.
- **CIFAR-10 + DCGAN** — vanilla MLP 의 한계를 CNN 으로 넘는 시도.
- **Spectral norm / WGAN-GP** 같은 안정화 기법으로 §3.1, §3.2 의 셋업이 회복되는지 ablation.

---

## 6. 실행 방법

1. Colab T4 / Pro+ A100 어느 쪽이든 OK (T4 권장 — 가성비).
2. 노트북 [gan_experiment.ipynb](gan_experiment.ipynb) 를 열고 위에서 아래로 실행.
3. §6 의 vanilla 학습 (60 epoch) ~15\~20분 → §8 Parzen 평가 → §9.1, §9.2 의 실패 셋업 (각 30 epoch) → §9.5 의 mode coverage 막대그래프.
4. §11 은 학습된 G 로 PNG 저장 + 분류기 기반 갤러리 + best-of 필터 (선택, 본 분석에는 필요 없음).
