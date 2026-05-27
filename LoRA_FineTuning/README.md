# LoRA Fine-Tuning — RoBERTa-base on SST-2

논문 **"LoRA: Low-Rank Adaptation of Large Language Models"** ([arXiv:2106.09685](https://arxiv.org/abs/2106.09685), Hu et al. 2021)의 LoRA 기법을 **from-scratch로 직접 구현**하고 (PEFT 라이브러리 미사용), RoBERTa-base의 attention `Wq`/`Wv`에 주입하여 **Full Fine-tuning / Head-only** baseline과 비교한 프로젝트입니다.

단일 Colab 노트북: [LoRA_FineTuning.ipynb](LoRA_FineTuning.ipynb)

## 핵심 방법 (논문 §4.1)

frozen 사전학습 weight $W_0 \in \mathbb{R}^{d\times k}$ 의 업데이트를 low-rank로 제약:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x,\quad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},\ r\ll\min(d,k)$$

- **A는 random Gaussian, B는 0으로 초기화** → 학습 시작 시 $\Delta W = BA = 0$
- $\Delta W x$를 $\alpha/r$로 스케일, $\alpha$는 고정(튜닝 안 함)
- Transformer에서는 **attention `Wq`, `Wv`에만** 적용, MLP는 freeze (논문 §4.2)
- RoBERTa-base 설정: `rq = rv = 8`, `α = 8` (논문 Table 9)

## 실행 결과 (실측)

RoBERTa-base + GLUE SST-2, Colab T4, 3 epochs.

| config | trainable params | % of full | **final dev acc** | train time | peak GPU mem |
|--------|-----------------:|----------:|:-----------------:|-----------:|-------------:|
| **LoRA (r=8)** | **887,042** | **0.71%** | **93.69%** | 413.9s | **1.52 GB** |
| Full Fine-tuning | 124,647,170 | 100% | 94.27% | 450.2s | 3.06 GB |
| Head-only | 592,130 | 0.48% | 82.11% | 163.0s | 1.16 GB |

### epoch별 dev accuracy

| epoch | LoRA | Full FT | Head-only |
|-------|------|---------|-----------|
| 1 | 0.9278 | 0.9415 | 0.8383 |
| 2 | 0.9300 | 0.9392 | 0.8268 |
| 3 | **0.9369** | **0.9427** | 0.8211 |

### 핵심 발견

1. **LoRA는 전체 파라미터의 0.71%(887K/125M)만 학습하고도 Full FT와 0.58%p 차이** (93.69% vs 94.27%). 논문의 헤드라인 — "comparable accuracy at a tiny fraction of trainable parameters" — 가 그대로 재현됨.
2. **메모리 절반**: LoRA 1.52 GB vs Full FT 3.06 GB. frozen 파라미터의 optimizer state를 저장 안 하므로 (논문 §4.2의 "reduce VRAM usage by up to 2/3").
3. **Head-only는 82.11%로 크게 뒤처짐** → attention adaptation이 실제로 기여함을 보여줌. LoRA의 작은 파라미터가 단순히 헤드만 학습하는 것보다 +11.6%p 우위.
4. LoRA epoch별 곡선이 우상향(92.78→93.69%)하는 반면 Head-only는 우하향(83.83→82.11%, 과적합) — frozen encoder의 표현력 한계.

## 논문 수치와의 비교

| | 논문 Table 2 (RoBERTa-base) | 본 구현 |
|---|---|---|
| LoRA SST-2 | 95.1 | 93.69 |
| Full FT SST-2 | 94.8 | 94.27 |

본 구현이 논문보다 ~1.4%p 낮은 이유는 **deviation 때문** (아래). 절대 수치보다 **"LoRA ≈ Full FT, 파라미터는 0.71%"라는 관계가 재현된 것**이 핵심.

## 논문 대비 Deviation (Colab 현실, 노트북에도 명시)

| 항목 | 논문 (Table 9) | 본 구현 | 이유 |
|------|---------------|--------|------|
| Epochs (SST-2) | 60 | **3** | 60 × 67k는 T4에서 수 시간; SST-2는 몇 epoch면 수렴 |
| Max seq length | 512 | **128** | SST-2 단문장이라 128로 충분, ~4배 빠름 |
| Full-FT LR | Table 9 미수록 (선행연구 재사용) | **1e-5** | 표준 RoBERTa GLUE full-FT LR |
| Random seeds | 5개 median | **1회** | 데모용 단일 run |
| A 초기화 std | "random Gaussian" (std 미명시) | **0.02** | RoBERTa 표준 init std 사용 |

위 외 모든 것 (LoRA 수식, A/B 초기화 방향, α/r 스케일, Wq/Wv 타겟, r=8/α=8, AdamW, linear decay + warmup 0.06, LoRA LR 5e-4)은 Table 9 그대로.

## 구현 핵심 — from-scratch LoRALinear

```python
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout=0.0, init_std=0.02):
        self.base = base                       # frozen W0
        self.base.weight.requires_grad_(False)
        self.scaling = alpha / r               # 논문: scale by alpha/r
        self.lora_A = nn.Parameter(torch.empty(r, in_f))    # Gaussian
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))   # zero → BA=0 at init
        nn.init.normal_(self.lora_A, std=init_std)

    def forward(self, x):
        out = self.base(x)                                  # W0 x
        lora = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return out + self.scaling * lora
```

`peft` 라이브러리를 쓰지 않고 논문 수식을 그대로 코드로 옮긴 게 핵심. RoBERTa의 `query`/`value` Linear를 이 모듈로 교체.

## Colab 실행법

1. [LoRA_FineTuning.ipynb](LoRA_FineTuning.ipynb)를 Colab에 업로드
2. 런타임 → 런타임 유형 변경 → **GPU (T4)** 선택
3. **런타임 → 모두 실행 (Run all)** — 전체 ~18분 (3 config 합산)

### 빠른 테스트
`Config`에서 `train_subset = 2000`으로 두면 1~2분 내 완료.

### 논문 수치에 더 가까이
`cfg.epochs = 60`, `cfg.max_seq_len = 512`로 두고 여러 seed 평균 (긴 T4 run 각오).

### 환경 이슈 (해결 완료, 참고용)
- GLUE 데이터셋이 HF Hub에서 `nyu-mll/glue`로 이동 → `load_dataset('nyu-mll/glue', ...)` + fallback
- Colab의 `datasets`↔`torchvision` 버전 충돌로 `set_format('torch')` 시 `VideoReader` import 크래시 → `set_format` 제거하고 `DataCollatorWithPadding(return_tensors='pt')`로 텐서 변환

## 다른 실험으로 확장

`Config`만 바꾸면 됩니다:
- **rank ablation** (논문 Table 6): `lora_r ∈ {1, 2, 4, 8, 64}` — 작은 r로도 경쟁력 있는지
- **target ablation** (논문 §7.1): `lora_targets`를 `('query',)`, `('query','key','value')` 등으로
- **다른 GLUE task**: `task`를 `'mrpc'`, `'cola'` 등으로 (단 sentence pair task는 preprocess 수정 필요)

## 출처

원 논문: Hu, Shen, Wallis, Allen-Zhu, Li, Wang, Wang, Chen. *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685, 2021 (Microsoft).
본 리포지토리는 교육·연구 목적의 비공식 재구현입니다.
