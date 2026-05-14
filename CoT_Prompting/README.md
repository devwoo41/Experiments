# Chain-of-Thought Prompting 논문 재현 실험

> Wei et al., 2022, **"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"** ([arxiv:2201.11903](https://arxiv.org/pdf/2201.11903))
>
> 이 저장소는 위 논문의 핵심 주장을 **Google Colab A100 40GB** 환경에서 오픈소스 모델만으로 재현·검증합니다.

---

## 1. 논문 한눈에

**Chain-of-Thought (CoT) Prompting**은 few-shot in-context learning에서 단순히 (Q, A) 쌍만 보여주는 대신, 정답에 도달하는 **중간 reasoning 단계 (rationale)** 를 함께 보여주는 방식입니다.

예) GSM8K
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls.
   5 + 6 = 11. The answer is 11.   ← 이 reasoning 부분이 CoT의 핵심
```

논문은 다음 세 가지를 보였습니다.
1. CoT는 reasoning 태스크에서 **표준 prompting을 크게 앞선다** (Table 1, 2, 3).
2. 이득은 **모델이 충분히 커야** 나타나는 *emergent ability*다 (Figure 4). 작은 모델에는 무용하거나 오히려 해롭다.
3. 후속 연구 *Self-Consistency* (Wang et al., 2022): CoT를 sampling으로 여러 번 돌려 **다수결** 하면 정확도가 더 올라간다.

---

## 2. 본 실험 설계

| 항목 | 값 |
|---|---|
| **추론 엔진** | [vLLM](https://github.com/vllm-project/vllm) (PagedAttention, batch sampling 효율적) |
| **Scale ladder (5개)** | `Qwen2.5-Instruct` **0.5B · 1.5B · 3B · 7B** (fp16) + **32B-AWQ** (4bit) |
| **태스크 (5개)** | GSM8K (산술) · Last Letter Concat (기호) · Coin Flip (기호) · CommonsenseQA (상식) · StrategyQA (상식) |
| **평가 문항 수** | 0.5B~7B: 각 태스크 300 / 32B: 각 150 |
| **프롬프트 조건 (3개)** | (A) Standard 8-shot · (B) CoT 8-shot · (C) Self-Consistency (CoT + sampling) |
| **Self-Consistency 샘플 수** | 0.5B~7B: n=10 / 32B: n=5 |
| **답 추출** | 논문 방식 regex (`The answer is X`) + 태스크별 fallback |

### 원 논문과의 차이 (정직한 보고)
- 원 논문은 GPT-3 (175B), LaMDA (137B), PaLM (540B) 등 비공개 거대 모델을 사용. 우리는 Colab에서 돌릴 수 있는 범위 내에서 같은 패밀리(`Qwen2.5`)로 scale 변수를 깔끔하게 잡습니다.
- **상한 32B는 논문의 emergent threshold(~62B)에 못 미칩니다**. 따라서 "급격한 emergent kink"보다는 **scale에 따른 점진적 증가 추세**를 관찰하는 실험이 됩니다. 이 한계는 결과 해석에 반영합니다.
- Qwen2.5는 instruction-tuned 모델. 논문은 base 모델 기준이라 작은 크기에서도 CoT가 어느 정도 작동할 수 있음.
- 32B는 AWQ 4bit 양자화 모델 (가중치 ~16GB, A100 40GB에 적재). fp16 대비 정확도 손실은 보통 ≤1%p 수준.
- Self-Consistency sampling 수는 원 SC 논문(n=40) 대비 시간 절약을 위해 축소.

---

## 3. 디렉토리 구조

```
CoT_Prompting/
├── README.md                    # ← 본 문서
├── requirements.txt             # Python 의존성 목록 (Colab에서는 노트북이 직접 설치)
├── notebooks/
│   └── cot_experiment.ipynb     # ★ 메인 노트북 — 이 한 파일이 실험 전체
├── src/                         # 노트북의 핵심 로직을 모듈로 분리한 참고용 코드
│   ├── datasets.py              #   데이터셋 로딩/생성
│   ├── prompts.py               #   논문 Appendix G의 8-shot exemplar
│   ├── extractor.py             #   정답 추출 regex
│   └── evaluator.py             #   정확도 / Self-Consistency 다수결
├── build_notebook.py            # 노트북 생성 스크립트 (재빌드용)
└── results/                     # 실행 후 생성
    ├── summary.json             #   모든 (모델, 태스크, 조건)의 정확도 요약
    ├── figure4_replication.png  #   논문 Figure 4 재현 plot
    ├── condition_comparison_bar.png
    └── raw/                     #   모델 응답 원본 JSONL (디버깅/재분석용)
```

> 📌 **단일 노트북 원칙**: `notebooks/cot_experiment.ipynb` 한 파일 안에 모든 코드와 설명이 들어있어 Colab에서 클릭만으로 실행됩니다. `src/`는 모듈식으로 재사용하고 싶은 경우의 참고용입니다 — 노트북 실행에는 필요 없습니다.

---

## 4. Colab에서 실행하기

### 4-1. 사전 준비

Colab 노트북에서 **런타임 → 런타임 유형 변경 → A100 GPU** 선택.

- T4로는 7B/32B fp16 적재가 어려움. 굳이 돌리려면 작은 모델만 선택 실행.
- L4 (24GB)면 0.5B~7B는 OK, 32B AWQ도 빠듯하게 가능.

API 키나 결제 설정 같은 별도 준비 사항은 없습니다 — 전부 오픈소스 모델로 돌아갑니다.

### 4-2. 노트북 실행

방법 1: GitHub에서 직접 열기 — Colab의 `파일 → GitHub` 메뉴에서 이 저장소 URL 입력 후 `notebooks/cot_experiment.ipynb` 선택.

방법 2: 노트북 파일을 로컬에 다운로드해 Colab에 업로드.

방법 3: Google Drive에 업로드 후 `.ipynb`를 더블클릭해 Colab으로 열기.

노트북 셀을 위에서부터 순서대로 실행 (`Ctrl+F9`로 전체 실행 가능).

### 4-3. 예상 소요 시간

| 모델 | 예상 시간 (A100, vLLM batch) |
|---|---|
| 0.5B / 1.5B | 각 ~20분 |
| 3B | ~30분 |
| 7B | ~1시간 |
| 32B-AWQ (150문항, SC n=5) | ~3시간 |
| **합계** | **약 5~7시간** |

Colab Pro 단일 세션(24시간)에 충분히 끝납니다. 중간에 끊겨도 `results/summary.json`에 진행상황이 누적 저장되므로 재시작 시 이어서 사용 가능.

---

## 5. 결과 읽는 법

노트북 끝부분에서 자동으로 생성되는 두 그림:

### `figure4_replication.png`
논문 Figure 4 재현. x축은 log scale 모델 크기, y축은 정확도. 태스크별 subplot, 조건별 선.

**확인할 것**:
- **주장 1**: CoT 선이 Standard 선보다 위에 있는가? (특히 GSM8K)
- **주장 2**: 모델이 커질수록 (Standard vs CoT) gap이 벌어지는가? 32B에서 가장 큰 gap을 보이는가?
- **주장 3**: Self-Consistency 선이 CoT 선 위에 있는가?

### `condition_comparison_bar.png`
가장 큰 두 모델(7B와 32B)에서 Standard / CoT / SC 정확도를 막대로 비교. 조건 간 차이를 한눈에.

### `results/summary.json`
모든 실험 셀의 raw 정확도 (`accuracy`, `correct`, `total`, `extraction_failed`, `extraction_fallback`). 추가 분석에 활용.

### `results/raw/*.jsonl`
모델 응답 원본. 한 줄에 한 문제씩, 다음 필드 포함:
- `question`, `gold` (정답), `pred` (추출된 답), `method` (추출 단계), `raw` (모델 응답 전체)

답 추출이 의심스러우면 `method` 가 `fallback_*` 또는 `failed` 인 케이스의 `raw` 텍스트를 직접 확인하면 됩니다.

---

## 6. 실험 결과 및 논문 검증

> 아래 수치는 본 실험(Qwen2.5-Instruct 0.5B~32B, A100 40GB, 2026-05 실행)의 결과입니다. 실행 환경/모델 버전에 따라 ±2~3%p 변동 가능.

### 6-1. 절대 정확도 요약 (Standard / CoT / SC, 단위 %)

| 모델 | GSM8K | Last Letter | Coin Flip | CSQA | StrategyQA |
|---|---|---|---|---|---|
| **0.5B** | 18 / 46 / 58 | 0 / 0 / 0 | 54 / 62 / 63 | 54 / 56 / 60 | 56 / 46 / 47 |
| **1.5B** | 15 / 72 / 82 | ~0 / 1 / 1 | 57 / 89 / 90 | 78 / 79 / 80 | 60 / 57 / 55 |
| **3B**   | 74 / 85 / 90 | 0 / 7 / 7    | 56 / 99 / 100 | 83 / 83 / 83 | 56 / 58 / 57 |
| **7B**   | 70 / 89 / 92 | 0 / 24 / 24  | 64 / 100 / 100 | 87 / 88 / 90 | 68 / 66 / 73 |
| **32B**  | 85 / 93 / 95 | 0 / 67 / 67  | 77 / 99 / 99 | 92 / 92 / 93 | 73 / 72 / 75 |

### 6-2. 논문 주장 vs 본 실험 (한눈에)

| 논문 주장 | 본 실험 결과 | 지지도 |
|---|---|---|
| **① CoT > Standard** (reasoning task에서 CoT가 크게 우세) | GSM8K, Coin Flip, Last Letter에서 강력 확인 (gap +20~67%p). CSQA/StrategyQA에선 효과 없음 또는 음수 — single-hop 태스크엔 reasoning이 noise | ⭐⭐⭐⭐ (태스크 의존) |
| **② Emergent ability** (작은 모델에선 CoT 무용, 큰 모델에서만 작동) | **Last Letter에서 교과서적 재현** (0→0→7→24→67%). 다른 태스크는 instruct 효과로 작은 모델에서도 CoT 작동해 극단적 kink 없음 | ⭐⭐⭐ (last_letter만 깨끗) |
| **③ Self-Consistency > CoT** (sampling 다수결로 추가 이득) | 평균 +1~+12%p로 일관되긴 함. **단 모델 크기에 반비례** (0.5B GSM8K +12%p → 32B +2%p) | ⭐⭐⭐ (작은 모델·어려운 task에 한정) |

### 6-3. 핵심 인사이트 4가지

1. **CoT가 빛나는 곳은 "multi-step이 정답에 본질적인" 태스크**
   - 효과 큼: GSM8K (산술), Coin Flip (state tracking), Last Letter (symbolic)
   - 효과 없음: CSQA (single-hop 상식), StrategyQA (yes/no)
   - StrategyQA 0.5B에서는 CoT가 오히려 **−10%p** — 작은 모델이 reasoning 형식을 못 따라가 답 추출 실패 가능성.

2. **Emergent 시점은 태스크 난이도에 따라 다르다**
   - Coin Flip: 1.5B에서 emergent 완료 (가장 빠름)
   - GSM8K: 0.5B에서도 CoT 작동 (instruct 효과로 임계점 하향)
   - Last Letter: 7B+에서 비로소 (OOD symbolic task라 임계점 높음, 32B에서도 미완성)

3. **Self-Consistency는 "작은 모델 + 어려운 reasoning"에 가장 가성비**
   - GSM8K SC 이득: 0.5B **+12%p** → 32B **+2%p**
   - CoT가 noisy할 때 다수결의 평균화 효과가 큼. 큰 모델은 이미 결정론적이라 다수결로 얻을 게 적음.
   - Coin Flip, Last Letter는 이미 ceiling 또는 floor라 SC 무용.

4. **Instruct 모델은 emergent 임계점을 크게 낮춘다**
   - 원 논문 GPT-3 base: GSM8K 0.4B에서 CoT 약 3% (효과 없음)
   - 우리 Qwen2.5-Instruct 0.5B: GSM8K CoT **46%** (이미 강하게 작동)
   - SFT 학습이 작은 모델에도 CoT 시드를 심어둠. **단 last_letter 같이 학습 분포에 없는 symbolic task에서는 여전히 large model 필수.**

### 6-4. 한 줄 결론

> **CoT는 reasoning task에 강력하고 그 이득은 scale에 따라 커진다 (원 논문 주장 지지). 다만 Instruct-tuned 모델에서는 CoT의 emergent 임계점이 크게 낮아져, 원 논문의 GPT-3 base에서 본 극단적 kink는 last_letter 같은 OOD task에서만 깨끗이 재현된다. Self-Consistency는 만능 부스터가 아니라 "작은 모델의 noisy한 CoT를 보정"하는 한정적 도구다.**

### 6-5. Deep dive — Last Letter 7B 케이스 스터디 (`notebooks/last_letter.ipynb`)

가장 흥미로운 emergent 패턴을 보인 Last Letter 태스크에서 7B의 **Standard 0% → CoT 23.7%** 차이를 raw 응답 단위로 해부했습니다 (별도 self-contained 노트북).

**관찰 1 — Standard가 0%인 이유는 task 자체를 못 풀어서**
Standard 응답의 53%가 글자 수조차 못 맞춤 (`length_short` 23% + `length_long` 30%), 43%는 글자 자체가 무관 (`wrong_letters`). "Edward Cynthia Steven Mary" → 정답 `dany` 인데 `nystrom` 같은 환각 응답. 즉 한 번에 처리할 표현이 없음.

**관찰 2 — CoT의 효과는 task decomposition**
CoT 응답은 항상 `The last letter of "Betty" is "y". ... Concatenating them is "yaat".` 형태로 4단계 펼침. 모델이 "4단어 last letter + concat"을 한 번에 못 푸니 step별로 sub-problem으로 환원시켜야 작동.

**관찰 3 — CoT 24% 한계의 진짜 정체: 글자별 step accuracy의 비대칭**

| 글자 | step 정확도 | 비고 |
|---|---|---|
| y, a, n, s, e, t, k | 96~100% | 영어 단어 끝에서 강하게 발음됨 |
| l, m | 83~91% | 중간 |
| d | 60% | 어말 약화 가능 ("Richard") |
| r | 38% | 영어 어말 r 약함 ("Jennifer") |
| **h** | **17%** | **거의 묵음** ("Joseph", "Sarah") |
| **b** | **14%** | **어말 약화** ("Robert") |
| **w** | **0%** | **완전 묵음** ("Matthew") |

→ 모델은 정확히 **영어 발음 기준의 끝소리**를 답하고 있음. "Matthew"는 `/ˈmæθ.juː/` 발음상 'w'가 들리지 않으니 모델은 'thew'에서 'e'/'h'로 답함. **CoT step accuracy 한계는 working memory 부족이 아니라 표현이 발음에 묶여 있어서** (ortho-phonetic mapping bias).

**관찰 4 — 24%의 수학적 분해가 맞아떨어짐**

```
한 단어 step 정확도 (가중평균)  ≈ 82%
4단어 모두 step 정답 (독립 가정) ≈ 0.82⁴ ≈ 45%
4 step 정답 → concat 성공률    ≈ 50~55%
최종 정답률 예측              ≈ 24%   ✓
```

Working memory 부족(`anagram` 6%)은 부수적이고, 압도적으로 글자별 step accuracy가 bottleneck.

**가설 — 32B의 67% 점프는 글자별 인식 편차의 균등화**
0.95⁴ × 80% ≈ 65% → 32B 67%와 거의 일치. **emergent의 정체는 working memory 확장이 아니라 h·w·b·r 같은 발음 영향 받는 글자도 철자 그대로 답하게 되는 representation 분리** 일 가능성. (32B raw로 같은 글자별 표를 뽑으면 직접 검증 가능.)

→ 결국 last_letter의 emergent ability는 reasoning 능력의 emergence가 아니라 **orthographic representation을 phonetic prior로부터 분리하는 능력의 emergence**로 해석됩니다.

---

## 7. 한계 (정직하게)

- **샘플링 노이즈**: 태스크당 300/150 문항이라 정확도 ±3~5%p 노이즈는 일반적. 작은 gap은 해석에 주의.
- **모델 크기 범위가 좁고 상한이 emergent threshold 미달**: 원 논문 0.4B~540B 대비 우리는 0.5B~32B. 32B도 논문이 거론한 emergent threshold(~62B)에 미달하므로 매끈한 "kink"를 그리기엔 부족.
- **Instruct 모델 효과**: Qwen2.5-Instruct는 RLHF/SFT 거친 모델이라 작은 크기에서도 CoT 효과가 어느 정도 나옵니다. 원 논문의 "0.4B에서는 거의 0%" 같은 극단적 패턴은 약하게만 나타날 수 있음.
- **32B AWQ 정확도 손실**: 4bit 양자화로 fp16 대비 보통 ≤1%p 차이. 32B의 절대 수치는 fp16 32B보다 살짝 낮을 수 있음.
- **답 추출 불완전성**: regex는 100% 정확하지 않음. `extraction_failed` / `extraction_fallback` 카운트와 raw JSONL 직접 검토를 권장.
- **Self-Consistency sampling 수 축소**: 원 SC 논문 n=40 → 우리 n=10 (32B는 n=5). 절대 정확도는 다소 과소평가될 수 있음.

---

## 8. 인용

```bibtex
@article{wei2022chain,
  title={Chain-of-thought prompting elicits reasoning in large language models},
  author={Wei, Jason and Wang, Xuezhi and Schuurmans, Dale and Bosma, Maarten and Ichter, Brian and Xia, Fei and Chi, Ed and Le, Quoc and Zhou, Denny},
  journal={Advances in neural information processing systems},
  volume={35},
  pages={24824--24837},
  year={2022}
}

@inproceedings{wang2023selfconsistency,
  title={Self-Consistency Improves Chain of Thought Reasoning in Language Models},
  author={Wang, Xuezhi and Wei, Jason and Schuurmans, Dale and Le, Quoc V and Chi, Ed H. and Narang, Sharan and Chowdhery, Aakanksha and Zhou, Denny},
  booktitle={ICLR},
  year={2023}
}
```
