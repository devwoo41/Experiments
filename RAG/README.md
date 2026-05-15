# RAG: Retrieval-Augmented Generation 재현 실험

Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (NeurIPS 2020, [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)) 의 핵심 결과 — RAG-Sequence (Eq. 1) 와 RAG-Token (Eq. 2) 의 디코딩 차이 및 Natural Questions open-domain QA 의 Exact Match — 를 Colab A100 80GB 환경에서 직접 재현해본 기록.

노트북: [rag_experiment.ipynb](rag_experiment.ipynb)

---

## 1. 실험 셋업

| 항목 | 값 |
|---|---|
| Retriever | DPR (`facebook/dpr-question_encoder-multiset-base`, `dpr-ctx_encoder-multiset-base`) |
| Generator | BART-large |
| Knowledge source | `wiki_dpr` (psgs_w100.multiset.compressed) — 21M passages |
| FAISS 인덱스 | OPQ-compressed (RAM ~30GB; `exact` 는 RAM ~65GB) |
| 사전학습 RAG | `facebook/rag-sequence-nq`, `facebook/rag-token-nq` |
| Task / 평가 | Natural Questions open (NQ-open) — dev 처음 500문항 (full dev = 3,610) |
| Decoding | `num_beams=4`, `n_docs=5`, `max_new_tokens=20` |
| Hardware | Colab Pro+ A100 80GB GPU, 167GB RAM, local-scratch 368GB |

노트북은 두 모드로 구성:
- **Mini 모드 (§4)**: 16개 패시지의 손-큐레이션 코퍼스 위에서 RAG-Sequence/RAG-Token 디코딩을 HF 의 RAG 클래스를 쓰지 않고 직접 구현 (Thorough decoding, Eq. 2 의 토큰별 marginalize). 수식이 코드로 어떻게 번역되는지 보는 용도.
- **Faithful 모드 (§5–6)**: wiki_dpr 21M 패시지 + 사전학습 RAG 체크포인트 + NQ-open 평가. 논문 수치와 비교 가능한 정량 결과.

---

## 2. 정량 결과 (NQ-open dev 500문항)

| Model | NQ EM (paper) | NQ EM (this run, n=500) | Δ |
|---|---:|---:|---:|
| Closed-Book T5-11B + SSM | 36.6 | — | — |
| REALM | 40.4 | — | — |
| DPR (extractive) | 41.5 | — | — |
| **RAG-Token** (facebook/rag-token-nq) | **44.1** | **37.0** | **−7.1** |
| **RAG-Sequence** (facebook/rag-sequence-nq) | **44.5** | **0.0** | **−44.5** |

RAG-Token 의 −7.1pt 차이는 (1) 평가 슬라이스가 dev 의 1/7 (500/3,610) 라서 분산이 크고, (2) `num_beams=4`, `max_new_tokens=20` 이 논문 보고 셋팅보다 작을 수 있어서이며 — 동일한 모델 가중치를 쓴다는 점에서 plausible 범위 안.

RAG-Sequence 의 0.0 은 **재현 실패**이며, 이유는 모델 가중치 자체가 아니라 추론 호출 측 — 자세한 분석은 §4 에서.

---

## 3. RAG-Sequence vs RAG-Token 정성 비교 (샘플)

같은 질문에 대해 두 변형이 어떻게 답하는지:

| Q | Gold | RAG-Sequence | RAG-Token |
|---|---|---|---|
| when was the last time anyone was on the moon | 14 December 1972 UTC | `National homelessnessFel Bav Lund apparFINrontal Todd ...` | `14 december 1972 utc` ✓ |
| who wrote he ain't heavy he's my brother lyrics | Bobby Scott / Bob Russell | `bond [\| errone Daw nont(); Macroigg ● walk Anth ...` | `bob russell` ✓ |
| how many seasons of the bastard executioner are there | one | `disguiseGeneric disguise Shuliamconservative ...` | `one` ✓ |
| when did the eagles win last super bowl | 2017 | `Capemoon� heavily appliance migrant depend metabol ...` | `2017` ✓ |
| who won last year's ncaa women's basketball | South Carolina | `Openingeven piled dwindlingAdditional TH sew Euguart ...` | `south carolina` ✓ |
| when did the isle of wight become an island | During the last Ice Age | `Memphis Memphis Memphis Memphis Memphis Memphis ...` | `1890` ✗ |
| love yourself by justin bieber is about who | Rihanna | `financediang [\| eject [\|knowledgeselect [\| ...` | `former lover` ✗ |
| who was the ruler of england in 1616 | James I | `Editors entitlement [\| Rid pitch victory warp ...` | `charles i` ✗ |

**관찰**:
- RAG-Token 은 정답이든 오답이든 **plausible 한 짧은 답** 을 깔끔하게 내고, retrieval 이 맞으면 답도 맞음 (eagles → 2017, ncaa women's → south carolina, eagles 등).
- RAG-Sequence 의 출력은 **완전 gibberish**. BART decoder 가 context-grounded 가 아닌 거의 random sample 처럼 동작. 같은 가중치 (`rag-sequence-nq` checkpoint) 인데도 이런다는 건 model 자체가 아니라 추론 경로 (HF `generate()` 호출 인자) 문제일 가능성이 높음.

---

## 4. RAG-Sequence 가 0% 가 나온 이유 (분석)

같은 노트북에서 RAG-Token 은 잘 도는데 RAG-Sequence 만 깨진 점이 진단의 핵심:

1. **가중치는 정상**. `facebook/rag-sequence-nq` 가 잘못 다운로드됐으면 그 출력이 BART pretraining 분포 (영어 문장) 에 가까워야 하는데, 출력은 그것보다도 더 노이지함 → 단순 가중치 문제는 아닐 가능성.
2. **Retriever 도 정상**. 두 모델이 같은 retriever 인스턴스를 공유하고 있고, RAG-Token 쪽은 정상이라 retriever 자체는 문제 없음.
3. **호출 경로 차이**. `RagSequenceForGeneration.generate()` 와 `RagTokenForGeneration.generate()` 는 HF 안에서 다른 디코딩 루틴을 탐:
   - **Token 변형**: BART decoder 의 매 스텝마다 K개의 encoder context 에 대한 logit 을 marginalize. 일반 beam search 와 인터페이스가 거의 동일.
   - **Sequence 변형**: K개의 doc-conditional beam search 결과를 모은 뒤 **Thorough decoding** 으로 re-scoring. 이 과정에서 `num_beams`, `num_return_sequences`, `n_docs`, `max_length` vs `max_new_tokens` 등 인자가 더 민감.
4. **가장 의심되는 한 가지**: 노트북에서는 `max_new_tokens=20` 으로 호출했는데, transformers 4.44 의 `RagSequenceForGeneration.generate()` 는 내부에서 이 인자를 호환되지 않게 처리하거나 `max_length` 로 덮어쓸 가능성이 있음. 결과적으로 BART decoder 가 context 없이 짧은 시퀀스를 sampling 하는 비슷한 상태가 됐을 가능성.

**다음 수정 후보** (재현 시도 시 가장 먼저 해볼 것):

```python
# 셀 25 의 rag_answer 를:
gen = model.generate(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'],
                     num_beams=num_beams, num_return_sequences=1,
                     max_length=max_new_tokens + 4,   # max_new_tokens 대신 max_length
                     min_length=2,
                     n_docs=n_docs,
                     early_stopping=True)
```

그리고/또는 RAG-Sequence 만 분리해서 `RagSequenceForGeneration` 의 `do_marginalize=True` / `do_deduplication=True` 명시.

이 한 가지를 바꿔서 RAG-Sequence EM 이 정상 범위 (paper 의 44.5 부근) 로 올라오는지가 다음 검증.

---

## 5. RAG 의 두 수식이 어떻게 다른지

논문의 핵심 차이는 **언제 retrieval 을 marginalize 하느냐**:

**RAG-Sequence (Eq. 1)** — 시퀀스 전체가 같은 문서를 본다:

$$p_{\text{RAG-Seq}}(y\mid x) \;\approx\; \sum_{z\in\text{top-}K} p_\eta(z\mid x)\, \prod_{i=1}^{N} p_\theta(y_i\mid x, z, y_{<i}).$$

**RAG-Token (Eq. 2)** — 토큰마다 다른 문서를 볼 수 있다:

$$p_{\text{RAG-Tok}}(y\mid x) \;=\; \prod_{i=1}^{N}\; \sum_{z\in\text{top-}K} p_\eta(z\mid x)\, p_\theta(y_i\mid x, z, y_{<i}).$$

Mini 모드 (노트북 §4.3–4.4) 에서 두 식의 디코딩을 손으로 구현해서, RAG-Sequence 는 문서별 beam search → 후보 union → re-scoring (Thorough), RAG-Token 은 매 스텝의 K-doc-marginalized 분포 위의 beam search 로 동작함을 확인.

---

## 6. 구현 시 부딪힌 함정들 (다음 사람에게)

이번 재현 과정에서 실제로 부딪혔던 이슈 — 다 노트북에 반영됨:

1. **wiki_dpr 디스크 OOM**. `compressed` 인덱스도 train split 생성 도중 60GB 이상 부풀어서 Colab 의 기본 / 디스크 (~235GB) 를 가득 채워 터짐. → 셀 4 가 `df` 로 가장 여유 큰 마운트 자동 탐색해서 `~/.cache/huggingface` 를 거기로 심볼릭링크 (Pro+ 의 `/mnt/disks/local-scratch` 활용).
2. **wiki_dpr 인덱스 RAM 중복 적재**. `RagRetriever.from_pretrained(...)` 를 두 RAG 모델에 각각 하면 인덱스가 RAM 에 2copies (`exact` 면 ~130GB). → retriever 하나를 두 모델이 공유.
3. **`trust_remote_code` forward 누락**. datasets ≥ 2.20 부터 script-based dataset 은 명시 동의가 필요한데 `RagRetriever` 가 이를 내부 `load_dataset` 으로 전달하지 않음. → `HF_DATASETS_TRUST_REMOTE_CODE=1` 환경변수 설정.
4. **numpy ABI 충돌**. `faiss-cpu<1.9` 가 `numpy<2` 를 강제 핀해서 Colab 의 기본 numpy 2.x 가 다운그레이드되고, torch 등 numpy-2-로-컴파일된 다른 C 확장과 ABI 충돌 ("Expected 96 from C header, got 88 from PyObject"). → `faiss-cpu>=1.9.0`.
5. **입력 포맷**. HF `RagRetriever.postprocess_docs` 의 실제 포맷은 `<title> / <text> // <question>` 이지 BART-NLI 의 `</s></s>` 가 아님. Mini 모드를 사전학습 RAG 와 일치시키려면 후자가 필요.
6. **BART decoder 의 leading token**. BART 는 `decoder_start_token_id == eos_token_id == </s>` 이고 `bos_token_id == <s>` 가 따로 있어, `generate()` 출력은 `[</s>, <s>, t1, ..., </s>, (pad)]` 형태. Custom Thorough-decoding 에서 label 로 쓰려면 leading `</s>` 만 떼고 사용해야 정합.

---

## 7. 한계 / 다음에 해볼 것

- **RAG-Sequence 의 0% 가 우선 수정 대상**. §4 의 `max_length` 수정 후 EM 이 paper 부근으로 올라오는지 확인.
- **`exact` 인덱스로 재실행**. compressed 와 exact 의 EM 차이가 RAG-Token 에서 얼마나 되는지 quantify. 논문은 exact (IndexFlatIP) 를 씀.
- **전체 NQ-open dev (3,610)** 로 평가 슬라이스 확대. 현재 500 의 EM 분산이 큼.
- **`n_docs` 스윕** (5 → 10 → 20). 논문 Figure 3 의 EM-vs-K 곡선 재현.
- **DPR-only baseline** (`facebook/dpr-reader-multiset-base` extractive reader) 를 같은 retriever 위에서 돌려 Table 1 의 DPR 41.5 와 비교 — RAG 의 generative 컴포넌트가 가져온 이득을 분리.
- **자체 도메인 코퍼스** 위에서 같은 파이프라인 — RAG 의 실제 효용은 여기서 드러남.

---

## 8. 실행 방법 (요약)

Colab A100 80GB / Pro+ 권장.

1. 노트북 열고 **셀 3 (pip)** 실행 → `Runtime → Restart session` (numpy ABI 정상화)
2. **셀 4 부터 끝까지** 위에서 아래로 실행. 셀 4 가 자동으로 디스크/캐시 라우팅, 셀 23–24 가 wiki_dpr + 사전학습 RAG 로드 (첫 실행 30분~1시간 다운로드)
3. 셀 28–29 가 NQ-open dev 500문항 평가, 셀 32 가 결과 테이블 + 막대그래프 시각화

세션이 끊기면 local-scratch 휘발돼서 wiki_dpr 다시 받아야 함 — 가능하면 한 세션에 끝까지.
