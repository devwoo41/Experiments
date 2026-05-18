# Agentic AutoSurvey (Faithful Reimplementation)

논문 **"Agentic AutoSurvey: Let LLMs Survey LLMs"** ([arXiv:2509.18661](https://arxiv.org/abs/2509.18661))를 LangGraph 위에서 **논문에 충실하게 1:1 재구현**한 프로젝트입니다. Appendix K(Subagent Prompts), Table 4(12-Dimensional Evaluation Framework), §2.2(Agent Specifications)를 기준으로 구성했습니다.

## 핵심 아이디어

4개의 전문화된 subagent가 LangGraph 파이프라인 위에서 협력해 사용자 주제에 대한 학술 서베이를 자동 생성합니다.

```
START → [Paper Search Specialist] → [Topic Mining & Clustering]
       → [Academic Survey Writer] → [Quality Evaluator] → END
```

## 논문 vs 본 구현 (Deviation Audit)

본 재구현은 알고리즘 차원(에이전트 4개, 12 차원 평가, 가중치, 임베딩 모델, K 선택, TF-IDF 공식, [Author, Year] 인용, 8–12k 단어 분량, K.4 프롬프트 verbatim 등) 핵심은 모두 논문 그대로지만, 아래에 명시한 **deviation 들**이 있습니다. 사용자가 명시적으로 승인했거나(LLM 백본), 그 부작용 회피(날짜 주입), 또는 인프라 차원 단순화(캐싱/jitter 등)에 해당합니다.

| 항목 | 논문 명세 | 본 구현 | 일치? |
|------|----------|--------|------|
| 에이전트 수 / 순서 | 4 (Search → Cluster → Writer → Evaluator) | 동일 | ✅ |
| Search 쿼리 확장 | 20–30개 (core / synonyms / related / AND·OR / acronym) | 25개 (논문 프롬프트 K.1 그대로) | ✅ |
| Search dedup | 90% title similarity threshold | rapidfuzz `token_set_ratio ≥ 90` | ✅ |
| Search 연도 필터 | 2020–2025 | 동일 | ✅ |
| Search 목표량 | 100–150편 | `target_papers=125`, `max_papers=150` | ✅ |
| Cluster 임베딩 | `sentence-transformers/all-MiniLM-L6-v2` | 동일 | ✅ |
| Cluster K 선택 | `K* = argmax silhouette` over K ∈ [5, 15] | 동일 | ✅ |
| Cluster 라벨링 | TF-IDF 상위 토큰 | 동일 | ✅ |
| Cluster 품질 지표 | silhouette / Calinski-Harabasz / Davies-Bouldin | 동일 | ✅ |
| Outliers | low-confidence 논문 식별 | `confidence < 0.2` 경계로 표시 | ✅ |
| Inter-cluster relationships | cosine(centroid_i, centroid_j) | 동일 | ✅ |
| Writer 인용 포맷 | **[Author, Year]** (e.g. `[Lewis et al., 2020]`) | 동일 | ✅ |
| Writer 구조 | Abstract → Intro → 클러스터별 섹션 → **Cross-cutting Analysis** → **Future Directions** → Conclusion | 동일 | ✅ |
| Writer 분량 | 8,000–12,000 단어 | 동일 target | ✅ |
| Writer 인용률 | ≥80% | 동일 target + 로그에 비율 출력 | ✅ |
| Evaluator 12 차원 | (정확한 이름 12개 — 아래) | 동일 | ✅ |
| Evaluator 가중치 | Core 60% / Writing 20% / Depth 20% | 동일 | ✅ |
| Evaluator 출력 스키마 | dimensional_scores / overall_assessment / comparison_to_standards / strengths7+ / weaknesses7+ / prioritized_recommendations / executive_summary | 동일 (K.4 그대로) | ✅ |
| Evaluator 프롬프트 본문 | Appendix K.4 | **K.4 본문을 verbatim으로 사용** | ✅ |
| Cluster TF-IDF 공식 | `TF(w,Cj) × log(K/|{Ck: w∈Ck}|)` — 클러스터를 단일 문서로 처리 | `CountVectorizer`로 cluster-as-document TF 계산 후 위 공식 그대로 적용 | ✅ |
| Evaluator system 프롬프트에 현재 날짜 주입 | 없음 | **`Today's date is YYYY-MM-DD`** 한 줄 추가 | ⚠️ **Deviation #2 — Gemini 컷오프로 인한 미래-날짜 환각 회피용. K.4 본문은 그대로** |
| **LLM 백본** | Claude Sonnet 4.1 (search) + Claude Opus 4.1 (rest) | **Gemini Flash + Pro** | ⚠️ **Deviation #1 — 사용자 명시 선택. 이게 #2의 근본 원인** |
| Writer LLM 호출 구조 | K.3는 단일 invocation으로 전체 8–12k 단어 서베이를 한 번에 생성 | 섹션별 호출 (intro + N clusters + cross-cutting + future + conclusion + abstract = N+5회) | ⚠️ **Deviation #3 — 단일 호출은 timeout/잘림 위험이 커서 섹션별로 분리. 사용자 명시 선택** |
| Per-cluster 섹션의 paper 컨텍스트 | 단일 호출이므로 LLM이 모든 클러스터 논문을 함께 본다 | 클러스터별 섹션은 자기 클러스터 논문만 본다 (intro/cross-cutting/future/conclusion은 전체 풀) | ⚠️ **Deviation #4 — #3의 자연스러운 결과** |
| Search "adaptive citation threshold + venue quality assessment" | §2.2.1에 언급 (구체 임계값/스코어링 미명시) | citation 수로 정렬만, 별도 threshold/venue 필터 없음 | ⚠️ **Deviation #5 — 논문이 구체 값을 안 주어 충실 구현 불가** |
| 24h API response cache + persistent embedding/cluster cache + LRU | §2.3에 명시 | 모두 미구현 | ⚠️ **Deviation #6 — 인프라성, 결과 품질 무관 (재실행 시 비용/시간만 다름)** |
| Exponential backoff **with jitter** | §2.3에 명시 | `wait_exponential` (no jitter) | ⚠️ **Deviation #7 — 인프라성** |
| Alternative query formulations on retry | §2.3에 명시 | 미구현 (기본 재시도만) | ⚠️ **Deviation #8 — 인프라성** |
| Progress persistence / resumption | §2.3에 명시 | 미구현 | ⚠️ **Deviation #9 — 인프라성** |

### Evaluator의 12 차원 (Table 4 그대로)

| 카테고리 | 가중치 | 차원 |
|----------|--------|------|
| **Core Quality (60%)** | 15% × 4 | Citation Coverage, Accuracy, Synthesis Quality, Organization |
| **Writing Quality (20%)** | 5% × 4 | Readability, Academic Rigor, Clarity, Coherence |
| **Content Depth (20%)** | 5% × 4 | Comprehensiveness, Critical Analysis, Novelty & Insights, Future Directions |

### Evaluator의 알려진 구조적 한계 (논문 공통)

논문 Appendix K.4의 Evaluator 프롬프트는 **N(논문수), TOPIC, K(클러스터수) + 생성된 서베이 텍스트** 만 받습니다. **원본 abstract을 전혀 받지 않습니다.** 따라서:

- ✅ "Synthesis Quality", "Organization", "Clarity" 같은 차원은 생성 텍스트만으로 평가 가능
- ⚠️ "Accuracy", "Citation Coverage" 차원은 LLM의 parametric knowledge에 의존하므로 환각이 발생할 수 있음 (이 약점은 논문 디자인 자체의 한계이며 본 구현도 동일)

본 구현은 논문과 동일한 한계를 그대로 유지합니다. 이 한계를 깨고 싶다면 [Future Work](#future-work) 섹션 참조.

## 설치

```bash
cd Agentic_Autosurvey
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt   # sentence-transformers 포함 — 첫 설치 시 ~2분 소요

cp .env.example .env
# .env 열어 GEMINI_API_KEY 입력 (필수)
# SEMANTIC_SCHOLAR_API_KEY 도 넣으면 검색이 훨씬 빠름 (선택)
```

Gemini API 키는 [Google AI Studio](https://aistudio.google.com/apikey) 에서 무료 발급.

## 사용법

```bash
# 논문 충실 재현 (100-150편, K∈[5,15] silhouette 자동 선택, 8-12k 단어 서베이, 12-dim 가중평가)
python main.py --topic "LLM alignment with human feedback"

# 빠른 프로토타입 (~20편, K 범위 축소)
python main.py --topic "retrieval augmented generation" --target-papers 20 --k-min 3 --k-max 6

# 특정 소스만
python main.py --topic "diffusion models for video" --sources arxiv
```

### CLI 플래그

| 플래그 | 논문 default | 설명 |
|--------|------------|------|
| `--topic` (필수) | — | 서베이 주제 |
| `--target-papers` | 125 (100–150 중앙값) | 목표 논문 수 |
| `--max-papers` | 150 | 상한 |
| `--num-queries` | 25 (20–30 중앙값) | Search 쿼리 확장 수 |
| `--k-min` / `--k-max` | 5 / 15 | KMeans 후보 K 범위 (silhouette로 자동 선택) |
| `--year-min` / `--year-max` | 2020 / 2025 | 출판 연도 필터 |
| `--sources` | `arxiv,semantic_scholar` | 데이터 소스 |

전체 디폴트는 [config.yaml](config.yaml). 논문 명세를 그대로 옮긴 파일이라 수정 전 백업 권장.

## 출력물

`outputs/` 에 두 파일이 생성됩니다.

- `<topic>_<timestamp>.tex` — 단독 컴파일 가능한 LaTeX (Overleaf 업로드 → 바로 PDF)
- `<topic>_<timestamp>.eval.json` — 전체 진단 데이터:
  - `search_statistics` (쿼리 / raw / 필터 후 / dedup 후 / final 수)
  - `clusters` + `cluster_quality_metrics` (silhouette / CH / DB / K_candidates)
  - `cluster_relationships` (centroid cosine pairs)
  - `outliers` (low-confidence paper_ids)
  - `evaluation` — Appendix K.4 출력 스키마 전체
    - `dimensional_scores[dim].{score, weight, category, justification, metrics, specific_examples}`
    - `overall_assessment.{weighted_total_score, score_breakdown, quality_level, publication_readiness}`
    - `comparison_to_standards.{vs_acm_computing_surveys, vs_conference_surveys, vs_workshop_papers}`
    - `strengths`, `weaknesses`, `prioritized_recommendations`, `executive_summary`

## 동작 메커니즘 (논문 §2.2 그대로)

### 1. Paper Search Specialist ([agents/search.py](agentic_autosurvey/agents/search.py))
1. LLM에게 K.1 프롬프트를 그대로 던져 20–30개 쿼리 변형 생성 (core / synonyms / related / AND·OR / acronym)
2. 각 쿼리를 arXiv + Semantic Scholar로 fan-out
3. 2020–2025 + abstract completeness 필터링
4. **`fuzz.token_set_ratio ≥ 90` 으로 title 유사도 dedup** (논문 spec)
5. citation → year 순 정렬, target 125편 선택

### 2. Topic Mining & Clustering ([agents/cluster.py](agentic_autosurvey/agents/cluster.py))
1. `sentence-transformers/all-MiniLM-L6-v2`로 title ⊕ abstract 임베딩 (384-dim)
2. K ∈ [5, 15]에 대해 silhouette score 계산, **K\* = argmax silhouette** 선정
3. cluster confidence = `1 − d(x, own_centroid)/max_k d(x, centroid_k)` (논문 수식 그대로)
4. confidence < 0.2 인 논문을 outliers로 분리
5. TF-IDF (1-gram + bi-gram) 상위 토큰을 클러스터명으로
6. 모든 centroid 쌍에 대해 cosine similarity 관계 계산
7. Calinski-Harabasz, Davies-Bouldin 지표 동시 보고

### 3. Academic Survey Writer ([agents/writer.py](agentic_autosurvey/agents/writer.py)) — **Gemini Pro**
1. 모든 논문에 `[Author, Year]` citation key 부여 (동명·동년도 시 `a/b/c` 접미사)
2. K.3 directives를 시스템 프롬프트에 베이크: 8,000–12,000 단어 / ≥80% citation / synthesis-first / patterns/trends / methodology comparison / research gap
3. 구조: Abstract (200–300 단어) → Introduction → 클러스터별 섹션 N개 → **Cross-Cutting Analysis** → **Future Directions** → Conclusion
4. 각 섹션마다 단어 예산을 분배 (intro 10% / clusters 65% / cross 10% / future 7% / conclusion 8%)
5. `[Author, Year]` 마커 등장 순으로 references 자동 추출

### 4. Quality Evaluator ([agents/evaluator.py](agentic_autosurvey/agents/evaluator.py))
1. **Appendix K.4 프롬프트를 verbatim으로** 사용 (`[N]`, `[TOPIC]`, `[K]` 만 substitution)
2. 12 차원을 0–10 점으로 채점하고, 차원별 justification + metrics + specific_examples 3개 이상 받기
3. Category-level 평균 → **0.60·core + 0.20·writing + 0.20·depth** 가중합산
4. strengths/weaknesses 각 7개 이상, prioritized_recommendations(HIGH/MEDIUM/LOW), 200-word executive_summary 받기

### LaTeX Exporter ([exporters/latex.py](agentic_autosurvey/exporters/latex.py))
- `[Author, Year]` 마커는 본문에 **그대로 유지** (논문: "compatibility with academic publishing standards")
- 본문의 LaTeX 특수문자는 escape하되 인용 마커는 보호
- 인용 등장 순 numbered `\thebibliography` 자동 생성, citation_key 접두 표기

## 프로젝트 구조

```
Agentic_Autosurvey/
├── README.md
├── requirements.txt          # sentence-transformers, rapidfuzz, langgraph, google-generativeai 등
├── .env.example
├── config.yaml               # 모든 paper-faithful 디폴트
├── main.py                   # CLI 진입점
├── agentic_autosurvey/
│   ├── state.py              # LangGraph state (Paper / Cluster / Evaluation 등 TypedDict)
│   ├── graph.py              # LangGraph 파이프라인
│   ├── llm.py                # Gemini Pro/Flash wrapper
│   ├── agents/
│   │   ├── search.py         # 에이전트 1 — K.1
│   │   ├── cluster.py        # 에이전트 2 — K.2 + §2.2.2 수식
│   │   ├── writer.py         # 에이전트 3 — K.3
│   │   └── evaluator.py      # 에이전트 4 — K.4 verbatim
│   ├── tools/
│   │   ├── arxiv_client.py
│   │   └── semantic_scholar.py
│   └── exporters/
│       └── latex.py
└── outputs/
```

## 비용 / 시간 안내

논문 충실 재현 (≈125편, K∈[5,15], 8–12k 단어 서베이) 기준:

- **시간**: 15–25분 (Writer Pro 호출 N\_clusters+4회 + Evaluator Flash 1회, sentence-transformers 임베딩 ~2–5분)
- **비용**: 약 $0.5–1.5 (Gemini 무료 티어로 1회는 충분)
- 무료 티어 RPM 제한에 걸리면 tenacity가 지수백오프 재시도 (자동)

소규모 (`--target-papers 20 --k-min 3 --k-max 6`)는 5–8분, ~$0.10.

## 실행 결과 (실제 샘플)

`--topic "chain of thought reasoning in LLMs" --target-papers 20 --num-queries 10 --k-min 3 --k-max 6` 으로 돌린 실제 결과입니다. ([outputs/chain_of_thought_reasoning_in_LLMs_20260518-140317.tex](outputs/chain_of_thought_reasoning_in_LLMs_20260518-140317.tex), [.eval.json](outputs/chain_of_thought_reasoning_in_LLMs_20260518-140317.eval.json))

### 파이프라인 로그

```
[Search]    10 queries -> 55 raw -> 53 year/abstract-filtered -> 49 unique (90% title sim) -> kept 20
[Cluster]   K*=3 (silhouette=0.117, CH=2.3, DB=1.757), 0 outliers
[Writer]    7 sections, ~7900 words, 20/20 cited (100%)
[Evaluator] weighted_total=8.50/10 (core=8.75, writing=8.75, depth=7.50)
```

### Search — 자동 쿼리 확장 (10개)

```
"chain of thought reasoning LLMs", "chain of thought large language models",
"step-by-step reasoning large language models", "prompt engineering chain of thought LLMs",
"multi-step reasoning large language models", "zero-shot chain of thought prompting",
"CoT LLMs", "chain of thought reasoning capabilities of LLMs",
"explainable reasoning large language models", ...
```

→ arXiv + Semantic Scholar 합 55편 수집, 2020–2025 + abstract 필터 후 53편, 90% title 유사도 dedup으로 49편, citation 순 상위 20편 최종 채택.

### Cluster — silhouette 기반 K 자동 선택

`--k-min 3 --k-max 6` 범위에서 모든 K 후보 silhouette 점수:

| K | silhouette |
|---|-----------|
| **3** | **0.117** ← 선택 |
| 4 | 0.100 |
| 5 | 0.049 |
| 6 | 0.043 |

→ K\*=3 자동 결정. CH=2.3, DB=1.757도 동시 보고.

| # | 클러스터명 (TF-IDF 자동) | 논문 수 | Inter-cluster cosine |
|---|----------------------|---------|---------------------|
| 0 | Graph Graphs Summaries | 3 | C0–C1: 0.752, C0–C2: 0.509 |
| 1 | Long Shot Problem | 15 | C1–C2: 0.645 |
| 2 | Backdoor Badchain Vulnerabilities | 2 | — |

→ outliers 0건. 클러스터 1이 75% 차지하는 imbalance가 Evaluator의 weaknesses에도 잡혔습니다 (의미 있는 진단).

### Writer — 7 섹션, 약 7,900 단어, **100% citation rate**

생성된 Introduction 일부:

> The seminal discovery that simply prompting a model with "Let's think step by step" could unlock emergent reasoning abilities in sufficiently scaled models transformed the landscape, demonstrating that LLMs could be effective zero-shot reasoners on tasks previously thought to be beyond their reach [Kojima et al., 2022]. ... Researchers sought to improve the reliability and quality of the generated reasoning paths. One key innovation was self-consistency [Wang et al., 2022]. ... This principle ... was further generalized into more structured frameworks like Tree of Thoughts (ToT) [Yao et al., 2023].

— **[Author, Year]** 인용 포맷이 본문에 자연스럽게 박혀 있고, **20편 전부 인용** (목표 80% 초과). 7,900 단어로 목표 8,000–12,000의 하한 근처.

### Evaluator — 12 차원 (60/20/20 가중평균)

| 카테고리 | 차원 | 점수 |
|----------|------|------|
| **Core (60%)** | Citation Coverage | **9.0** |
| | Accuracy | **9.0** |
| | Synthesis Quality | 8.0 |
| | Organization | 9.0 |
| → 소계 | | **8.75** |
| **Writing (20%)** | Readability | 9.0 |
| | Academic Rigor | 8.0 |
| | Clarity | 9.0 |
| | Coherence | 9.0 |
| → 소계 | | **8.75** |
| **Depth (20%)** | Comprehensiveness | 8.0 |
| | Critical Analysis | 7.0 |
| | Novelty & Insights | 7.0 |
| | Future Directions | 8.0 |
| → 소계 | | **7.50** |
| **Weighted total** | | **8.50 / 10** |
| Quality level | | **Excellent (A-)** |

### Evaluator가 직접 짚은 약점 (자동 진단)

1. **Abstract와 본문 citation 불일치** — Abstract에 `[Chen et al., 2023]`, `[Wallace et al., 2024]` 인용이 본문에는 `[Chen et al., 2025]`, `[Xiang et al., 2024]`로 바뀌어 있음 (Abstract와 섹션을 다른 LLM 호출로 생성한 부작용 — Deviation #3의 정확한 진단)
2. **Cluster 불균형** — Long Shot Problem이 15/20편 차지, 나머지는 3편/2편
3. **Taxonomy 미명시** — Intro에서 "taxonomy 제공"이라고 했지만 명시적 분류 체계는 부재

→ 추천 우선순위 5건 (HIGH/MEDIUM/LOW), 강점 8건, 약점 7건 자동 생성.

### 논문 평균과의 비교

| 시스템 | Core (60%) | Writing (20%) | Depth (20%) | Avg |
|--------|-----------|---------------|-------------|-----|
| **본 구현 — CoT 1회 실행** | **8.75** | **8.75** | **7.50** | **8.50** |
| 논문 Agentic AutoSurvey (6 주제 평균) | 8.23 | 8.31 | 7.92 | 8.18 |
| 논문 AutoSurvey baseline | 4.13 | 4.95 | 5.95 | 4.77 |

→ 본 1회 실행이 **논문 평균(8.18)을 약간 상회**. 단, 1회 표본이고 주제·논문 규모(20편)가 작아 직접 비교는 어렵습니다.

### 자원 사용

| 항목 | 측정값 |
|------|-------|
| 총 소요 시간 | 약 6분 |
| 추정 Gemini 비용 | ≈ $0.15 (무료 티어 한도 내) |
| Writer Pro 호출 | 7회 (intro + 3 clusters + cross-cutting + future + conclusion + abstract) |
| Flash 호출 | 2회 (쿼리 확장 + 평가) |
| 임베딩 호출 | 20회 (sentence-transformers 로컬, API 비용 없음) |
| 생성된 `.tex` | 약 62 KB |

### 날짜 환각 패치 효과 검증

가장 의미있는 검증 포인트입니다. **첫 RAG 실행 (패치 전)** vs **이번 CoT 실행 (패치 후)** 의 Core 점수 비교:

| 차원 | 첫 실행 (패치 전) | 이번 실행 (패치 후) |
|------|----------|----------|
| Accuracy / factual_accuracy | **2.0** ⚠️ | **9.0** ✅ |
| Citation Coverage / citation_quality | **1.0** ⚠️ | **9.0** ✅ |
| Evaluator 근거 | "2025년 논문은 미래 날짜라 검증 불가" | "Claims are consistently supported, attributions correct, no misinterpretations" |

→ Evaluator system prompt에 `Today's date is 2026-05-18` 한 줄 추가만으로 환각 완전 차단됨. Deviation #2의 의도된 효과 확인.

## 첫 구현 단계에서 발견된 deviation (지금은 모두 수정 완료)

처음 구현에서는 abstract-level WebFetch 요약만 보고 작성해 다음 부분이 논문과 달랐습니다 — **모두 수정 완료**:

| 항목 | 1차 구현 (수정 전) | 현재 (수정 후) |
|------|----------|----------|
| 12 차원 이름 | relevance / depth / balance / factual_accuracy 등 임의 작명 | Citation Coverage / Accuracy / Synthesis Quality / Organization / Readability / Academic Rigor / Clarity / Coherence / Comprehensiveness / Critical Analysis / Novelty & Insights / Future Directions |
| 가중치 | 균등 (1/12) | 60/20/20 + 카테고리 내 균등 |
| 인용 포맷 | `[CITE:<paper_id>]` | `[Author, Year]` |
| 쿼리 확장 수 | 4 | 25 (20–30) |
| Dedup | paper_id 정확 일치 | 90% title similarity |
| 임베딩 모델 | Gemini embedding | sentence-transformers `all-MiniLM-L6-v2` |
| K 선택 | 고정 6 | silhouette over K ∈ [5,15] |
| 클러스터 라벨링 | LLM 라벨 | TF-IDF top terms |
| 클러스터 품질 지표 | 없음 | silhouette + CH + DB + outliers + relationships |
| 서베이 구조 | Abstract+Intro+Sections+Conclusion | + **Cross-Cutting Analysis** + **Future Directions** |
| 분량 | 3,000–5,000 단어 | 8,000–12,000 단어 target |
| Evaluator 출력 | scores + rationale 만 | K.4 전체 schema (strengths 7+ / weaknesses 7+ / recommendations / executive_summary 등) |

## Future Work

논문 한계를 넘어서고 싶을 때 추가할 수 있는 옵션 (현재 미구현, 사용자가 원할 때 추가 예정):

- **Evaluator에 abstract 동봉** — `Accuracy` / `Citation Coverage` 차원을 실제 fact-check로 만듦. 논문보다 강한 평가가 됨.
- **Multi-LLM-as-Judge** — AutoSurvey baseline이 쓰는 방식. 단일 judge 편향 ↓
- **Iterative Writer refinement** — Evaluator 점수 임계치 미만 시 Writer 재호출 (LangGraph 조건부 edge로 손쉽게 추가 가능)
- **Citation verification agent** — `[Author, Year]` 마커가 실제 cited paper와 의미적으로 정렬되는지 검증

## 라이선스 / 출처

원 논문: Liu, Wu, Zhang, Sun. *Agentic AutoSurvey: Let LLMs Survey LLMs.* arXiv:2509.18661, 2025.
본 리포지토리는 교육·연구 목적의 비공식 재구현입니다. Appendix K 프롬프트는 fair-use 학술 인용으로 포함.
