# Source Attribution Evaluation Framework

논문 **"Cited but Not Verified: Parsing and Evaluating Source Attribution in LLM Deep Research Agents"** ([arXiv:2605.06635](https://arxiv.org/abs/2605.06635), Onweller et al. 2026, PwC)을 **LangGraph + Gemini**로 재구현한 프로젝트입니다.

## 핵심 아이디어 — Algorithm 1 (paper §3.1)

3-Phase 파이프라인:

```
START
  └─ Phase 0: Deep Research Agent  → 마크다운 보고서 + inline citations
  └─ Phase 1: Markdown AST Parser  → AttributionDocument(citations, attributions)
  └─ Phase 2: Evaluation Runner
        ├─ fetch URLs (parallel)
        ├─ Link Works     ← HTTP 접근성 (LLM 없음)
        ├─ Relevant Content ← Gemini-as-a-judge (binary + 설명, source 5000자 truncate)
        └─ Fact Check     ← Gemini-as-a-judge (사실/숫자/날짜 검증, binary)
END
```

논문 핵심 발견: 프론티어 모델도 **Link Works 94%+, Relevant Content 80%+**지만 **Fact Check 39–77%**. 검색 깊이↑ → Fact Check↓ (~42%↓).

## 논문 vs 본 구현 (Deviation Audit)

본 재구현은 알고리즘 핵심을 그대로 따르되, 아래와 같은 **deviation**이 있습니다.

| 항목 | 논문 spec | 본 구현 | 상태 |
|------|---------|--------|------|
| 파이프라인 (Phase 0/1/2) | Algorithm 1 그대로 | 동일 | ✅ |
| AST Parser — 지원 인용 포맷 | `[1]`, `[^note]`, `[text](url)`, `<url>`, `[1-3]` range | 5개 모두 지원 | ✅ |
| AST Parser — Canonicalize | line endings 정규화, code block 제거 | 동일 | ✅ |
| AST Parser — Backward attribution | 단락 끝 citation이 앞쪽 미인용 문장에 적용 | 단락 단위 구현, 동일 | ✅ |
| Phase 2 — Link Works | HTTP 접근성 (LLM 없음) | `requests` HEAD→GET + Trafilatura content check | ✅ |
| Phase 2 — Source truncation | 5,000 chars | 동일 | ✅ |
| Phase 2 — Relevant Content | LLM-as-a-judge binary + 설명 | Gemini-as-a-judge binary + 설명 | ✅ |
| Phase 2 — Fact Check | LLM-as-a-judge binary + 설명 (사실/숫자/날짜) | 동일 | ✅ |
| 동시성 | 10 agents / 15 evaluators / 5 retries (5s delay) | `ThreadPoolExecutor(max_workers=15)` + tenacity 3 retries | ✅ (소폭 단순화) |
| **Phase 0 — Deep Research Agent** | OpenAI/Anthropic/Google/오픈소스 14 LLM × web search | **Gemini 2.5 Pro/Flash + 내장 `google_search` grounding** | ⚠️ **Deviation #1 — 사용자 명시 선택 (Gemini만 사용)** |
| **Tool-call ablation** | tool calls ∈ {2, 10, 30, 50, 70, 100, 150} 직접 제어 | **prompt-based depth (brief / moderate / extensive)** — Gemini grounding이 tool-call 직접 제어를 노출하지 않음 | ⚠️ **Deviation #2 — 사용자 명시 선택 (#1의 기술적 귀결)** |
| **쿼리 데이터셋** | 130개 (DeepResearch Bench + BrowseComp) | **8개 커스텀 쿼리** ([queries.yaml](queries.yaml)) | ⚠️ **Deviation #3 — 사용자 명시 선택 (프로토타입용)** |
| **LLM-as-a-judge 프롬프트** | "rubric-based" — verbatim 미공개 | §3.3.2 / §3.3.3 rubric 기반으로 본 구현에서 작성. 본문 §3.3에 명시된 항목(facts / numbers / dates / assertions, topical alignment)을 그대로 평가 기준으로 포함 | ⚠️ **Deviation #4 — 논문이 verbatim prompt를 공개 안 함** |
| **Human calibration** | Fact Check 평가자를 50–100 manual label로 calibration | **미구현** | ⚠️ **Deviation #5 — 미구현 (Future Work)** |
| LLM-as-a-judge 백본 | 미명시 | Gemini 2.5 Pro (temperature 0.0) | ⚠️ **Deviation #6 — 논문이 judge LLM 미명시** |
| Concurrency: 10 agents | 동시 10 agent 요청 | 본 구현은 cell loop 순차 (tqdm) | ⚠️ **Deviation #7 — 단일 워커, 안정성 우선** |

## 설치

```bash
cd Parsing_Evaluating_Source_Attribution_in_LLM_Deep_Research_Agents
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# .env 열어서 GEMINI_API_KEY 입력
```

Gemini API 키: [Google AI Studio](https://aistudio.google.com/apikey)에서 무료 발급.

## 사용법

```bash
# 모든 쿼리 × 모든 모델 × 모든 depth (8 × 2 × 3 = 48 cells)
python main.py

# 스모크 테스트 — 한 쿼리, Pro만, moderate depth
python main.py --queries q1_quantum --models gemini-2.5-pro --depths moderate

# Pro vs Flash만, moderate depth만
python main.py --depths moderate

# 두 쿼리만
python main.py --queries q1_quantum,q3_llm_safety
```

### CLI 플래그

| 플래그 | 기본값 | 설명 |
|--------|-------|------|
| `--queries` | 전부 | 쉼표 구분 쿼리 ID (queries.yaml 참조) |
| `--models` | config 값 | 쉼표 구분 Gemini 모델 (`gemini-2.5-pro,gemini-2.5-flash`) |
| `--depths` | `brief,moderate,extensive` | depth instruction 부분집합 |
| `--queries-file` | `queries.yaml` | 다른 쿼리 파일 사용 |
| `--config` | `config.yaml` | 다른 config 파일 사용 |
| `--output-dir` | `outputs/` | 결과 저장 위치 |

## 출력물

`outputs/run_<timestamp>/` 디렉터리 안에:

- `<query_id>__<model>__<depth>.json` — 각 cell의 full output:
  - `document.raw_markdown` — Phase 0 결과 마크다운
  - `document.citations[]` — 유니크 citation (id, url, raw_labels, url_content)
  - `document.attributions[]` — claim sentences (text, text_nocite, citation_ids)
  - `document.evals[]` — 각 (attribution, citation) 쌍의 3-dim 점수와 설명
- `_summary.json` — 전체 cell 집계:
  - `by_cell`: (model, depth) 별 LW%/RC%/FC%
  - `by_model`: 모델 별 전체 평균
  - `ablation_fact_check_by_depth`: 모델 별 depth → FC% 표

## 프로젝트 구조

```
Parsing_Evaluating_Source_Attribution_in_LLM_Deep_Research_Agents/
├── README.md
├── requirements.txt
├── .env.example
├── config.yaml                            # paper-faithful 디폴트 + deviation 메모
├── queries.yaml                           # 8개 테스트 쿼리
├── main.py                                # CLI 진입점
├── source_attribution_eval/
│   ├── state.py                           # AttributionDocument, Citation, Attribution, PairEval
│   ├── llm.py                             # Gemini wrapper (+ google_search grounding, JSON judge)
│   ├── graph.py                           # LangGraph: researcher -> parser -> fetcher -> evaluator
│   ├── agent/researcher.py                # Phase 0
│   ├── parser/ast_parser.py               # Phase 1 — Algorithm 1 lines 2-7
│   ├── tools/web_fetch.py                 # Trafilatura + requests
│   ├── evaluators/
│   │   ├── link_works.py                  # §3.3.1
│   │   ├── relevant_content.py            # §3.3.2 + rubric prompt
│   │   └── fact_check.py                  # §3.3.3 + rubric prompt
│   └── reports/aggregator.py              # 집계 표
└── outputs/
```

## 동작 메커니즘 (논문 §3 그대로)

### Phase 0 — Deep Research Agent ([agent/researcher.py](source_attribution_eval/agent/researcher.py))
- Gemini 2.5 Pro/Flash에 `tools=[{"google_search": {}}]` 그라운딩 활성화
- system prompt가 (1) inline `[text](url)` 마크다운 인용 형식, (2) depth instruction(brief/moderate/extensive) 강제
- 출력: 3–8 단락 마크다운 보고서 + (옵션) grounding_metadata 의 URL 리스트

### Phase 1 — Markdown AST Parser ([parser/ast_parser.py](source_attribution_eval/parser/ast_parser.py))
1. **Canonicalize** — `\r\n` 정규화, 펜스 `\`\`\`...\`\`\`` + 들여쓰기 코드블록 제거
2. **Build AST** — `markdown-it-py`로 token 스트림 (parse 검증)
3. **Extract citations** (5개 포맷):
   - `[1]`, `[2]` → reference list (`[1]: https://...`)에서 URL 매핑
   - `[^note]` footnote
   - `[text](url)` inline markdown link
   - `<https://...>` autolink
   - `[1-3]` range → `[1][2][3]`로 자동 확장
4. **Sentence segmentation** — regex 또는 NLTK punkt
5. **Backward attribution** — 단락 마지막 citation을 그 단락의 모든 앞선 미인용 문장에 전파 (paper §3.2: "a single reference supports multiple related claims")
6. **Deduplicate** — URL 정규화 (lowercase host, trailing punctuation strip) 후 unique citation registry

### Phase 2 — Evaluators (병렬)
**Fetcher + Link Works** (병렬, 15 workers):
- requests HEAD → GET (HEAD 거부 시 GET fallback)
- HTTP < 400 + Trafilatura 추출 결과 non-empty → score 1
- 추출된 main text를 citation에 캐싱 (judge 단계 재사용)

**Relevant Content** (병렬, LLM judge):
- 입력: `text_nocite` claim, 첫 5,000 chars source content, URL
- Gemini-as-a-judge → `{"score": 0|1, "explanation": "..."}` (JSON forced)
- Link Works가 0인 pair는 skip

**Fact Check** (병렬, LLM judge):
- 같은 입력 + 더 strict rubric (facts / numbers / dates / quotes 확인)
- Gemini-as-a-judge → `{"score": 0|1, "explanation": "..."}`

### Aggregator ([reports/aggregator.py](source_attribution_eval/reports/aggregator.py))
- `by_cell` — (model, depth) 별 LW%/RC%/FC%, n_pairs, n_citations, n_attributions
- `by_model` — 모델 별 평균
- `ablation_fact_check_by_depth` — 논문 §4.3의 핵심 plot에 해당

## 비용 / 시간 안내

쿼리 1개 × 모델 1개 × depth 1개 (1 cell) 기준:
- **Phase 0** Gemini grounding 호출 1회 — 약 30–90초, ~$0.05 (Pro 기준)
- **Phase 1** AST 파싱 — 0.1초 미만 (LLM 호출 없음)
- **Phase 2** N개 unique citation × HTTP fetch + 2N LLM judge 호출 — citation 수에 비례
  - 평균 6–15개 citation/report 가정 시 약 2–5분, ~$0.05–0.15

**스모크 (1 쿼리 × Pro × moderate)**: 약 3–6분, ~$0.10
**기본 매트릭스 (8 × 2 × 3 = 48 cells)**: 약 2–5시간, ~$5–15 (Gemini 무료 티어 RPM에 자주 걸릴 수 있음)

## 실행 결과 (24-cell 매트릭스)

`--queries q1_quantum,q3_llm_safety,q5_fusion,q7_chip` × Pro/Flash × brief/moderate/extensive = **24 cell** 매트릭스 결과입니다. ([outputs/run_20260518-173540/](outputs/run_20260518-173540/))

**총 990 (attribution, citation) pair 평가** — 단일 cell의 N=27보다 약 37배 큰 표본. Judge LLM은 Flash로 전환 (Deviation #6 폭 내, ~10× 비용 절감).

### 셀별 결과 (paper Table-2 형식)

| Model × Depth | pairs | LW% | RC% | **FC%** |
|---------------|-------|-----|-----|---------|
| Pro × brief | 99 | 83.8% | 85.5% | **43.4%** |
| Pro × moderate | 141 | 85.8% | 83.5% | **43.0%** |
| Pro × extensive | 189 | 86.2% | 88.3% | **28.8%** |
| Flash × brief | 180 | 86.7% | 87.8% | **44.2%** |
| Flash × moderate | 168 | 85.7% | 78.5% | **27.8%** |
| Flash × extensive | 213 | 82.6% | 69.9% | **27.3%** |
| **Pro 전체** | **429** | **85.5%** | **86.1%** | **36.8%** |
| **Flash 전체** | **561** | **84.8%** | **78.4%** | **33.0%** |

### 핵심 발견 #1 — 논문 §4.1 ("표면 인용 품질이 사실 실패를 가린다")

| 차원 | 본 매트릭스 (Pro+Flash 평균) | 논문 (frontier models) |
|------|---------------------------|---------------------|
| **Link Works** | **85.1%** | 94%+ |
| **Relevant Content** | **81.7%** | 80%+ ✅ |
| **Fact Check** | **34.7%** | **39–77%** ⚠️ |

→ **LW/RC는 80%대 양호한데 FC는 30%대로 급락** — 정확히 논문이 발견한 disconnect. Pro/Flash 모두에서 동일 패턴.

논문 대비 LW가 다소 낮은 이유: Gemini grounding이 종종 paywalled / JS-heavy 페이지로 redirect되는데 (예: McKinsey, NEJM), Trafilatura가 정적 HTML에서 본문 추출 못 하면 "no content"로 FAIL 처리. 논문은 JS rendering capable extractor를 명시했으므로 (§3.3.1), 우리 Trafilatura가 그보다 보수적인 셈.

### 핵심 발견 #2 — 논문 §4.3 ("검색을 더 한다고 사실 정확도가 올라가지 않는다")

**FC% by depth (ablation):**

```
                brief   moderate  extensive
gemini-2.5-pro   43.4%   43.0%    28.8%    (Δ = -14.6 pp)
gemini-2.5-flash 44.2%   27.8%    27.3%    (Δ = -16.9 pp)
```

→ **두 모델 모두 depth 늘리면 Fact Check 정확도가 떨어짐**. 논문에선 GPT-5.4가 tool calls 2→150 사이에 79%→17% 떨어졌고 ("approximately 42% drop"), 우리도 brief→extensive 사이 **-14.6 / -16.9 pp 감소**. 절대치는 다르지만 **방향과 메커니즘이 정확히 일치**.

해석: 더 많이 검색할수록 → 더 많은 source 인용 → 모델이 종합·추정·일반화 빈도 ↑ → source에 없는 디테일 (연도, 회사 수, 정확한 수치) fabrication ↑.

특히 흥미로운 점: **Pro는 moderate까지는 brief 수준을 유지하다가(43%→43%) extensive에서 급락(43%→29%)**. Flash는 **brief→moderate 단계에서 이미 급락(44%→28%)** 후 plateau. **Pro가 search depth에 더 robust**하다는 신호.

### 핵심 발견 #3 — Pro vs Flash 비교

| 차원 | Pro | Flash | Δ |
|------|-----|-------|---|
| Link Works | 85.5% | 84.8% | +0.7 pp |
| Relevant Content | **86.1%** | 78.4% | **+7.7 pp** |
| Fact Check | **36.8%** | 33.0% | **+3.8 pp** |

→ Pro가 RC에서 분명히 우위, FC에서도 소폭 우위. **더 큰 모델이 동일 grounding tool을 더 신중하게 활용**한다는 신호. 다만 두 모델 모두 FC는 30%대로 *낮은 절대 점수*는 공통.

### 흥미로운 부산물 — Pro의 citation 절약

| Model × Depth | n_pairs |
|---------------|---------|
| Pro × brief | **99** |
| Flash × brief | 180 |
| Pro × extensive | 189 |
| Flash × extensive | 213 |

→ **같은 쿼리·같은 depth에서 Pro가 더 적은 citation을 만듦.** Pro는 같은 출처를 여러 claim에 재사용하는 경향이 강하고 (논문 §4.2의 "citation quantity trades off against quality"와 일치), Flash는 출처를 더 흩뿌리는 경향.

### 자원 사용

| 항목 | 값 |
|------|-----|
| 총 소요 시간 | **1시간 11분** (24 cell, cell당 평균 178초) |
| 추정 Gemini 비용 | **~$1.5–2** (Flash judge로 전환해서 절감) |
| 평가된 총 pair | **990** |
| Phase 0 Pro 호출 | 12회 |
| Phase 0 Flash 호출 | 12회 |
| Phase 2 Flash judge 호출 | ~2,000회 (RC + FC × 통과 pair) |

### 1-셀 스모크 vs 24-셀 매트릭스

| 차원 | 스모크 N=27 | 매트릭스 N=990 | 차이 |
|------|------------|---------------|------|
| LW% | 100% | 85.1% | -14.9pp — 스모크가 운이 좋았음 |
| RC% | 77.8% | 81.7% | +3.9pp — 큰 차이 없음 |
| FC% | 51.9% | 34.7% | -17.2pp — **스모크 표본 작아 noise 컸음** |

→ N=27짜리 단일 스모크는 변동이 컸음. **N=990 매트릭스가 훨씬 신뢰할 만한 추정**.

## 실행 결과 (스모크 샘플 — 참고용)

`--queries q1_quantum --models gemini-2.5-pro --depths brief` 1 cell 스모크 결과입니다. ([outputs/run_20260518-170847/](outputs/run_20260518-170847/))

### 파이프라인 로그

```
[Phase 0]        gemini-2.5-pro (brief): 9512 chars markdown, 9 grounded sources
[Phase 1]        parsed 9 unique citations, 18 attributions
[Fetch+LinkWorks] 9/9 URLs accessible
[Phase 2]        27 pairs | LinkWorks 27/27 (100%), Relevant 21/27 (78%), FactCheck 14/27 (52%)

cells: 100% [03:54<00:00, 234.55s/it]
```

### 결과 vs 논문

| 차원 | 본 실행 (Gemini 2.5 Pro, brief) | 논문 (frontier models) |
|------|-------------------------------|----------------------|
| **Link Works** | **100%** (27/27) | 94%+ ✅ 일치 |
| **Relevant Content** | **77.8%** (21/27) | 80%+ ✅ 거의 일치 |
| **Fact Check** | **51.9%** (14/27) | **39–77%** ✅ 정확히 그 범위 |

→ **논문의 핵심 발견 — "표면적 인용 품질은 우수하지만 Fact Check는 절반 수준" — 단 1 cell 스모크에서도 그대로 재현됨.**

### Pipeline 단계별 출력

**Phase 0 — Grounding으로 inline citation 주입**: Gemini 2.5 Pro가 brief depth (≤2 sources)로 9개 grounding chunk를 사용한 9,512자 markdown 리포트를 생성. 9개 redirect URL (`vertexaisearch.cloud.google.com/grounding-api-redirect/...`)이 본문 곳곳에 `[source](url)` 형태로 주입됨.

**Phase 1 — AST parsing**: 9개 unique URL + 18개 attribution 문장 추출. backward attribution도 정상 작동 (단락 끝 citation이 앞선 미인용 문장에 전파됨). 18 attributions × 평균 1.5 citation = 27 (attribution, citation) pairs 생성.

**Phase 2 — 평가**:
- **Link Works** — 9/9 URL 모두 응답 OK, Trafilatura 본문 추출 성공
- **Relevant Content** — 27 pair 중 21 (77.8%) 통과. 6 pair는 source가 다른 주제 (예: 더 오래된 양자컴퓨팅 일반 기사가 cited).
- **Fact Check** — 27 pair 중 14 (51.9%) 만 통과. 13 pair에서 claim의 fact가 source content와 불일치.

### Evaluator가 잡아낸 실제 실패 사례

**✅ Pass 사례** (Fact Check = 1):
> CLAIM: "These advancements are moving the industry from the era of noisy, intermediate-scale quantum (NISQ) devices toward the long-awaited goal of fault-tolerant quantum computation."
>
> Judge: "The source describes multiple recent breakthroughs that address noise and error correction... explicitly states 'multiple viable paths toward fault-tolerant quantum computing are progressing simultaneously,' directly supporting the claim."

**❌ Fail 사례 1** (Fact Check = 0 — 미래 날짜 fabrication):
> CLAIM: "The years 2024 and **2025** have marked a significant turning point in quantum computing..."
>
> Judge: "While the source extensively details multiple breakthroughs in quantum computing during 2024, it makes **no mention of the year 2025**, rendering that part of the claim unsupported."

**❌ Fail 사례 2** (Fact Check = 0 — 출처 불일치):
> CLAIM: "...several companies reporting substantial progress in quantum error correction..."
>
> Judge: "The claim states progress occurred in 2024 and 2025, but the source dates the milestone it discusses to **February 2023**. Additionally, the claim mentions 'several companies,' whereas the provided source only discusses a breakthrough by Google."

→ 정확히 논문이 §4.1에서 말한 **"surface-level citation quality masks factual failures"** 현상. Gemini가 grounded sources에서 가져온 정보를 일반화/추정하면서 source에 없는 디테일(`2025`, `several companies`)을 추가함.

### 자원 사용

| 항목 | 값 |
|------|-----|
| 총 소요 시간 | **3분 54초** |
| Phase 0 (grounding 생성) | ~90초 |
| Phase 1 (parsing, LLM 없음) | <1초 |
| Phase 2 (fetch + 18×2=36 LLM 호출) | ~140초 |
| Citation pairs 평가됨 | 27 |
| 추정 Gemini 비용 | ≈ $0.20 (Pro grounding + 36 Pro judge call) |

### 초기 timeout 이슈 (해결 완료)

처음 실행에서는 1 cell에 **20분 넘게 stuck** 됐는데, 원인은 ThreadPool 안의 일부 URL fetch가 hang되면서 무한 대기. 다음과 같이 fix:

1. **`http_probe`**: `(connect=5s, read=8s)` hard timeout, retries 제거, body 2MB cap
2. **fetcher ThreadPool**: 각 future에 `timeout+10s` 강제 abandon (한 URL이 전체를 못 막게)
3. **LLM judge ThreadPool**: 각 호출 120s timeout
4. **Gemini grounding**: `http_options.timeout=180s` 명시
5. **tenacity retries**: 3 → 2 회

→ 같은 cell 실행이 hang → **234초로 안정**.

## 알려진 구조적 한계 (논문 공통)

논문 §5 (Limitations) 가 명시한 한계는 본 재구현에도 그대로 적용됩니다:

1. **LLM-as-a-judge bias** — position bias / self-enhancement (Wang 2024, Zheng 2023). 본 구현은 single judge이므로 ensemble로 완화하려면 multi-judge 모드 추가 필요.
2. **Web temporal instability** — 같은 URL이 시점에 따라 다른 콘텐츠로 응답. 재현성 ↓.
3. **JS-only 페이지** — Trafilatura는 정적 HTML 기반. SPA/JS-rendered 콘텐츠 접근에 한계. 필요하면 Playwright 추가.
4. **Private/enterprise corpora** — 본 framework는 공개 웹만 평가 가능. RAG private knowledge base에 그대로 적용 불가.

## Future Work

- **Multi-LLM judge ensemble** — 단일 judge 편향 완화 (Gemini + Claude + GPT)
- **Human calibration loop** — 본 구현이 빠뜨린 50–100 manual label 단계 추가
- **Playwright fallback** — JS-rendered 페이지에서 Trafilatura 실패 시 자동 전환
- **DeepResearch Bench / BrowseComp full reproduction** — 130 쿼리 데이터셋 통합
- **Tool-call exact-control mode** — Gemini grounding 대신 ReAct + DuckDuckGo로 `max_tool_calls=N` 정확 제어 (논문의 ablation 정확 재현)

## 라이선스 / 출처

원 논문: Onweller, Lumer, Huber, Ramchandani, Subbiah, Feld. *Cited but Not Verified: Parsing and Evaluating Source Attribution in LLM Deep Research Agents.* arXiv:2605.06635, May 2026 (PwC Commercial Technology and Innovation Office).
본 리포지토리는 교육·연구 목적의 비공식 재구현입니다.
