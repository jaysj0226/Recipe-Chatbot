# 🍳 Recipe Chatbot (RAG 개선) & 그룹 A/B/C 실험

요리 도메인의 RAG(🔎 Retrieval-Augmented Generation) 실험·개발 레포입니다.  
기존 **CRAG 워크플로우**를 **도메인 OOD 게이트 → 유사도 필터 → Clarify → 웹검색 Fallback**으로 개선했고, **A/B/C 그룹 비교 실험(Baseline/Rewrite/Refuse)** 러너와 **Group A 베이스라인 서버**를 제공합니다.

<p align="left">
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue.svg"></a>
  <a href="#"><img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-ready-success.svg"></a>
  <a href="#"><img alt="LangChain" src="https://img.shields.io/badge/LangChain-0.2%2B-0A7BBB.svg"></a>
  <a href="#"><img alt="ChromaDB" src="https://img.shields.io/badge/Chroma-DB-orange.svg"></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey.svg"></a>
</p>

## 🗂️ 목차

- [핵심 기능](#-핵심-기능)
- [레포 구조](#-레포-구조)
- [요구사항 & 설치](#-요구사항--설치)
- [환경변수](#-환경변수)
- [빠른 시작](#-빠른-시작)
- [A/B/C 실험 실행](#-abc-실험-실행)
- [평가 지표](#-평가-지표)
- [개선된 CRAG 워크플로우](#-개선된-crag-워크플로우)
- [실험 결과(요약)](#-실험-결과요약)
- [재현 팁](#-재현-팁)
- [트러블슈팅](#-트러블슈팅)
- [로드맵](#-로드맵)
- [기여](#-기여)
- [라이선스](#-라이선스)
- [참고 자료](#-참고-자료)

---

## 📌 핵심 기능

- **도메인 OOD 게이트(요리/레시피/영양)**: 비요리 질의는 초기에 차단(`in/out` 분기).
- **유사도 필터링(`FILTER_SCORE`)**: `notSure`일 때 Top-K 중 임계값 미달 청크 제거.
- **Clarify 노드**: 컨텍스트가 비면 사용자 재질문(**ask**) 또는 자동 재작성(**auto**).
- **웹검색 Fallback**: `notGrounded`면 최후에 웹 검색으로 보강.
- **Group A 베이스라인**: 벡터DB Top-K → LLM 답변(문서 없으면 **절대 생성 금지**).
- **A/B/C 실험 러너**: 동일 질의셋으로 세 그룹 비교(+ LLM Judge).

---

## 🧱 레포 구조

```text
.
├─ rag_flow.py                          # 개선된 CRAG 라우팅(메인 플로우)
├─ rag_hybrid_app/
│  ├─ __init__.py
│  ├─ nodes/
│  │  ├─ __init__.py
│  │  ├─ ood_guard_node.py             # LLM+휴리스틱 도메인 분류(요리/레시피/영양)
│  │  ├─ filter_node.py                # 유사도 임계값 기반 청크 필터
│  │  ├─ clarify_node.py               # Clarify(ask/auto)
│  │  ├─ retrieve_node.py              # Top-K + relevance scores
│  │  ├─ llm_answer_node.py
│  │  ├─ rewrite_node.py
│  │  ├─ web_search_node.py
│  │  └─ relevance_check_node.py
│  └─ groupA_rag_flow/
│     ├─ app_groupA.py                 # FastAPI 서버(베이스라인)
│     ├─ groupA_rag.py                 # LangGraph/CLI
│     ├─ list_chroma.py
│     └─ smoke_retrieval.py
├─ experiments_rag_groups/
│  ├─ run_group_ABC_chroma.py          # A/B/C 실험 러너
│  ├─ queries_ko.json                  # 20~30개 질의(명확/애매/OOT 혼합)
│  └─ results/
│     ├─ abc_results.csv               # 질문×그룹 단위 결과
│     └─ abc_summary.csv               # 그룹 평균 요약
├─ data/
│  └─ 10000recipe_dataset.csv          # (선택) 컨텍스트 보강용
├─ .env.example
├─ requirements.txt
├─ LICENSE
└─ README.md
```

> ⚠️ 실제 경로/파일명이 다르면 명령어의 경로만 알맞게 바꾸세요.

---

## ⚙️ 요구사항 & 설치

- Python 3.10+

```bash
# 권장: 가상환경
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 패키지
pip install -U -r requirements.txt
# 또는
pip install -U pandas numpy fastapi uvicorn python-dotenv langchain-openai langchain-chroma chromadb tiktoken
```

---

## 🔑 환경변수

`.env` 또는 시스템 환경변수로 설정합니다.

| 키 | 예시/기본 | 설명 |
|---|---|---|
| `OPENAI_API_KEY` | `sk-...` | OpenAI API 키 |
| `RAG_PERSIST_DIR` | `.../chroma_rag_hybrid_db` | ChromaDB persist 경로 |
| `RAG_COLLECTION_NAME` | `recipe_hybrid_rag` | 컬렉션명 |
| `RAG_TOP_K` | `4` | 검색 Top-K |
| `FILTER_SCORE` | `0.20` | 유사도 필터 임계값(0~1, 권장 탐색: 0.15~0.30) |
| `OOD_MODEL` | `gpt-4o-mini` | OOD 분류용 모델 |
| `OOD_TEMPERATURE` | `0` | OOD 분류 온도 |
| `CLARIFY_MODE` | `ask` | `ask` \| `auto` |
| `GROUPA_PERSIST` | `.../chroma_rag_hybrid_db` | Group A용 persist 경로 |
| `GROUPA_COLLECTION` | `recipe_hybrid_rag` | Group A 컬렉션명 |
| `GROUPA_EMBED_MODEL` | `text-embedding-3-small` | 인덱싱/쿼리 임베딩 모델 일치 필요 |
| `GROUPA_SCORE_THRESHOLD` | `0.0` | Group A 스코어 컷 |
| `GROUPA_CSV` | `.../data/10000recipe_dataset.csv` | (선택) 재료/조리 보강용 CSV |

`.env.example` 예시:

```env
OPENAI_API_KEY=sk-...
RAG_PERSIST_DIR=./chroma_rag_hybrid_db
RAG_COLLECTION_NAME=recipe_hybrid_rag
RAG_TOP_K=4
FILTER_SCORE=0.20
OOD_MODEL=gpt-4o-mini
OOD_TEMPERATURE=0
CLARIFY_MODE=ask

GROUPA_PERSIST=./chroma_rag_hybrid_db
GROUPA_COLLECTION=recipe_hybrid_rag
GROUPA_EMBED_MODEL=text-embedding-3-small
GROUPA_SCORE_THRESHOLD=0.0
GROUPA_CSV=./data/10000recipe_dataset.csv
```

---

## 🚀 빠른 시작

### 1) Group A 베이스라인 서버 실행

```bash
cd rag_hybrid_app/groupA_rag_flow
uvicorn app_groupA:app --host 0.0.0.0 --port 8001 --reload
```

테스트:

```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"계란으로 만들 수 있는 메뉴와 레시피를 알려줘","k":4}'
```

- 검색 문서가 없으면 **절대 생성하지 않고** “제공된 문서가 없어…” 메시지를 반환합니다.  
- `GROUPA_CSV`를 지정하면 `url/recipe_id/title` 단서로 재료/조리를 보강해 품질 향상.

### 2) 개선된 CRAG 플로우 실행(예)

```bash
python rag_flow.py
# 내부에서 OOD → Retrieve → Check → Filter → Clarify → WebSearch 순으로 라우팅
```

---

## 🧪 A/B/C 실험 실행

```bash
cd experiments_rag_groups
python run_group_ABC_chroma.py \
  --queries queries_ko.json \
  --threshold 0.20 \
  --top_k 4
```

생성물:

- `results/abc_results.csv`
- `results/abc_summary.csv`

그룹 정의

- A: 저관련/키워드 미포함이어도 그냥 답변(Baseline)
- B: 저관련 or 키워드 미포함 → 질문 재작성 → 재검색 → 답변  
  (재작성 후 `low/top1` 재계산하여 전후 개선 지표 기록)
- C: 저관련 or 키워드 미포함 → **거절**(“답변할 수 없음”)

---

## 📊 평가 지표

- Faithfulness(1~5): 문서 근거 충실 — 높을수록 좋음  
- Relevance(1~5): 질문과의 직접 관련 — 높을수록 좋음  
- Hallucination(1~5): 문서에 없는 내용 생성 — 낮을수록 좋음  
- 보조 지표: Hall≥3 비율, 응답률/거절률, `low_rate`, Top-1 평균/분포, 지연/토큰

---

## 🧠 개선된 CRAG 워크플로우

```
[User Query]
     │
     ▼
[OOD Guard: in/out]
  in │          out
     ▼           └─▶ [Clarify ask: 도메인 밖 → 재질문/종료]
[Retrieve (Chroma Top-K + scores)]
     ▼
[LLM Answer #1] → [Relevance Check #1]
     │ grounded
     ├──────────────▶ [종료]
     │ notSure
     ▼
[Similarity Filter (FILTER_SCORE)]
  남음 │          없음
     ▼              └─▶ [Clarify (ask|auto)]
[Answer #2] → [Relevance Check #2]
     │ grounded
     ├──────────────▶ [종료]
     │ notGrounded
     ▼
[Web Search Fallback] → [Answer #2] → [Check #2] → [종료]
```

---

## 📈 실험 결과(요약)

| Group | avg_faith | avg_rel | avg_hall | low_rate |
|------:|:---------:|:-------:|:--------:|:--------:|
| A     | 2.8077 | 3.0385 | 1.1538 | 0.5769 |
| B     | 2.7308 | 2.9231 | 1.0769 | 0.5769 |
| C     | 2.8462 | 3.3846 | 1.0000 | 0.5769 |

해석

- **C(거절)**이 관련성·안전성에서 가장 우수(저관련 시 응답 회피).  
- A/B는 비슷 — 초기 세팅에서는 재작성의 이득이 제한적.  
- `low_rate` 동일은 초기 로깅 문제 → **재작성 후 재계산**으로 해결(코드 반영).

---

## 🧪 재현 팁

- `threshold`(저관련 컷): **0.15~0.30** 구간에서 데이터 분포에 맞춰 조정.
- `FILTER_SCORE`: 과도하면 근거 소실로 Clarify 빈발, 낮으면 잡음↑ → **0.20부터 탐색**.
- 키워드 게이트 예:  
  `메뉴, 레시피, 레시피명, 조리, 조리순서, 방법, 요리, 만드는법, 요리법, recipe, cook`.

---

## 🛠️ 트러블슈팅

- **`retrieved_count=0`**  
  → `PERSIST_DIR/컬렉션명` 확인, **임베딩 모델 일치** 확인.

- **문서는 있는데 답변 불가**  
  → 컨텍스트가 비었을 수 있음 → `GROUPA_CSV`(예: `data/10000recipe_dataset.csv`) 지정.

- **웹검색 실패**  
  → API 키/도메인 제한 확인(예: `SERPER_API_KEY` 등 사용 시).

- **OOD 분류 오탐/누락**  
  → `KW_IN/KW_OUT` 휴리스틱 보강 또는 OOD 프롬프트에 few-shot 예시 추가.

---

## 🗺️ 로드맵

- 질의 **카테고리별 분석**(명확/애매/OOT) 리포트 자동화  
- **Top-1 분포/표준편차/Hall≥3** 시계열 대시보드  
- 인덱싱 개선: `page_content`에 **제목/재료/조리** 저장(현재 CSV 보강 의존도↓)

---

## 🤝 기여

PR/이슈 환영합니다. 재현 가능한 예시(환경변수, 질의, 로그)를 함께 남겨주세요.

---

## 🧾 라이선스

이 레포의 코드와 문서는 `LICENSE`를 따릅니다.

---

## 📚 참고 자료

- LangChain: https://python.langchain.com  
- LangGraph: https://langchain-ai.github.io/langgraph/  
- ChromaDB: https://docs.trychroma.com  
- FastAPI: https://fastapi.tiangolo.com
