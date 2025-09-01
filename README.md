🍳 Recipe Chatbot (RAG 개선) & 그룹 A/B/C 실험

요리 도메인의 RAG(ChatGPT Retrieval-Augmented Generation) 실험·개발 레포입니다.
기존 CRAG 워크플로우를 도메인 OOD 게이트 → 유사도 필터 → Clarify → 웹검색 Fallback으로 개선했고,
A/B/C 그룹 비교 실험(Baseline/Rewrite/Refuse)을 위한 러너와 Group A 베이스라인 서버도 제공합니다.

📌 핵심 기능

도메인 OOD 게이트(요리/레시피/영양): 비요리 질의는 초기에 차단 (in/out 분기).

유사도 필터링(FILTER_SCORE): NotSure일 때 Top-K 중 임계값 미달 청크 제거.

Clarify 노드: 컨텍스트가 비면 사용자 재질문(ask) 또는 자동 재작성(auto).

웹검색 Fallback: NotGrounded면 최후에 웹 검색으로 보강.

Group A 베이스라인: 벡터DB Top-K → LLM 답변 (문서 없으면 답변 불가).

A/B/C 실험 러너: 동일 질의셋으로 세 그룹 비교(+ LLM Judge).

🧱 레포 구조(예시)
.
├─ rag_flow.py                         # CRAG 워크플로우(개선 라우팅 반영)
├─ rag_hybrid_app/
│  └─ nodes/
│     ├─ ood_guard_node.py            # LLM+휴리스틱 도메인 분류 (요리/레시피/영양)
│     ├─ filter_node.py               # 유사도 임계값 기반 청크 필터
│     ├─ clarify_node.py              # Clarify(ask/auto)
│     ├─ retrieve_node.py             # Top-K + relevance scores 반환
│     ├─ llm_answer_node.py
│     ├─ rewrite_node.py
│     ├─ web_search_node.py
│     └─ relevance_check_node.py
│
├─ groupA_rag_flow/
│  ├─ app_groupA.py                   # FastAPI 서버
│  ├─ groupA_rag.py                   # LangGraph CLI
│  ├─ list_chroma.py / smoke_retrieval.py
│  └─ README.md (선택)
│
├─ experiments_rag_groups_chroma/
│  ├─ run_group_ABC_chroma.py         # A/B/C 실험 러너
│  ├─ queries_ko.json                 # 20~30개 질의(명확/애매/OOT 혼합)
│  ├─ results/abc_results.csv         # 질문×그룹 단위 결과
│  └─ results/abc_summary.csv         # 그룹 평균 요약
│
├─ data/
│  └─ 10000recipe_dataset.csv         # (선택) 컨텍스트 보강용 CSV
└─ ...


실제 경로가 다르면 README의 명령어에서 경로만 바꿔주세요.

⚙️ 요구사항

Python 3.10+

pip install -U pandas numpy fastapi uvicorn python-dotenv langchain-openai langchain-chroma chromadb tiktoken

필수 환경변수:

OPENAI_API_KEY=sk-...


선택 환경변수(권장):

# CRAG / Retrieve
RAG_PERSIST_DIR=.../chroma_rag_hybrid_db
RAG_COLLECTION_NAME=recipe_hybrid_rag
RAG_TOP_K=4
FILTER_SCORE=0.20          # 필터 임계값(0~1). 0.15~0.30 권장 탐색

# OOD 분류
OOD_MODEL=gpt-4o-mini
OOD_TEMPERATURE=0

# Clarify
CLARIFY_MODE=ask           # ask | auto

# Group A 베이스라인
GROUPA_PERSIST=.../chroma_rag_hybrid_db
GROUPA_COLLECTION=recipe_hybrid_rag
GROUPA_EMBED_MODEL=text-embedding-3-small
GROUPA_SCORE_THRESHOLD=0.0
GROUPA_CSV=.../10000recipe_dataset.csv   # (선택) 컨텍스트 보강

🚀 빠른 시작
1) Group A 베이스라인 서버 실행
cd groupA_rag_flow
uvicorn app_groupA:app --host 0.0.0.0 --port 8001 --reload


질의 테스트:

curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"계란으로 만들 수 있는 메뉴와 레시피를 알려줘","k":4}'


검색 문서가 없으면 절대 생성하지 않고 “제공된 문서가 없어…” 메시지를 반환합니다.
GROUPA_CSV를 지정하면 url/recipe_id/title로 재료/조리를 보강해 품질 향상.

2) A/B/C 실험 실행
cd experiments_rag_groups_chroma
python run_group_ABC_chroma.py \
  --queries queries_ko.json \
  --threshold 0.20 \
  --top_k 4


출력:

results/abc_results.csv

results/abc_summary.csv

📊 실험 설계
그룹 정의

A: 저관련/키워드 미포함이라도 그냥 답변 (Baseline)

B: 저관련 or 키워드 미포함 → 질문 재작성 → 재검색 → 답변
(재작성 후 low/top1 재계산하여 전후 개선 지표 기록)

C: 저관련 or 키워드 미포함 → 거절(“답변할 수 없음”)

평가 지표 (1~5점)

Faithfulness (문서 근거 충실) – 높을수록 좋음

Relevance (질문과 직접 관련) – 높을수록 좋음

Hallucination (문서에 없는 내용 생성) – 낮을수록 좋음

보조: Hall≥3 비율, 응답률/거절률, low_rate, Top-1 평균/분포, 지연/토큰.

🧠 개선된 CRAG 워크플로우

OOD Guard(요리/레시피/영양 분류 in/out)

Retrieve(Chroma Top-K + relevance scores)

LLM Answer 1 → Relevance Check 1

grounded → 종료

notSure → Similarity Filter(FILTER_SCORE) →

남음: Answer 2 → Relevance Check 2

없음: Clarify(ask: 사용자 재질문 | auto: 자동 재작성→재검색)

notGrounded → Web Search(최후 Fallback) → Answer 2 → Check 2

📈 실험 결과(요약)
Group	avg_faith	avg_rel	avg_hall	low_rate
A	2.8077	3.0385	1.1538	0.5769
B	2.7308	2.9231	1.0769	0.5769
C	2.8462	3.3846	1.0000	0.5769

해석:

**C(거절)**가 관련성·안전성에서 가장 우수(저관련 시 응답 회피).

A/B는 비슷 — 기존 세팅에서 재작성의 이득이 제한적.

low_rate 동일은 초기 버전의 로깅 문제 → 재작성 후 재계산으로 해결(코드 반영).

🧪 재현 팁

threshold(저관련 컷): 0.15~0.30 구간에서 데이터 분포에 맞춰 조정.

FILTER_SCORE: 과도하면 근거가 사라져 Clarify 빈발, 낮으면 잡음↑ → 0.20부터 탐색.

키워드 게이트: “메뉴/레시피/레시피명/조리/조리순서/방법/요리/만드는법/요리법/recipe/cook”.

🛠️ 트러블슈팅

retrieved_count=0: PERSIST_DIR/COLLECTION 경로·컬렉션명 확인, 임베딩 모델 일치 여부 확인.

문서는 있는데 답변 불가: 컨텍스트가 비었을 수 있음 → CSV 보강(data/10000recipe_dataset.csv) 경로 지정.

웹검색 실패: 키·도메인 제한 확인(SERPER_API_KEY 등).

OOD 분류 오탐/누락: KW_IN/KW_OUT 휴리스틱 보강 또는 OOD LLM 프롬프트에 예시 추가.

🗺️ 로드맵

질의 카테고리별 분석(명확/애매/OOT) 리포트 자동화

Top-1 분포/표준편차/Hall≥3 시계열 대시보드

인덱싱 파이프라인 개선: page_content에 제목/재료/조리 저장(현재 CSV 보강 의존도↓)

🤝 기여

PR/이슈 환영합니다. 재현 가능한 예시(환경변수, 질의, 로그)를 함께 남겨주세요.

🧾 라이선스

본 레포의 코드와 문서의 라이선스는 LICENSE 파일을 참고하세요.

🙏 감사

이 프로젝트는 산학 협력 프로젝트(테스트빌더 RAG) 맥락에서 개발되었습니다.
문의/피드백은 이슈 트래커나 PR로 남겨주세요.
