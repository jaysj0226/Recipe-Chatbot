`.env` 파일을 프로젝트 루트에 생성:

```env
# OpenAI API Key
OPENAI_API_KEY=your_api_key_here

# Model Configuration
GROUPA_ROUTER_MODEL=gpt-4o-mini
GROUPA_GENERATION_MODEL=gpt-4o-mini
GROUPA_REWRITE_MODEL=gpt-4o-mini

# Feature Flags
ALLOW_NO_CONTEXT_ANSWER=1
ENABLE_QUERY_REWRITE=1
GROUPA_DEBUG_RAW=0

# Search Configuration
GROUPA_SCORE_THRESHOLD=0.0
```

### 3. 벡터 DB 경로 수정

`config/settings.py`에서 벡터 DB 경로를 본인 환경에 맞게 수정:

```python
VECTOR_DIR = r"C:/Users/YourName/path/to/chroma_db"
```

### 4. 서버 실행

```bash
# 방법 1: Python 직접 실행
python main.py

# 방법 2: Uvicorn으로 실행
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## 📡 API 엔드포인트

### 메인 엔드포인트

#### POST `/ask`
RAG 파이프라인 실행

**Request:**
```json
{
  "query": "김치찌개 만드는 법",
  "k": 10,
  "model": "gpt-4o-mini",
  "enable_rewrite": true
}
```

**Response:**
```json
{
  "answer": "...",
  "intent": "recipe",
  "original_query": "김치찌개 만드는 법",
  "rewritten_query": "...",
  "context_found": true,
  "mode": "context_based",
  "pipeline": ["router", "rewrite", "retrieve", "context_builder", "generate"]
}
```

### 테스트 엔드포인트

- `GET /health` - 서비스 상태 확인
- `GET /doc_count` - 벡터 DB 문서 수 확인
- `GET /test_search/{query}` - 검색 기능 테스트
- `POST /test_pipeline` - 파이프라인 각 단계 확인

## 🔧 노드별 역할

### 1. Router Node (`router_node.py`)
- **역할**: 사용자 질문의 의도 분류
- **출력**: `intent`, `needs_retrieval`, `notes`

### 2. Rewrite Node (`rewrite_node.py`)
- **역할**: 검색 최적화를 위한 쿼리 재작성
- **출력**: 재작성된 쿼리 문자열

### 3. Retrieve Node (`retrieve_node.py`)
- **역할**: 벡터 DB에서 유사 문서 검색
- **출력**: `retrieved_docs`, `retrieved_scores`, `branch`

### 4. Context Builder Node (`context_builder_node.py`)
- **역할**: 검색된 문서를 컨텍스트로 구성
- **출력**: 포맷팅된 컨텍스트 문자열

### 5. Generate Node (`generate_node.py`)
- **역할**: LLM으로 최종 답변 생성
- **출력**: 생성된 답변 문자열

## 🎯 지원 Intent

- `recipe` - 레시피 요청
- `dish_overview` - 음식 소개
- `storage` - 보관 방법
- `substitution` - 재료 대체
- `nutrition` - 영양 정보
- `equipment` - 조리 도구
- `shopping` - 장보기 팁
- `unknown` - 일반 질문
- `out_of_domain` - 도메인 외 질문

## 📝 개발 가이드

### 새로운 노드 추가

1. `nodes/` 디렉토리에 새 파일 생성
2. 노드 함수 작성 (입력/출력 명확히)
3. `main.py`의 파이프라인에 통합

### 프롬프트 수정

`prompts/templates.py`에서 각 Intent별 프롬프트 수정 가능

### 설정 변경

`config/settings.py`에서 전역 설정 관리

## 🐛 문제 해결

### 파일 잠금 오류 (Windows)
- `@lru_cache`와 `allow_reset=False` 설정으로 해결됨

### 벡터 DB 연결 실패
- `VECTOR_DIR` 경로 확인
- Chroma DB 권한 확인

### 모듈 import 오류
- 프로젝트 루트에서 실행 확인
- `__init__.py` 파일 존재 확인

## 📊 성능 모니터링

- `retrieved_scores` - 검색 품질 확인
- `context_len` - 컨텍스트 크기 모니터링
- `pipeline` - 실행된 노드 추적

## 🔐 보안
- 프로덕션에서는 HTTPS 사용 권장

## 📄 라이선스

MIT License

## CRAG 통합(정확도 향상)

- 활성화: 환경변수 `ENABLE_CRAG=1` (기본값 1)
- Judge 모델: `GROUPA_JUDGE_MODEL` (기본값 `gpt-4o-mini`)
- 동작 요약:
  - 1차 답변 생성 후 판정 노드가 답변의 근거 충실도를 평가합니다.
  - 판정이 `notGrounded`/`notSure` 이면 자동으로 2차 루프(질문 재작성 → 재검색 → 재생성 → 재판정)를 수행합니다.
  - 응답 필드: `judge_verdict_1`, `judge_verdict_2`, `corrected`, `final_pass`.
  - 파이프라인에는 `judge1`, `rewrite2`, `retrieve2`, `context_builder2`, `generate2`, `judge2` 단계가 추가될 수 있습니다.

## 이미지 URL 포함 응답

- 임베딩 DB 문서 메타데이터/본문에 포함된 이미지 URL을 추출하여 `image_urls` 필드로 함께 반환합니다.
- 예: `image_urls: ["https://...", "https://..."]` (최대 5개)
