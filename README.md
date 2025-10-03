# 🍳 Recipe Chatbot - Routing RAG System

AI 기반 레시피 추천 및 요리 질의응답 챗봇 시스템입니다. LLM Router를 활용한 Intent 분류와 RAG(Retrieval-Augmented Generation) 기술을 결합하여 정확하고 유용한 요리 정보를 제공합니다.

## 📋 목차

- [주요 기능](#주요-기능)
- [시스템 아키텍처](#시스템-아키텍처)
- [기술 스택](#기술-스택)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [API 문서](#api-문서)
- [평가 및 테스트](#평가-및-테스트)
- [프로젝트 구조](#프로젝트-구조)

## ✨ 주요 기능

### 🔀 Routing RAG System
- **Intent Classification**: LLM 기반 질문 의도 자동 분류
  - Recipe (레시피)
  - Storage (보관 방법)
  - Dish Overview (요리 개요)
  - Nutrition (영양 정보)
  - Substitution (대체재)
  - Equipment (조리 도구)
  - Shopping (장보기)

### 🧠 Hybrid Answer Generation
- **Context-based**: 벡터 DB에서 관련 문서를 검색하여 답변
- **General Knowledge**: 컨텍스트가 없을 경우 LLM의 일반 지식 활용
- **Query Rewriting**: Router가 검색에 최적화된 쿼리로 재작성

### 📚 Vector Database
- **ChromaDB** 기반 레시피 문서 저장
- **OpenAI Embeddings** (text-embedding-3-large) 활용
- 효율적인 유사도 검색

## 🏗 시스템 아키텍처

```
Query → Router (Intent Classification) → Vector DB Search → LLM Prompt → Answer
         ↓                                      ↓
    Intent 분류                           Context 검색
    - Recipe                              - 관련 문서 5개
    - Storage                             - 중복 제거
    - Nutrition                           - 포맷 변환
    - etc.                                
                                              ↓
                                     Intent별 특화 프롬프트
                                              ↓
                                        답변 생성
```

### 워크플로우

1. **Query 입력**: 사용자가 요리 관련 질문 입력
2. **Router**: LLM이 질문의 Intent를 분석 및 분류
3. **Query Rewriting**: 검색에 최적화된 쿼리로 재작성
4. **Vector Search**: ChromaDB에서 관련 문서 검색
5. **Context Building**: 검색된 문서를 컨텍스트로 구성
6. **LLM Generation**: Intent별 특화 프롬프트로 답변 생성
7. **Fallback**: 컨텍스트 부족 시 일반 지식으로 대응

## 🛠 기술 스택

### Backend
- **FastAPI**: 고성능 웹 프레임워크
- **LangChain**: LLM 오케스트레이션
- **OpenAI GPT-4o-mini**: 라우팅 및 답변 생성
- **ChromaDB**: 벡터 데이터베이스
- **Python 3.10+**

### Frontend
- **HTML/CSS/JavaScript**: 웹 인터페이스
- **Responsive Design**: 모바일 친화적 UI

### Embeddings & Models
- **text-embedding-3-large**: 문서 임베딩
- **gpt-4o-mini**: Router 및 답변 생성

## 📥 설치 방법

### 1. 저장소 클론

```bash
git clone https://github.com/jaysj0226/Recipe-Chatbot.git
cd Recipe-Chatbot
```

### 2. 가상환경 생성 및 활성화

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 입력:

```env
OPENAI_API_KEY=your_openai_api_key_here
GROUPA_ROUTER_MODEL=gpt-4o-mini
GROUPA_SCORE_THRESHOLD=0.0
ALLOW_NO_CONTEXT_ANSWER=1
GROUPA_DEBUG_RAW=0
```

### 5. ChromaDB 설정

벡터 DB 경로를 설정하거나 새로 생성:

```python
# chatbot_routing_rag.py에서 경로 수정
PERSIST = "path/to/your/chroma_db"
COLLECTION = "recipes-v1"
```

## 🚀 사용 방법

### 서버 실행

```bash
# 기본 실행
python chatbot_routing_rag.py

# 또는 uvicorn 직접 실행
uvicorn chatbot_routing_rag:app --host 127.0.0.1 --port 8000 --reload
```

### 웹 인터페이스 접속

브라우저에서 `http://127.0.0.1:8000` 접속

### API 호출 예시

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/ask",
    json={
        "query": "김치찌개 레시피 알려줘",
        "k": 10,
        "model": "gpt-4o-mini"
    }
)

print(response.json())
```

## 📡 API 문서

### POST /ask

레시피 질문에 대한 답변을 생성합니다.

**Request Body:**
```json
{
  "query": "김치찌개 레시피 알려줘",
  "k": 10,
  "model": "gpt-4o-mini"
}
```

**Response:**
```json
{
  "answer": "김치찌개를 만들기 위한 레시피...",
  "intent": "recipe",
  "router": {
    "intent": "recipe",
    "needs_retrieval": true,
    "rewritten_query": "김치찌개 만드는 방법 재료",
    "notes": ""
  },
  "context_found": true,
  "used_docs": 3,
  "retrieved_count": 10,
  "mode": "context_based",
  "sources": [...]
}
```

### GET /health

시스템 상태를 확인합니다.

**Response:**
```json
{
  "ok": true,
  "persist": "path/to/chroma_db",
  "collection": "recipes-v1",
  "embed_model": "text-embedding-3-large",
  "total_docs": 1500,
  "router_model": "gpt-4o-mini",
  "status": "일반 지식 활용 가능"
}
```

### GET /test_search/{query}

벡터 검색을 테스트합니다.

**Response:**
```json
{
  "query": "김치찌개",
  "found_docs": 5,
  "results": [
    {
      "title": "김치찌개 레시피",
      "url": "https://...",
      "content_length": 1024,
      "content_preview": "김치찌개를 만들기 위한..."
    }
  ]
}
```

## 🧪 평가 및 테스트

### 자동 평가 실행
./experiments/experiment_codes/에서 crag_test.py와 routing_rag_test.py
20개의 테스트 케이스로 시스템을 평가합니다:

CRAG 버전일 경우
```bash
python -m uvicorn main:app --reload
python crag_test
```
Routing RAG 버전일 경우

```bash
python -m uvicorn chatbot_routing_rag:app --reload
python routing_rag_test
```

### 평가 지표

- **성공률**: API 호출 성공 비율
- **응답 시간**: 평균/최소/최대/중앙값
- **Context 활용률**: Context 기반 vs 일반 지식 비율
- **Intent 분류 정확도**: Intent별 성공률
- **문서 활용 통계**: 평균 사용 문서 개수

### 평가 카테고리

- 기본 레시피 (4개)
- 재료 기반 (3개)
- 상황별 질문 (3개)
- 조리 방법 (3개)
- 복합 질문 (3개)
- 영양 정보 (2개)
- 보관 방법 (2개)

### 결과 예시

```
📊 Routing RAG 평가 결과 요약
================================
✅ 성공률: 20/20 (100.0%)
⏱️ 평균 응답 시간: 14.97초
📚 Context 기반 응답: 15/18 (83.3%)
🧠 일반 지식 응답: 3/18 (16.7%)

응답 시간 분포:
  - 최소: 12.34초
  - 최대: 30.27초
  - 중앙값: 20.15초
```

## 📂 Routing RAG 프로젝트 구조

```
Recipe-Chatbot/
├── chatbot_routing_rag.py          # 메인 서버 코드 (Routing RAG)
├── static/                         # 웹 인터페이스
│   └── index.html
│   ├── style.css
│   └── app.js
│       └── assets
│           ├── hero
│           ├── icons
│           └── special
└────────────────────────────
```

## 🔧 커스터마이징

### Intent 추가

`chatbot_routing_rag.py`에서 새로운 Intent를 추가:

```python
# Router에 Intent 추가
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "가능한 intent: ['recipe',...,'your_new_intent']\n"
     ...
])

# 새 프롬프트 정의
YOUR_NEW_PROMPT = ChatPromptTemplate.from_messages([...])

# 프롬프트 매핑에 추가
PROMPT_BY_INTENT = {
    ...
    "your_new_intent": YOUR_NEW_PROMPT
}
```

### 검색 파라미터 조정

```python
# 검색 문서 개수
k = 10  # 기본값

# 컨텍스트 최대 길이
context_text = "\n\n---\n\n".join(contexts)[:6000]

# 응답 온도
llm = ChatOpenAI(model=req.model, temperature=0.3)
```

## 📊 성능 벤치마크

| 항목 | CRAG | Routing RAG |
|------|----------|-------------|
| 성공률 | 90% | 100% |
| 평균 응답시간 | 22.4초 | 14.97초 |
| Context 활용률 | N/A | 83.3% |
| Intent 분류 | ❌ | ✅ |
| Fallback 지원 | ❌ | ✅ |

## 🤝 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

This project is licensed under the MIT License.

## 👥 개발자

- **jaysj0226** - [GitHub](https://github.com/jaysj0226)

## 📧 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해주세요.

---

**Made with ❤️ for better cooking experience**
