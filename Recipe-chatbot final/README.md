# Recipe RAG System (Modular Architecture)

모듈화된 구조의 요리 레시피 RAG 시스템

## 📁 프로젝트 구조

```
recipe_rag_project/
├── main.py                      # FastAPI 메인 애플리케이션
├── requirements.txt             # 의존성 패키지
├── .env                        # 환경 변수 (직접 생성)
│
├── config/                     # 설정 관리
│   ├── __init__.py
│   ├── settings.py            # 환경변수 & 전역 설정
│   └── schemas.py             # Pydantic 스키마
│
├── nodes/                      # RAG 파이프라인 노드
│   ├── __init__.py
│   ├── router_node.py         # Step 1: 의도 분류
│   ├── rewrite_node.py        # Step 2: 쿼리 재작성
│   ├── retrieve_node.py       # Step 3: 벡터 검색
│   ├── context_builder_node.py # Step 4: 컨텍스트 구성
│   └── generate_node.py       # Step 5: 답변 생성
│
├── prompts/                    # 프롬프트 템플릿
│   ├── __init__.py
│   └── templates.py           # Intent별 프롬프트
│
├── utils/                      # 유틸리티 함수
│   ├── __init__.py
│   ├── vectorstore.py         # 벡터스토어 관리
│   └── text_formatter.py      # 텍스트 포맷팅
│
└── static/                     # 프론트엔드 (옵션)
    └── index.html
```

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성:

```env
# OpenAI A능
