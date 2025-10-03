import asyncio
import time
import json
from datetime import datetime
from typing import List, Dict
import httpx
from pydantic import BaseModel

# ========== 평가 설정 ==========
API_URL = "http://127.0.0.1:8000/query"
TIMEOUT = 30.0

# ========== 테스트 질문 데이터셋 ==========
TEST_QUESTIONS = [
    # 기본 레시피 검색
    {
        "query": "김치찌개 레시피 알려줘",
        "category": "기본_레시피",
        "expected_keywords": ["김치", "돼지고기", "두부", "국물"]
    },
    {
        "query": "된장찌개 만드는 법",
        "category": "기본_레시피",
        "expected_keywords": ["된장", "두부", "애호박", "감자"]
    },
    {
        "query": "불고기 레시피",
        "category": "기본_레시피",
        "expected_keywords": ["소고기", "간장", "설탕", "배"]
    },
    {
        "query": "계란찜 만들기",
        "category": "기본_레시피",
        "expected_keywords": ["계란", "물", "소금", "찜"]
    },
    
    # 특정 재료 기반 질문
    {
        "query": "닭가슴살로 만들 수 있는 요리 추천해줘",
        "category": "재료_기반",
        "expected_keywords": ["닭가슴살", "요리", "추천"]
    },
    {
        "query": "두부를 활용한 레시피",
        "category": "재료_기반",
        "expected_keywords": ["두부"]
    },
    {
        "query": "고구마 요리법",
        "category": "재료_기반",
        "expected_keywords": ["고구마"]
    },
    
    # 상황별 질문
    {
        "query": "손님 초대할 때 좋은 요리",
        "category": "상황별",
        "expected_keywords": ["요리", "추천"]
    },
    {
        "query": "10분 안에 만들 수 있는 간단한 요리",
        "category": "상황별",
        "expected_keywords": ["간단", "빠른", "쉬운"]
    },
    {
        "query": "다이어트 식단 추천",
        "category": "상황별",
        "expected_keywords": ["다이어트", "칼로리", "건강"]
    },
    
    # 조리 방법 질문
    {
        "query": "밥 짓는 방법",
        "category": "조리_방법",
        "expected_keywords": ["밥", "쌀", "물"]
    },
    {
        "query": "계란 삶는 시간",
        "category": "조리_방법",
        "expected_keywords": ["계란", "삶", "시간", "분"]
    },
    {
        "query": "고기 재우는 방법",
        "category": "조리_방법",
        "expected_keywords": ["재우", "양념", "시간"]
    },
    
    # 복합 질문
    {
        "query": "김치찌개 만들 때 신김치 대신 일반 김치 써도 돼?",
        "category": "복합_질문",
        "expected_keywords": ["김치", "가능", "대체"]
    },
    {
        "query": "된장찌개와 청국장찌개의 차이점",
        "category": "복합_질문",
        "expected_keywords": ["된장", "청국장", "차이"]
    },
    {
        "query": "파스타 면 삶을 때 소금을 왜 넣어?",
        "category": "복합_질문",
        "expected_keywords": ["소금", "이유", "면"]
    },
    
    # 영양 정보 질문
    {
        "query": "김치찌개 칼로리",
        "category": "영양_정보",
        "expected_keywords": ["칼로리", "kcal"]
    },
    {
        "query": "고단백 저칼로리 음식",
        "category": "영양_정보",
        "expected_keywords": ["단백질", "칼로리"]
    },
    
    # 저장 및 보관
    {
        "query": "김치 보관 방법",
        "category": "보관_방법",
        "expected_keywords": ["보관", "냉장", "저장"]
    },
    {
        "query": "남은 찌개 어떻게 보관해?",
        "category": "보관_방법",
        "expected_keywords": ["보관", "냉장", "용기"]
    },
]


# ========== 평가 결과 모델 ==========
class EvaluationResult(BaseModel):
    query: str
    category: str
    response_time: float
    status: str  # success, error, timeout
    response_length: int
    has_expected_keywords: bool
    keyword_match_rate: float
    error_message: str = ""
    response_preview: str = ""


# ========== API 호출 함수 ==========
async def call_api(query: str) -> tuple[dict, float, str]:
    """
    API 호출 및 응답 시간 측정
    Returns: (response_data, response_time, status)
    """
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            response = await client.post(
                API_URL,
                json={"query": query}
            )
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                return response.json(), elapsed, "success"
            else:
                return {}, elapsed, f"error_{response.status_code}"
                
        except httpx.TimeoutException:
            elapsed = time.time() - start_time
            return {}, elapsed, "timeout"
        except Exception as e:
            elapsed = time.time() - start_time
            return {"error": str(e)}, elapsed, "error"


# ========== 키워드 매칭 평가 ==========
def evaluate_keywords(response_text: str, expected_keywords: List[str]) -> tuple[bool, float]:
    """
    응답에 예상 키워드가 포함되어 있는지 평가
    """
    if not expected_keywords:
        return True, 1.0
    
    response_lower = response_text.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    match_rate = matched / len(expected_keywords)
    
    return matched > 0, match_rate


# ========== 응답 텍스트 추출 ==========
def extract_response_text(response_data: dict) -> str:
    """
    API 응답에서 실제 텍스트 추출
    """
    # RAGState 구조에 따라 조정 필요
    if isinstance(response_data, dict):
        # 일반적인 패턴들 시도
        if "answer" in response_data:
            return str(response_data["answer"])
        elif "response" in response_data:
            return str(response_data["response"])
        elif "result" in response_data:
            return str(response_data["result"])
        else:
            return json.dumps(response_data, ensure_ascii=False)
    
    return str(response_data)


# ========== 단일 질문 평가 ==========
async def evaluate_single_query(test_case: dict) -> EvaluationResult:
    """
    개별 질문에 대한 평가 수행
    """
    query = test_case["query"]
    category = test_case["category"]
    expected_keywords = test_case.get("expected_keywords", [])
    
    # API 호출
    response_data, response_time, status = await call_api(query)
    
    # 응답 텍스트 추출
    response_text = extract_response_text(response_data)
    
    # 키워드 평가
    has_keywords, keyword_rate = evaluate_keywords(response_text, expected_keywords)
    
    # 에러 메시지
    error_msg = ""
    if status != "success":
        error_msg = response_data.get("error", f"Status: {status}")
    
    return EvaluationResult(
        query=query,
        category=category,
        response_time=response_time,
        status=status,
        response_length=len(response_text),
        has_expected_keywords=has_keywords,
        keyword_match_rate=keyword_rate,
        error_message=error_msg,
        response_preview=response_text[:200]
    )


# ========== 전체 평가 실행 ==========
async def run_evaluation():
    """
    모든 테스트 케이스 실행 및 결과 집계
    """
    print("=" * 80)
    print("🧪 레시피 챗봇 평가 시작")
    print("=" * 80)
    print(f"총 테스트 케이스: {len(TEST_QUESTIONS)}개")
    print(f"API 엔드포인트: {API_URL}")
    print(f"타임아웃: {TIMEOUT}초\n")
    
    results: List[EvaluationResult] = []
    
    # 각 질문 평가 (순차 실행)
    for i, test_case in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] 평가 중: {test_case['query'][:50]}...")
        result = await evaluate_single_query(test_case)
        results.append(result)
        
        # 간단한 진행 상황 출력
        status_icon = "✅" if result.status == "success" else "❌"
        print(f"    {status_icon} {result.response_time:.2f}초 | {result.response_length}자")
        
        # API 부하 방지를 위한 약간의 딜레이
        await asyncio.sleep(0.5)
    
    print("\n" + "=" * 80)
    print("📊 평가 결과 요약")
    print("=" * 80)
    
    # 통계 계산
    total = len(results)
    successful = sum(1 for r in results if r.status == "success")
    avg_response_time = sum(r.response_time for r in results) / total
    avg_keyword_match = sum(r.keyword_match_rate for r in results if r.status == "success")
    avg_keyword_match = avg_keyword_match / successful if successful > 0 else 0
    
    print(f"\n✅ 성공률: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"⏱️  평균 응답 시간: {avg_response_time:.2f}초")
    print(f"🎯 평균 키워드 매칭률: {avg_keyword_match*100:.1f}%")
    
    # 응답 시간 분포
    if successful > 0:
        success_times = [r.response_time for r in results if r.status == "success"]
        print(f"\n응답 시간 분포:")
        print(f"  - 최소: {min(success_times):.2f}초")
        print(f"  - 최대: {max(success_times):.2f}초")
        print(f"  - 중앙값: {sorted(success_times)[len(success_times)//2]:.2f}초")
    
    # 카테고리별 성능
    print(f"\n📁 카테고리별 성능:")
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)
    
    for cat, cat_results in sorted(categories.items()):
        cat_success = sum(1 for r in cat_results if r.status == "success")
        cat_total = len(cat_results)
        cat_avg_time = sum(r.response_time for r in cat_results) / cat_total
        print(f"  - {cat}: {cat_success}/{cat_total} 성공, 평균 {cat_avg_time:.2f}초")
    
    # 실패한 케이스 출력
    failed = [r for r in results if r.status != "success"]
    if failed:
        print(f"\n❌ 실패한 케이스 ({len(failed)}개):")
        for r in failed:
            print(f"  - {r.query[:60]}")
            print(f"    상태: {r.status}, 에러: {r.error_message}")
    
    # 결과를 JSON 파일로 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_results_{timestamp}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump() for r in results],
            f,
            ensure_ascii=False,
            indent=2
        )
    
    print(f"\n💾 상세 결과가 저장되었습니다: {filename}")
    
    return results


# ========== 메인 실행 ==========
if __name__ == "__main__":
    asyncio.run(run_evaluation())
