import asyncio
import time
import json
from datetime import datetime
from typing import List, Dict
import httpx
from pydantic import BaseModel

# ========== 평가 설정 ==========
API_URL = "http://127.0.0.1:8000/ask"
TIMEOUT = 30.0

# ========== 기존 평가 데이터 로드 ==========
PREVIOUS_RESULTS_FILE = "evaluation_results_20251001_222348.json"

# 평가에 사용할 질문들 (기존 JSON에서 추출)
TEST_QUESTIONS = [
    {"query": "김치찌개 레시피 알려줘", "category": "기본_레시피"},
    {"query": "된장찌개 만드는 법", "category": "기본_레시피"},
    {"query": "불고기 레시피", "category": "기본_레시피"},
    {"query": "계란찜 만들기", "category": "기본_레시피"},
    {"query": "닭가슴살로 만들 수 있는 요리 추천해줘", "category": "재료_기반"},
    {"query": "두부를 활용한 레시피", "category": "재료_기반"},
    {"query": "고구마 요리법", "category": "재료_기반"},
    {"query": "손님 초대할 때 좋은 요리", "category": "상황별"},
    {"query": "10분 안에 만들 수 있는 간단한 요리", "category": "상황별"},
    {"query": "다이어트 식단 추천", "category": "상황별"},
    {"query": "밥 짓는 방법", "category": "조리_방법"},
    {"query": "계란 삶는 시간", "category": "조리_방법"},
    {"query": "고기 재우는 방법", "category": "조리_방법"},
    {"query": "김치찌개 만들 때 신김치 대신 일반 김치 써도 돼?", "category": "복합_질문"},
    {"query": "된장찌개와 청국장찌개의 차이점", "category": "복합_질문"},
    {"query": "파스타 면 삶을 때 소금을 왜 넣어?", "category": "복합_질문"},
    {"query": "김치찌개 칼로리", "category": "영양_정보"},
    {"query": "고단백 저칼로리 음식", "category": "영양_정보"},
    {"query": "김치 보관 방법", "category": "보관_방법"},
    {"query": "남은 찌개 어떻게 보관해?", "category": "보관_방법"},
]


# ========== 평가 결과 모델 ==========
class RoutingRAGResult(BaseModel):
    query: str
    category: str
    response_time: float
    status: str
    response_length: int
    error_message: str = ""
    response_preview: str = ""
    # Routing RAG 특화 필드
    intent: str = ""
    needs_retrieval: bool = True
    rewritten_query: str = ""
    context_found: bool = False
    used_docs: int = 0
    retrieved_count: int = 0
    mode: str = ""  # context_based or general_knowledge


# ========== API 호출 함수 ==========
async def call_routing_rag_api(query: str, k: int = 10, model: str = "gpt-4o-mini") -> tuple[dict, float, str]:
    """
    Routing RAG API 호출 및 응답 시간 측정
    Returns: (response_data, response_time, status)
    """
    start_time = time.time()
    
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            response = await client.post(
                API_URL,
                json={
                    "query": query,
                    "k": k,
                    "model": model
                }
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


# ========== 단일 질문 평가 ==========
async def evaluate_single_query(test_case: dict) -> RoutingRAGResult:
    """
    개별 질문에 대한 Routing RAG 평가 수행
    """
    query = test_case["query"]
    category = test_case["category"]
    
    # API 호출
    response_data, response_time, status = await call_routing_rag_api(query)
    
    # 응답 파싱
    answer = response_data.get("answer", "")
    intent = response_data.get("intent", "unknown")
    router_info = response_data.get("router", {})
    needs_retrieval = router_info.get("needs_retrieval", True) if router_info else True
    rewritten_query = router_info.get("rewritten_query", query) if router_info else query
    context_found = response_data.get("context_found", False)
    used_docs = response_data.get("used_docs", 0)
    retrieved_count = response_data.get("retrieved_count", 0)
    mode = response_data.get("mode", "unknown")
    
    # 에러 메시지
    error_msg = ""
    if status != "success":
        error_msg = response_data.get("error", f"Status: {status}")
    
    return RoutingRAGResult(
        query=query,
        category=category,
        response_time=response_time,
        status=status,
        response_length=len(answer),
        error_message=error_msg,
        response_preview=answer[:200] if answer else "",
        intent=intent,
        needs_retrieval=needs_retrieval,
        rewritten_query=rewritten_query,
        context_found=context_found,
        used_docs=used_docs,
        retrieved_count=retrieved_count,
        mode=mode
    )


# ========== 전체 평가 실행 ==========
async def run_evaluation():
    """
    모든 테스트 케이스 실행 및 결과 집계
    """
    print("=" * 80)
    print("🔀 Routing RAG 시스템 평가 시작")
    print("=" * 80)
    print(f"총 테스트 케이스: {len(TEST_QUESTIONS)}개")
    print(f"API 엔드포인트: {API_URL}")
    print(f"타임아웃: {TIMEOUT}초\n")
    
    results: List[RoutingRAGResult] = []
    
    # 각 질문 평가 (순차 실행)
    for i, test_case in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] 평가 중: {test_case['query'][:50]}...")
        result = await evaluate_single_query(test_case)
        results.append(result)
        
        # 간단한 진행 상황 출력
        status_icon = "✅" if result.status == "success" else "❌"
        mode_icon = "📚" if result.mode == "context_based" else "🧠"
        print(f"    {status_icon} {mode_icon} [{result.intent}] {result.response_time:.2f}초 | {result.response_length}자 | 문서 {result.used_docs}개")
        
        # API 부하 방지를 위한 약간의 딜레이
        await asyncio.sleep(0.5)
    
    print("\n" + "=" * 80)
    print("📊 Routing RAG 평가 결과 요약")
    print("=" * 80)
    
    # 통계 계산
    total = len(results)
    successful = sum(1 for r in results if r.status == "success")
    avg_response_time = sum(r.response_time for r in results) / total
    
    # Context 기반 vs 일반 지식 비율
    context_based = sum(1 for r in results if r.mode == "context_based")
    general_knowledge = sum(1 for r in results if r.mode == "general_knowledge")
    
    print(f"\n✅ 성공률: {successful}/{total} ({successful/total*100:.1f}%)")
    print(f"⏱️  평균 응답 시간: {avg_response_time:.2f}초")
    print(f"📚 Context 기반 응답: {context_based}/{successful} ({context_based/successful*100 if successful > 0 else 0:.1f}%)")
    print(f"🧠 일반 지식 응답: {general_knowledge}/{successful} ({general_knowledge/successful*100 if successful > 0 else 0:.1f}%)")
    
    # 응답 시간 분포
    if successful > 0:
        success_times = [r.response_time for r in results if r.status == "success"]
        print(f"\n응답 시간 분포:")
        print(f"  - 최소: {min(success_times):.2f}초")
        print(f"  - 최대: {max(success_times):.2f}초")
        print(f"  - 중앙값: {sorted(success_times)[len(success_times)//2]:.2f}초")
    
    # Intent별 성능
    print(f"\n🎯 Intent별 성능:")
    intents = {}
    for r in results:
        if r.intent not in intents:
            intents[r.intent] = []
        intents[r.intent].append(r)
    
    for intent, intent_results in sorted(intents.items()):
        intent_success = sum(1 for r in intent_results if r.status == "success")
        intent_total = len(intent_results)
        intent_avg_time = sum(r.response_time for r in intent_results) / intent_total
        intent_context = sum(1 for r in intent_results if r.context_found)
        print(f"  - {intent}: {intent_success}/{intent_total} 성공, 평균 {intent_avg_time:.2f}초, 컨텍스트 {intent_context}개")
    
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
        cat_context = sum(1 for r in cat_results if r.context_found)
        print(f"  - {cat}: {cat_success}/{cat_total} 성공, 평균 {cat_avg_time:.2f}초, 컨텍스트 {cat_context}개")
    
    # 문서 활용 통계
    print(f"\n📖 문서 활용 통계:")
    doc_counts = [r.used_docs for r in results if r.status == "success"]
    if doc_counts:
        print(f"  - 평균 사용 문서: {sum(doc_counts)/len(doc_counts):.1f}개")
        print(f"  - 최대 사용 문서: {max(doc_counts)}개")
        print(f"  - 최소 사용 문서: {min(doc_counts)}개")
    
    # 실패한 케이스 출력
    failed = [r for r in results if r.status != "success"]
    if failed:
        print(f"\n❌ 실패한 케이스 ({len(failed)}개):")
        for r in failed:
            print(f"  - {r.query[:60]}")
            print(f"    상태: {r.status}, 에러: {r.error_message}")
    
    # Query 재작성 예시
    print(f"\n✏️  Query 재작성 예시 (상위 5개):")
    for i, r in enumerate(results[:5], 1):
        if r.rewritten_query != r.query:
            print(f"  {i}. 원본: {r.query}")
            print(f"     재작성: {r.rewritten_query}")
    
    # 결과를 JSON 파일로 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"routing_rag_results_{timestamp}.json"
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            [r.model_dump() for r in results],
            f,
            ensure_ascii=False,
            indent=2
        )
    
    print(f"\n💾 상세 결과가 저장되었습니다: {filename}")
    
    # 기존 결과와 비교 (있는 경우)
    try:
        with open(PREVIOUS_RESULTS_FILE, "r", encoding="utf-8") as f:
            previous_results = json.load(f)
        
        print(f"\n📊 이전 결과와 비교:")
        prev_success = sum(1 for r in previous_results if r.get("status") == "success")
        prev_total = len(previous_results)
        prev_avg_time = sum(r.get("response_time", 0) for r in previous_results) / prev_total
        
        print(f"  성공률: {successful}/{total} ({successful/total*100:.1f}%) vs 이전 {prev_success}/{prev_total} ({prev_success/prev_total*100:.1f}%)")
        print(f"  평균 응답시간: {avg_response_time:.2f}초 vs 이전 {prev_avg_time:.2f}초")
        
        improvement = (successful/total - prev_success/prev_total) * 100
        time_diff = avg_response_time - prev_avg_time
        
        if improvement > 0:
            print(f"  ✅ 성공률 {improvement:.1f}%p 향상")
        elif improvement < 0:
            print(f"  ⚠️ 성공률 {abs(improvement):.1f}%p 하락")
        
        if time_diff < 0:
            print(f"  ⚡ 응답시간 {abs(time_diff):.2f}초 단축")
        elif time_diff > 0:
            print(f"  ⏳ 응답시간 {time_diff:.2f}초 증가")
            
    except FileNotFoundError:
        print(f"\n⚠️  이전 결과 파일을 찾을 수 없습니다: {PREVIOUS_RESULTS_FILE}")
    
    return results


# ========== 메인 실행 ==========
if __name__ == "__main__":
    print("\n🚀 Routing RAG 시스템 평가를 시작합니다...")
    print("⚠️  서버가 http://127.0.0.1:8000 에서 실행 중인지 확인하세요!\n")
    
    asyncio.run(run_evaluation())
