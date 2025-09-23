import requests
import csv
import time
from datetime import datetime
import json

# 챗봇 서버 설정 - 수정된 부분
BASE_URL = "http://127.0.0.1:8000"  # /docs 제거
ASK_ENDPOINT = f"{BASE_URL}/ask"

# 테스트 질문들 (다양한 intent와 난이도)
TEST_QUESTIONS = [
    # 기본 레시피 질문
    {"query": "김치찌개 만드는 법 알려줘", "category": "기본_레시피", "expected_intent": "recipe"},
    {"query": "불고기 레시피 단계별로 설명해줘", "category": "기본_레시피", "expected_intent": "recipe"},
    {"query": "된장찌개 끓이는 방법", "category": "기본_레시피", "expected_intent": "recipe"},
    {"query": "계란찜 만들기", "category": "기본_레시피", "expected_intent": "recipe"},
    {"query": "짜장면 만드는 법", "category": "기본_레시피", "expected_intent": "recipe"},
    
    # 음식 개요/소개 질문
    {"query": "비빔밥이 뭐야?", "category": "음식_소개", "expected_intent": "dish_overview"},
    {"query": "파스타에 대해 알려줘", "category": "음식_소개", "expected_intent": "dish_overview"},
    {"query": "마라탕은 어떤 음식이야?", "category": "음식_소개", "expected_intent": "dish_overview"},
    {"query": "타코의 특징은?", "category": "음식_소개", "expected_intent": "dish_overview"},
    
    # 재료 대체/치환 질문
    {"query": "설탕 대신 뭘 쓸 수 있어?", "category": "재료_대체", "expected_intent": "substitution"},
    {"query": "버터 없으면 뭘로 대체할까?", "category": "재료_대체", "expected_intent": "substitution"},
    {"query": "우유 대신 사용할 수 있는 재료", "category": "재료_대체", "expected_intent": "substitution"},
    {"query": "달걀 없이 케이크 만들 수 있어?", "category": "재료_대체", "expected_intent": "substitution"},
    
    # 보관/저장 질문
    {"query": "김치 어떻게 보관해야 해?", "category": "보관_저장", "expected_intent": "storage"},
    {"query": "고기 냉동보관 방법", "category": "보관_저장", "expected_intent": "storage"},
    {"query": "빵 오래 보관하는 법", "category": "보관_저장", "expected_intent": "storage"},
    {"query": "야채 신선하게 보관하기", "category": "보관_저장", "expected_intent": "storage"},
    
    # 영양 정보 질문
    {"query": "닭가슴살 칼로리 얼마야?", "category": "영양_정보", "expected_intent": "nutrition"},
    {"query": "아보카도 영양성분 알려줘", "category": "영양_정보", "expected_intent": "nutrition"},
    {"query": "현미의 영양가는?", "category": "영양_정보", "expected_intent": "nutrition"},
    
    # 도구/장비 질문
    {"query": "파스타 만들 때 필요한 도구는?", "category": "조리_도구", "expected_intent": "equipment"},
    {"query": "빵 굽는데 필요한 기구", "category": "조리_도구", "expected_intent": "equipment"},
    {"query": "중국요리 할 때 필요한 웍팬", "category": "조리_도구", "expected_intent": "equipment"},
    
    # 장보기/쇼핑 질문
    {"query": "파티 요리 재료 리스트", "category": "장보기", "expected_intent": "shopping"},
    {"query": "일주일치 밑반찬 재료", "category": "장보기", "expected_intent": "shopping"},
    {"query": "한식 기본 양념 뭐 사야 해?", "category": "장보기", "expected_intent": "shopping"},
    
    # 복합/어려운 질문
    {"query": "다이어트용 닭가슴살 요리 3가지와 각각의 칼로리", "category": "복합_질문", "expected_intent": "recipe"},
    {"query": "글루텐프리 빵 만들기와 보관법", "category": "복합_질문", "expected_intent": "recipe"},
    {"query": "비건 파스타 레시피와 영양성분", "category": "복합_질문", "expected_intent": "recipe"},
    
    # 애매한/모호한 질문
    {"query": "뭔가 매운 거 만들고 싶어", "category": "모호한_질문", "expected_intent": "unknown"},
    {"query": "간단한 요리 추천해줘", "category": "모호한_질문", "expected_intent": "unknown"},
    {"query": "오늘 저녁 뭐 먹지?", "category": "모호한_질문", "expected_intent": "unknown"},
    
    # 도메인 외 질문 (거절되어야 함)
    {"query": "오늘 날씨 어때?", "category": "도메인_외", "expected_intent": "out_of_domain"},
    {"query": "파이썬 코딩 도와줘", "category": "도메인_외", "expected_intent": "out_of_domain"},
    {"query": "주식 투자 조언해줘", "category": "도메인_외", "expected_intent": "out_of_domain"},
    
    # 특수 문자/이모지 포함 질문
    {"query": "🍕 피자 만들기! 😋", "category": "특수_문자", "expected_intent": "recipe"},
    {"query": "김치볶음밥 🔥🔥 레시피", "category": "특수_문자", "expected_intent": "recipe"},
    
    # 긴 질문
    {"query": "집에 양파, 당근, 감자, 돼지고기가 있는데 이 재료들로 맛있고 간단하게 만들 수 있는 요리가 뭐가 있을까요? 조리시간은 30분 이내였으면 좋겠어요.", "category": "긴_질문", "expected_intent": "recipe"},
]

def test_chatbot():
    """챗봇 테스트 실행"""
    print(f"🤖 챗봇 테스트 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"총 {len(TEST_QUESTIONS)}개의 질문을 테스트합니다.\n")
    
    results = []
    
    for i, test_case in enumerate(TEST_QUESTIONS, 1):
        print(f"[{i}/{len(TEST_QUESTIONS)}] 테스트 중: {test_case['query'][:50]}...")
        
        try:
            # API 호출
            start_time = time.time()
            response = requests.post(
                ASK_ENDPOINT,
                json={
                    "query": test_case["query"],
                    "k": 5,
                    "model": "gpt-4o-mini"
                },
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                # 결과 분석
                result = {
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "question_id": i,
                    "category": test_case["category"],
                    "query": test_case["query"],
                    "expected_intent": test_case["expected_intent"],
                    "actual_intent": data.get("intent", "unknown"),
                    "intent_match": test_case["expected_intent"] == data.get("intent", "unknown"),
                    "needs_retrieval": data.get("router", {}).get("needs_retrieval", False),
                    "rewritten_query": data.get("router", {}).get("rewritten_query", ""),
                    "retrieved_count": data.get("retrieved_count", 0),
                    "used_docs": data.get("used_docs", 0),
                    "context_len": data.get("context_len", 0),
                    "answer_length": len(data.get("answer", "")),
                    "answer": data.get("answer", "")[:500],  # 첫 500자만 저장
                    "response_time": round(end_time - start_time, 2),
                    "status": "SUCCESS",
                    "error": ""
                }
            else:
                result = {
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "question_id": i,
                    "category": test_case["category"],
                    "query": test_case["query"],
                    "expected_intent": test_case["expected_intent"],
                    "actual_intent": "ERROR",
                    "intent_match": False,
                    "needs_retrieval": False,
                    "rewritten_query": "",
                    "retrieved_count": 0,
                    "used_docs": 0,
                    "context_len": 0,
                    "answer_length": 0,
                    "answer": "",
                    "response_time": round(end_time - start_time, 2),
                    "status": f"HTTP_ERROR_{response.status_code}",
                    "error": response.text
                }
                
        except requests.RequestException as e:
            result = {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "question_id": i,
                "category": test_case["category"],
                "query": test_case["query"],
                "expected_intent": test_case["expected_intent"],
                "actual_intent": "ERROR",
                "intent_match": False,
                "needs_retrieval": False,
                "rewritten_query": "",
                "retrieved_count": 0,
                "used_docs": 0,
                "context_len": 0,
                "answer_length": 0,
                "answer": "",
                "response_time": 0,
                "status": "REQUEST_ERROR",
                "error": str(e)
            }
        
        results.append(result)
        
        # 간단한 결과 출력
        if result["status"] == "SUCCESS":
            print(f"  ✅ {result['actual_intent']} | {result['response_time']}초 | 답변길이: {result['answer_length']}")
        else:
            print(f"  ❌ {result['status']} | {result['error'][:100]}")
        
        # 서버 부하 방지를 위한 대기
        time.sleep(0.5)
    
    return results

def save_to_csv(results):
    """결과를 CSV 파일로 저장"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"chatbot_test_results_{timestamp}.csv"
    
    fieldnames = [
        "timestamp", "question_id", "category", "query", "expected_intent", 
        "actual_intent", "intent_match", "needs_retrieval", "rewritten_query",
        "retrieved_count", "used_docs", "context_len", "answer_length", 
        "answer", "response_time", "status", "error"
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n📊 테스트 결과가 '{filename}'에 저장되었습니다.")
    return filename

def print_summary(results):
    """테스트 결과 요약 출력"""
    total = len(results)
    success = len([r for r in results if r["status"] == "SUCCESS"])
    intent_matches = len([r for r in results if r["intent_match"]])
    avg_response_time = sum(r["response_time"] for r in results if r["response_time"] > 0) / max(1, len([r for r in results if r["response_time"] > 0]))
    
    print(f"\n📈 테스트 결과 요약:")
    print(f"  • 전체 질문: {total}개")
    print(f"  • 성공률: {success}/{total} ({success/total*100:.1f}%)")
    print(f"  • Intent 정확도: {intent_matches}/{total} ({intent_matches/total*100:.1f}%)")
    print(f"  • 평균 응답시간: {avg_response_time:.2f}초")
    
    # 카테고리별 성공률
    categories = {}
    for result in results:
        cat = result["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "success": 0}
        categories[cat]["total"] += 1
        if result["status"] == "SUCCESS":
            categories[cat]["success"] += 1
    
    print(f"\n📊 카테고리별 성공률:")
    for cat, stats in categories.items():
        success_rate = stats["success"] / stats["total"] * 100
        print(f"  • {cat}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")

def main():
    """메인 실행 함수"""
    print("🚀 레시피 챗봇 자동 테스트 도구")
    print("=" * 50)
    
    # 서버 연결 확인 - 수정된 부분
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=10)  # timeout 증가
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ 챗봇 서버 연결 확인 완료")
            print(f"  • 총 문서 수: {health_data.get('total_docs', 'N/A'):,}")
            print(f"  • 컬렉션: {health_data.get('collection', 'N/A')}")
            print(f"  • 모델: {health_data.get('router_model', 'N/A')}")
        else:
            print("❌ 서버 연결 실패. 챗봇이 실행 중인지 확인하세요.")
            return
    except requests.RequestException as e:
        print(f"❌ 서버 연결 오류: {e}")
        print("챗봇 서버가 http://127.0.0.1:8000 에서 실행 중인지 확인하세요.")
        return
    
    print()
    
    # 테스트 실행
    results = test_chatbot()
    
    # 결과 저장
    filename = save_to_csv(results)
    
    # 요약 출력
    print_summary(results)
    
    print(f"\n🎉 테스트 완료! 상세 결과는 '{filename}' 파일을 확인하세요.")

if __name__ == "__main__":
    main()