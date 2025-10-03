"""RAG 시스템 테스트 스크립트"""

from rag_hybrid_app.graph.rag_flow import RAGState
from rag_hybrid_app.nodes.retrieve_node import retrieve_node

# 테스트 1: 정상 검색
print("=" * 60)
print("🧪 Test 1: 정상 검색")
print("=" * 60)

state = RAGState(query="김치찌개 만드는 법")
result = retrieve_node(state)

print(f"검색된 문서: {len(result['retrieved_docs'])}개")
if result['retrieved_docs']:
    print(f"유사도 점수: {result.get('retrieved_scores', [])}")
    print(f"\n첫 번째 문서 미리보기:")
    print(result['retrieved_docs'][0][:200] + "...")
else:
    print("⚠️ 검색된 문서 없음")

# 테스트 2: 전체 플로우
print("\n" + "=" * 60)
print("🧪 Test 2: 전체 RAG 플로우")
print("=" * 60)

from rag_hybrid_app.graph.rag_flow import rag_flow

initial_state = RAGState(query="김치찌개 레시피 알려줘")
print(f"질문: {initial_state.query}\n")

final_state = rag_flow.invoke(initial_state)

print(f"최종 답변:")
print("-" * 60)
print(final_state.get('answer', 'N/A'))
print("-" * 60)

print("\n✅ 테스트 완료!")