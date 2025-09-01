🍳 Recipe Chatbot (RAG 개선) & 그룹 A/B/C 실험

요리 도메인의 RAG(🔎 Retrieval-Augmented Generation) 실험·개발 레포입니다.
기존 CRAG 워크플로우를 도메인 OOD 게이트 → 유사도 필터 → Clarify → 웹검색 Fallback으로 개선했고, A/B/C 그룹 비교 실험(Baseline/Rewrite/Refuse) 러너와 Group A 베이스라인 서버를 제공합니다.

<p align="left"> <a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue.svg"></a> <a href="#"><img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-ready-success.svg"></a> <a href="#"><img alt="LangChain" src="https://img.shields.io/badge/LangChain-0.2%2B-0A7BBB.svg"></a> <a href="#"><img alt="ChromaDB" src="https://img.shields.io/badge/Chroma-DB-orange.svg"></a> <a href="#"><img alt="License" src="https://img.shields.io/badge/License-MIT-lightgrey.svg"></a> </p>
🗂️ 목차

핵심 기능

레포 구조

요구사항 & 설치

환경변수

빠른 시작

A/B/C 실험 실행

평가 지표

개선된 CRAG 워크플로우

실험 결과(요약)

재현 팁

트러블슈팅

로드맵

기여

라이선스

참고 자료

📌 핵심 기능

도메인 OOD 게이트(요리/레시피/영양): 비요리 질의는 초기에 차단(in/out 분기).

유사도 필터링(FILTER_SCORE): notSure일 때 Top-K 중 임계값 미달 청크 제거.

Clarify 노드: 컨텍스트가 비면 사용자 재질문(ask) 또는 자동 재작성(auto).

웹검색 Fallback: notGrounded면 최후에 웹 검색으로 보강.

Group A 베이스라인: 벡터DB Top-K → LLM 답변(문서 없으면 절대 생성 금지).

A/B/C 실험 러너: 동일 질의셋으로 세 그룹 비교(+ LLM Judge).
