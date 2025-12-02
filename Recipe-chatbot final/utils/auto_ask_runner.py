"""Auto-ask runner utilities.

Provides:
- load_questions(): load from file or fallback list
- run_once(): run questions and persist results to JSONL
- start_background_job(): spawn a background run and return metadata
- start_background_if_enabled(): run on startup when AUTO_ASK_ENABLED=1
"""
from __future__ import annotations

import os
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from config import settings
from config.schemas import AskRequest
from services.pipeline import run_pipeline


DEFAULT_QUESTIONS: List[str] = [
    # 재료 기반 검색
    "닭가슴살로 만들 수 있는 간단한 저녁 메뉴 추천해줘.",
    "두부와 버섯만으로 가능한 비건 요리 있어?",
    "토마토, 바질, 올리브오일로 만들 수 있는 파스타 알려줘.",
    "감자와 달걀로 20분 안에 만들 수 있는 요리는?",
    "소고기 다짐육 남았는데 한 그릇 요리 추천해줘.",
    "애호박, 양파, 당근으로 국이나 찌개 할 수 있을까?",
    "생연어로 오븐 없이 만들 수 있는 요리 뭐가 있어?",
    "냉동새우로 술안주 추천해줘.",
    # 식단/알레르기/영양
    "글루텐 프리 치킨요리 추천해줘.",
    "유당 불내증 있어. 크림 없이도 고소한 파스타 있을까?",
    "땅콩 알레르기 있어. 땅콩 없이 아시아풍 면요리 알려줘.",
    "600kcal 이하 저염 한 끼 식단 추천해줘.",
    "고단백 저지방 도시락 레시피 3가지 알려줘.",
    "비건 디저트 중 초콜릿 느낌 나는 레시피 있을까?",
    "키토 다이어트에 맞는 아침 식사 추천해줘.",
    # 시간/난이도/도구 제약
    "초보자도 실패 없는 15분 저녁 레시피 알려줘.",
    "에어프라이어만으로 만들 수 있는 치킨요리 있어?",
    "오븐 없이 만드는 라자냐 가능한가?",
    "조리시간 30분 이하, 재료 7개 이하인 파스타 추천.",
    "설거지 최소화 원해. 원팬 레시피 알려줘.",
    "캠핑에서 버너 하나로 만들 수 있는 국물요리 있을까?",
    "전자레인지만으로 가능한 건강한 점심 뭐가 있어?",
    # 조리 기술/방법 안내
    "소고기 스테이크 미디엄 레어로 굽는 핵심 포인트 알려줘.",
    "파스타 면 삶는 물 소금 비율이 어떻게 돼?",
    "닭다리살 수비드로 조리할 때 시간/온도 가이드 알려줘.",
    "계란 스크램블 부드럽게 만드는 방법 단계별로 설명해줘.",
    "타코용 양파 피클 빠르게 만드는 비법 있을까?",
    "김치찌개 국물 더 깊게 만드는 감칠맛 팁 알려줘.",
    # 세계 요리/테마
    "태국식 그린커리 기본 레시피 알려줘.",
    "인도식 버터치킨과 잘 어울리는 사이드 뭐가 있어?",
    "멕시코 스트리트 타코 정통 레시피 알려줘.",
    "이탈리아 정통 카르보나라(크림 없이) 만드는 법 알려줘.",
    "한식 집들이 메뉴로 상차림 추천해줘.",
    "일본식 덮밥(돈부리) 중 20분 레시피 추천해줘.",
    # 변환/치환/대체
    "간장 대신 사용할 수 있는 재료와 비율 알려줘.",
    "버터 없는 베이킹에서 오일로 치환하는 방법은?",
    "밀가루 2컵을 아몬드가루로 바꾸려면 얼마나 써야 해?",
    "4인분 레시피를 10인분으로 늘릴 때 주의점은?",
    "컵→그램 변환표(밀가루/설탕/버터) 간단히 알려줘.",
    "신선 바질 없을 때 대체 허브와 양 조절 팁 알려줘.",
    # 남은 음식/활용
    "남은 로티세리 치킨으로 10분 점심 만들기 아이디어 줘.",
    "남은 밥으로 만들 수 있는 이색 볶음밥 레시피 알려줘.",
    "삶은 파스타 면이 남았어. 마르는 거 방지 팁과 활용법?",
    "익은 아보카도 빨리 써야 해. 샐러드 말고 다른 레시피 있어?",
    # 대화/맥락/엣지/오류
    "매운거 잘 못 먹어. 이전에 추천한 레시피 덜 맵게 바꿔줘.",
    "방금 준 파스타 레시피, 버섯 알레르기 반영해서 수정해줘.",
    "그 레시피를 에어프라이어 버전으로 변환해줄래?",
    "재료가 ‘베이컨 100g?’ 정확히 몇 줄 정도야?",
    "‘코리앤더’가 고수 맞지? 고수 싫으면 뭘로 대체해?",
    "레시피에 ‘한 꼬집’이 몇 g 정도야?",
    "오늘 뭐 먹을지 모르겠어. 내 냉장고 재료로 메뉴 추천해줘: 계란, 시금치, 우유, 식빵.",
    "디저트 말고 담백한 야식 추천해줘.",
    "밀프렙으로 3일치 만들 수 있는 닭가슴살 레시피 알려줘.",
    "아이들용 순한 카레 레시피, 매운 맛 없이 부탁해.",
    "저번에 말한 타코 소스, 장보기 리스트로 정리해줘.",
    "‘ㅁㄴㅇㄹ’ 같은 오타가 있는데 혹시 의미 추론 가능해?",
    "오늘 날씨 어때? (레시피 외 질문 처리 테스트)",
    "비슷한 레시피 3개 비교해서 차이점 정리해줘.",
    "재료 가격 고려해서 1만원 이하 저녁 식단 추천해줘.",
]


def _default_questions_path() -> Path:
    # Env override or default file next to BASE_DIR
    env_path = os.getenv("AUTO_ASK_FILE")
    if env_path:
        return Path(env_path)
    return Path(settings.BASE_DIR) / "auto_questions.txt"


def load_questions(path: Optional[str | Path] = None) -> List[str]:
    p = Path(path) if path else _default_questions_path()
    if p.exists():
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
            qs = []
            for line in lines:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                qs.append(s)
            if qs:
                return qs
        except Exception:
            pass
    return list(DEFAULT_QUESTIONS)


def _default_output_path() -> Path:
    out_dir = Path(settings.BASE_DIR) / "autotest_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"qa_{ts}.jsonl"


def run_once(output_path: Optional[str | Path] = None) -> Path:
    """Run all questions once and append JSONL lines to output_path.
    Returns the Path to the output file.
    """
    out_path = Path(output_path) if output_path else _default_output_path()
    questions = load_questions()
    model_name = settings.GENERATION_MODEL

    with out_path.open("a", encoding="utf-8") as f:
        for i, q in enumerate(questions, start=1):
            payload = AskRequest(query=q, k=settings.K_DEFAULT, model=model_name)
            record = {"index": i, "question": q, "timestamp": datetime.now().isoformat()}
            try:
                resp = run_pipeline(payload)
                record.update({"response": resp})
            except Exception as e:
                record.update({"error": str(e)})
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def start_background_job(output_path: Optional[str | Path] = None) -> dict:
    """Start a background thread to run all questions once.
    Returns metadata including planned output_path and count.
    """
    planned_path = Path(output_path) if output_path else _default_output_path()
    total = len(load_questions())

    def _runner():
        try:
            run_once(planned_path)
        except Exception:
            pass

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return {
        "started": True,
        "questions": total,
        "output_path": str(planned_path),
    }


_BG_STARTED_FLAG = False


def start_background_if_enabled() -> None:
    global _BG_STARTED_FLAG
    if _BG_STARTED_FLAG:
        return
    if os.getenv("AUTO_ASK_ENABLED", "0") != "1":
        return
    _BG_STARTED_FLAG = True
    start_background_job()

