"""RAGAS 평가 스크립트 (clarify / OOD 제외 버전, 컨텍스트 정렬)."""

import sys
import io
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# Windows 콘솔 인코딩 이슈 해결
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dotenv import load_dotenv
from datasets import Dataset

sys.path.append(str(Path(__file__).parent))

load_dotenv()

try:
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall, answer_relevancy, faithfulness

    print("OK RAGAS 라이브러리 로드 성공")
except ImportError:
    print("ERROR RAGAS 패키지 필요: pip install ragas")
    raise

from services.pipeline import run_pipeline
from config.schemas import AskRequest


# 1. 테스트 케이스 (확장 50개, 한글 정상 인코딩)
TEST_CASES: List[Dict] = [
    # === 한식 찌개/국 (10개) ===
    {
        "id": 1,
        "query": "김치찌개 레시피 알려줘",
        "intent": "recipe",
        "category": "한식_찌개",
        "difficulty": "easy",
        "expected_keywords": ["김치", "돼지고기", "물", "조리", "끓이기"],
        "ground_truth": "김치찌개는 김치와 돼지고기를 볶은 뒤 물을 붓고 끓여 만드는 찌개입니다.",
    },
    {
        "id": 2,
        "query": "된장찌개 만드는 법",
        "intent": "recipe",
        "category": "한식_찌개",
        "difficulty": "easy",
        "expected_keywords": ["된장", "멸치육수", "채소", "끓이기"],
        "ground_truth": "된장찌개는 멸치육수에 된장을 풀고 감자, 애호박, 두부 등을 넣고 끓여 만듭니다.",
    },
    {
        "id": 3,
        "query": "순두부찌개 어떻게 만들어?",
        "intent": "recipe",
        "category": "한식_찌개",
        "difficulty": "easy",
        "expected_keywords": ["순두부", "고춧가루", "계란", "조리"],
        "ground_truth": "순두부찌개는 고춧가루를 기름에 볶아 양념을 만든 뒤, 육수와 순두부를 넣고 끓인 후 계란을 넣어 마무리합니다.",
    },
    {
        "id": 4,
        "query": "부대찌개 레시피",
        "intent": "recipe",
        "category": "한식_찌개",
        "difficulty": "medium",
        "expected_keywords": ["부대찌개", "라면", "햄", "소시지"],
        "ground_truth": "부대찌개는 햄과 소시지, 김치를 육수에 넣고 라면사리를 추가해 끓이는 찌개입니다.",
    },
    {
        "id": 5,
        "query": "미역국 끓이는 방법",
        "intent": "recipe",
        "category": "한식_국",
        "difficulty": "easy",
        "expected_keywords": ["미역", "소고기", "참기름", "국"],
        "ground_truth": "미역국은 불린 미역을 참기름에 볶다가 물을 넣고 소고기와 함께 끓여 만듭니다.",
    },
    {
        "id": 6,
        "query": "육개장 만들기",
        "intent": "recipe",
        "category": "한식_국",
        "difficulty": "hard",
        "expected_keywords": ["소고기", "대파", "고사리", "매운국"],
        "ground_truth": "육개장은 소고기를 삶아 찢은 뒤 고춧가루로 양념하고 대파, 고사리, 숙주 등과 함께 끓여 만드는 매운 국입니다.",
    },
    {
        "id": 7,
        "query": "감자탕 레시피 알려줘",
        "intent": "recipe",
        "category": "한식_찌개",
        "difficulty": "hard",
        "expected_keywords": ["돼지등뼈", "감자", "우거지", "얼큰"],
        "ground_truth": "감자탕은 돼지등뼈를 삶아 고춧가루 양념을 하고 감자, 우거지를 넣어 푹 끓인 얼큰한 국물 요리입니다.",
    },
    {
        "id": 8,
        "query": "콩나물국 끓이는 법",
        "intent": "recipe",
        "category": "한식_국",
        "difficulty": "easy",
        "expected_keywords": ["콩나물", "멸치육수", "간단"],
        "ground_truth": "콩나물국은 멸치육수에 콩나물을 넣고 끓이다가 마늘과 대파를 넣어 간단히 끓여 만듭니다.",
    },
    {
        "id": 9,
        "query": "떡국 만드는 방법",
        "intent": "recipe",
        "category": "한식_국",
        "difficulty": "medium",
        "expected_keywords": ["떡국떡", "육수", "계란지단"],
        "ground_truth": "떡국은 사골이나 멸치육수에 떡국떡을 넣고 끓인 뒤 계란지단과 김으로 고명을 얹어 만듭니다.",
    },
    {
        "id": 10,
        "query": "동태찌개 레시피",
        "intent": "recipe",
        "category": "한식_찌개",
        "difficulty": "medium",
        "expected_keywords": ["동태", "무", "고춧가루", "얼큰"],
        "ground_truth": "동태찌개는 동태와 무를 넣고 고춧가루로 양념해 얼큰하게 끓인 생선 찌개입니다.",
    },

    # === 한식 구이/볶음 (10개) ===
    {
        "id": 11,
        "query": "불고기 레시피",
        "intent": "recipe",
        "category": "한식_구이",
        "difficulty": "medium",
        "expected_keywords": ["소고기", "간장양념", "굽기"],
        "ground_truth": "불고기는 간장, 설탕, 다진 마늘, 참기름 등으로 만든 양념에 소고기를 재운 뒤 볶거나 구워 만드는 요리입니다.",
    },
    {
        "id": 12,
        "query": "제육볶음 만들기",
        "intent": "recipe",
        "category": "한식_볶음",
        "difficulty": "medium",
        "expected_keywords": ["돼지고기", "고추장", "볶기"],
        "ground_truth": "제육볶음은 돼지고기에 고추장 양념을 더해 채소와 함께 볶아 만드는 매운 볶음 요리입니다.",
    },
    {
        "id": 13,
        "query": "삼겹살 굽는 방법",
        "intent": "recipe",
        "category": "한식_구이",
        "difficulty": "easy",
        "expected_keywords": ["삼겹살", "구이", "팬"],
        "ground_truth": "삼겹살은 팬이나 석쇠에 올려 중불에서 기름을 빼가며 노릇하게 구워 먹습니다.",
    },
    {
        "id": 14,
        "query": "닭갈비 만드는 법",
        "intent": "recipe",
        "category": "한식_볶음",
        "difficulty": "medium",
        "expected_keywords": ["닭고기", "고추장양념", "볶기"],
        "ground_truth": "닭갈비는 닭고기를 고추장 양념에 재운 뒤 양배추, 고구마, 떡과 함께 볶아 만듭니다.",
    },
    {
        "id": 15,
        "query": "오징어볶음 레시피",
        "intent": "recipe",
        "category": "한식_볶음",
        "difficulty": "medium",
        "expected_keywords": ["오징어", "고추장", "채소"],
        "ground_truth": "오징어볶음은 손질한 오징어를 고추장 양념과 채소와 함께 센 불에서 빠르게 볶아 만듭니다.",
    },
    {
        "id": 16,
        "query": "LA갈비 구이 방법",
        "intent": "recipe",
        "category": "한식_구이",
        "difficulty": "medium",
        "expected_keywords": ["LA갈비", "간장양념", "그릴"],
        "ground_truth": "LA갈비는 간장 양념에 재운 뒤 그릴이나 팬에서 노릇하게 구워 만드는 소갈비 요리입니다.",
    },
    {
        "id": 17,
        "query": "두부김치 만들기",
        "intent": "recipe",
        "category": "한식_볶음",
        "difficulty": "easy",
        "expected_keywords": ["두부", "김치", "돼지고기"],
        "ground_truth": "두부김치는 두부를 부쳐내고 김치와 돼지고기를 볶아 함께 곁들여 먹는 요리입니다.",
    },
    {
        "id": 18,
        "query": "낙지볶음 레시피 알려줘",
        "intent": "recipe",
        "category": "한식_볶음",
        "difficulty": "hard",
        "expected_keywords": ["낙지", "고추장", "매운"],
        "ground_truth": "낙지볶음은 손질한 낙지를 고추장 양념에 버무려 파와 함께 센 불에서 빠르게 볶아 만드는 매운 요리입니다.",
    },
    {
        "id": 19,
        "query": "고등어구이 하는 법",
        "intent": "recipe",
        "category": "한식_구이",
        "difficulty": "easy",
        "expected_keywords": ["고등어", "구이", "소금"],
        "ground_truth": "고등어구이는 손질한 고등어에 소금을 뿌려 팬이나 그릴에서 노릇하게 구워 만듭니다.",
    },
    {
        "id": 20,
        "query": "돼지고기 김치볶음 만들기",
        "intent": "recipe",
        "category": "한식_볶음",
        "difficulty": "easy",
        "expected_keywords": ["돼지고기", "김치", "볶음"],
        "ground_truth": "돼지고기 김치볶음은 돼지고기와 김치를 함께 볶아 간단히 만드는 볶음 요리입니다.",
    },

    # === 한식 밥/면 (10개) ===
    {
        "id": 21,
        "query": "비빔밥 레시피 알려줘",
        "intent": "recipe",
        "category": "한식_밥",
        "difficulty": "medium",
        "expected_keywords": ["밥", "나물", "고추장", "비비기"],
        "ground_truth": "비빔밥은 밥 위에 나물과 고기, 계란을 올리고 고추장을 넣어 비벼 먹는 한식 요리입니다.",
    },
    {
        "id": 22,
        "query": "간단한 계란볶음밥 만드는 법",
        "intent": "recipe",
        "category": "한식_밥",
        "difficulty": "easy",
        "expected_keywords": ["밥", "계란", "간단", "볶음밥"],
        "ground_truth": "계란볶음밥은 기름에 계란을 먼저 볶은 뒤 밥을 넣어 함께 볶고 간을 맞춰 만드는 간단한 볶음밥입니다.",
    },
    {
        "id": 23,
        "query": "김치볶음밥 레시피",
        "intent": "recipe",
        "category": "한식_밥",
        "difficulty": "easy",
        "expected_keywords": ["김치", "밥", "볶음밥"],
        "ground_truth": "김치볶음밥은 잘게 썬 김치와 밥을 함께 볶아 간단히 만드는 볶음밥입니다.",
    },
    {
        "id": 24,
        "query": "잔치국수 끓이는 법",
        "intent": "recipe",
        "category": "한식_면",
        "difficulty": "easy",
        "expected_keywords": ["국수", "멸치육수", "고명"],
        "ground_truth": "잔치국수는 멸치육수에 삶은 국수를 넣고 김, 계란지단, 김치를 고명으로 얹어 만듭니다.",
    },
    {
        "id": 25,
        "query": "비빔국수 만드는 방법",
        "intent": "recipe",
        "category": "한식_면",
        "difficulty": "easy",
        "expected_keywords": ["국수", "고추장양념", "채소"],
        "ground_truth": "비빔국수는 삶은 국수에 고추장 양념을 넣고 채소와 함께 비벼 먹는 국수 요리입니다.",
    },
    {
        "id": 26,
        "query": "냉면 만들기",
        "intent": "recipe",
        "category": "한식_면",
        "difficulty": "medium",
        "expected_keywords": ["냉면", "육수", "시원"],
        "ground_truth": "냉면은 메밀면을 삶아 차가운 육수에 담고 고명을 얹어 시원하게 먹는 여름 국수 요리입니다.",
    },
    {
        "id": 27,
        "query": "볶음밥 맛있게 만드는 법",
        "intent": "recipe",
        "category": "한식_밥",
        "difficulty": "easy",
        "expected_keywords": ["볶음밥", "밥", "센불"],
        "ground_truth": "볶음밥은 센 불에서 밥과 재료를 빠르게 볶아 고슬고슬하게 만드는 것이 핵심입니다.",
    },
    {
        "id": 28,
        "query": "주먹밥 만들기",
        "intent": "recipe",
        "category": "한식_밥",
        "difficulty": "easy",
        "expected_keywords": ["주먹밥", "밥", "간단"],
        "ground_truth": "주먹밥은 밥에 소금이나 참기름으로 간을 하고 속재료를 넣어 손으로 동그랗게 만듭니다.",
    },
    {
        "id": 29,
        "query": "콩나물밥 짓는 법",
        "intent": "recipe",
        "category": "한식_밥",
        "difficulty": "easy",
        "expected_keywords": ["콩나물", "밥", "양념장"],
        "ground_truth": "콩나물밥은 밥을 지을 때 콩나물을 얹어 함께 짓고 양념장을 곁들여 비벼 먹습니다.",
    },
    {
        "id": 30,
        "query": "칼국수 끓이는 방법",
        "intent": "recipe",
        "category": "한식_면",
        "difficulty": "medium",
        "expected_keywords": ["칼국수", "육수", "면"],
        "ground_truth": "칼국수는 멸치 육수에 밀가루로 만든 칼국수 면을 넣고 채소와 함께 끓여 만듭니다.",
    },

    # === 한식 반찬/찜 (10개) ===
    {
        "id": 31,
        "query": "계란말이 만드는 방법",
        "intent": "recipe",
        "category": "한식_반찬",
        "difficulty": "easy",
        "expected_keywords": ["계란", "지단", "말기"],
        "ground_truth": "계란말이는 풀어 섞은 계란을 얇게 부친 뒤 여러 번 말아 익혀 만드는 반찬입니다.",
    },
    {
        "id": 32,
        "query": "무생채 만들기",
        "intent": "recipe",
        "category": "한식_반찬",
        "difficulty": "easy",
        "expected_keywords": ["무", "고춧가루", "새콤"],
        "ground_truth": "무생채는 무를 채 썰어 고춧가루, 식초, 설탕으로 새콤달콤하게 무친 반찬입니다.",
    },
    {
        "id": 33,
        "query": "시금치나물 무치는 법",
        "intent": "recipe",
        "category": "한식_반찬",
        "difficulty": "easy",
        "expected_keywords": ["시금치", "참기름", "나물"],
        "ground_truth": "시금치나물은 데친 시금치를 참기름, 마늘, 소금으로 간단히 무쳐 만듭니다.",
    },
    {
        "id": 34,
        "query": "잡채 만드는 방법",
        "intent": "recipe",
        "category": "한식_반찬",
        "difficulty": "hard",
        "expected_keywords": ["당면", "채소", "볶음"],
        "ground_truth": "잡채는 당면과 여러 채소, 고기를 각각 볶아 간장 양념으로 버무려 만드는 볶음 요리입니다.",
    },
    {
        "id": 35,
        "query": "감자조림 레시피",
        "intent": "recipe",
        "category": "한식_반찬",
        "difficulty": "easy",
        "expected_keywords": ["감자", "간장", "조림"],
        "ground_truth": "감자조림은 감자를 간장, 설탕, 물엿으로 달콤짭짤하게 조려 만드는 반찬입니다.",
    },
    {
        "id": 36,
        "query": "멸치볶음 만들기",
        "intent": "recipe",
        "category": "한식_반찬",
        "difficulty": "easy",
        "expected_keywords": ["멸치", "볶음", "견과류"],
        "ground_truth": "멸치볶음은 멸치를 볶아 간장과 설탕으로 간하고 견과류를 넣어 만드는 반찬입니다.",
    },
    {
        "id": 37,
        "query": "갈비찜 레시피 알려줘",
        "intent": "recipe",
        "category": "한식_찜",
        "difficulty": "hard",
        "expected_keywords": ["갈비", "간장양념", "찜"],
        "ground_truth": "갈비찜은 소갈비를 간장 양념에 재워 채소와 함께 푹 찜질해 부드럽게 만드는 요리입니다.",
    },
    {
        "id": 38,
        "query": "계란찜 만드는 법",
        "intent": "recipe",
        "category": "한식_찜",
        "difficulty": "easy",
        "expected_keywords": ["계란", "찜", "부드러운"],
        "ground_truth": "계란찜은 계란에 물이나 육수를 넣고 저어 찜기에 쪄 부드럽게 만듭니다.",
    },
    {
        "id": 39,
        "query": "콩나물무침 만들기",
        "intent": "recipe",
        "category": "한식_반찬",
        "difficulty": "easy",
        "expected_keywords": ["콩나물", "무침", "참기름"],
        "ground_truth": "콩나물무침은 데친 콩나물에 참기름, 마늘, 소금을 넣어 무친 반찬입니다.",
    },
    {
        "id": 40,
        "query": "어묵볶음 레시피",
        "intent": "recipe",
        "category": "한식_반찬",
        "difficulty": "easy",
        "expected_keywords": ["어묵", "볶음", "간장"],
        "ground_truth": "어묵볶음은 어묵을 채 썰어 간장과 설탕으로 간단히 볶아 만드는 반찬입니다.",
    },

    # === 글로벌 요리 (10개) ===
    {
        "id": 41,
        "query": "카레라이스 어떻게 만들어?",
        "intent": "recipe",
        "category": "글로벌_요리",
        "difficulty": "easy",
        "expected_keywords": ["카레", "밥", "감자", "당근"],
        "ground_truth": "카레라이스는 감자와 당근, 양파, 고기를 볶은 뒤 물과 카레 블록을 넣고 끓여 밥과 함께 내는 요리입니다.",
    },
    {
        "id": 42,
        "query": "스파게티 만드는 법",
        "intent": "recipe",
        "category": "글로벌_요리",
        "difficulty": "medium",
        "expected_keywords": ["파스타", "토마토소스", "면"],
        "ground_truth": "스파게티는 삶은 면에 토마토소스나 크림소스를 넣고 버무려 만드는 이탈리아 파스타 요리입니다.",
    },
    {
        "id": 43,
        "query": "크림파스타 레시피",
        "intent": "recipe",
        "category": "글로벌_요리",
        "difficulty": "medium",
        "expected_keywords": ["파스타", "크림소스", "우유"],
        "ground_truth": "크림파스타는 삶은 면에 생크림이나 우유로 만든 크림소스를 넣어 만드는 파스타입니다.",
    },
    {
        "id": 44,
        "query": "마라탕 만들기",
        "intent": "recipe",
        "category": "글로벌_요리",
        "difficulty": "hard",
        "expected_keywords": ["마라", "중국요리", "얼얼"],
        "ground_truth": "마라탕은 마라 양념과 육수에 채소, 고기, 면을 넣고 끓여 얼얼하게 먹는 중국식 훠궈입니다.",
    },
    {
        "id": 45,
        "query": "짜장면 만드는 방법",
        "intent": "recipe",
        "category": "글로벌_요리",
        "difficulty": "medium",
        "expected_keywords": ["짜장", "춘장", "중국집"],
        "ground_truth": "짜장면은 춘장을 기름에 볶아 양파와 고기를 넣고 삶은 면에 얹어 만드는 중화 요리입니다.",
    },
    {
        "id": 46,
        "query": "짬뽕 레시피 알려줘",
        "intent": "recipe",
        "category": "글로벌_요리",
        "difficulty": "hard",
        "expected_keywords": ["짬뽕", "해물", "매운국물"],
        "ground_truth": "짬뽕은 해물과 채소를 볶다가 육수를 넣고 고춧가루로 얼큰하게 끓여 면과 함께 먹는 요리입니다.",
    },
    {
        "id": 47,
        "query": "오므라이스 만들기",
        "intent": "recipe",
        "category": "글로벌_요리",
        "difficulty": "medium",
        "expected_keywords": ["오므라이스", "계란", "볶음밥"],
        "ground_truth": "오므라이스는 볶음밥을 지단으로 감싸고 케챱을 뿌려 만드는 일본식 양식 요리입니다.",
    },
    {
        "id": 48,
        "query": "돈까스 만드는 법",
        "intent": "recipe",
        "category": "글로벌_요리",
        "difficulty": "medium",
        "expected_keywords": ["돈까스", "튀김", "소스"],
        "ground_truth": "돈까스는 돼지고기에 빵가루를 묻혀 튀긴 뒤 소스를 곁들여 먹는 일본식 튀김 요리입니다.",
    },
    {
        "id": 49,
        "query": "탕수육 레시피",
        "intent": "recipe",
        "category": "글로벌_요리",
        "difficulty": "hard",
        "expected_keywords": ["탕수육", "튀김", "소스"],
        "ground_truth": "탕수육은 돼지고기를 튀긴 뒤 새콤달콤한 소스를 부어 만드는 중화 요리입니다.",
    },
    {
        "id": 50,
        "query": "20분 안에 만들 수 있는 쉬운 저녁 메뉴",
        "intent": "recipe",
        "category": "시간_제약",
        "difficulty": "easy",
        "expected_keywords": ["20분", "간단", "저녁", "빠른 요리"],
        "ground_truth": "20분 안에 만들 수 있는 저녁 메뉴로는 계란볶음밥, 간단한 파스타, 김치찌개 등이 있습니다.",
    },

    # === 추가 한식 찌개/국 (10개) ===
    {
        "id": 51,
        "query": "북어국 끓이는 법",
        "intent": "recipe",
        "category": "한식_국",
        "difficulty": "medium",
        "expected_keywords": ["북어", "국", "해장"],
        "ground_truth": "북어국은 북어를 물에 불려 육수에 넣고 끓인 뒤 파와 마늘로 간을 하는 해장국입니다.",
    },
    {
        "id": 52,
        "query": "아욱국 만들기",
        "intent": "recipe",
        "category": "한식_국",
        "difficulty": "easy",
        "expected_keywords": ["아욱", "된장", "국"],
        "ground_truth": "아욱국은 된장을 푼 육수에 아욱을 넣고 끓여 만드는 간단한 국입니다.",
    },
    {
        "id": 53,
        "query": "무국 끓이는 방법",
        "intent": "recipe",
        "category": "한식_국",
        "difficulty": "easy",
        "expected_keywords": ["무", "국", "멸치육수"],
        "ground_truth": "무국은 멸치육수에 무를 넣고 끓인 뒤 소금으로 간을 맞춰 만듭니다.",
    },
    {
        "id": 54,
        "query": "해물탕 레시피",
        "intent": "recipe",
        "category": "한식_찌개",
        "difficulty": "hard",
        "expected_keywords": ["해물", "탕", "매운"],
        "ground_truth": "해물탕은 여러 해산물을 넣고 고추장이나 고춧가루로 양념해 얼큰하게 끓인 탕입니다.",
    },
    {
        "id": 55,
        "query": "김치국 끓이기",
        "intent": "recipe",
        "category": "한식_국",
        "difficulty": "easy",
        "expected_keywords": ["김치", "국", "돼지고기"],
        "ground_truth": "김치국은 김치와 돼지고기를 넣고 멸치육수에 끓여 만드는 국입니다.",
    },
    {
        "id": 56,
        "query": "청국장찌개 만드는 법",
        "intent": "recipe",
        "category": "한식_찌개",
        "difficulty": "medium",
        "expected_keywords": ["청국장", "찌개", "두부"],
        "ground_truth": "청국장찌개는 청국장을 풀고 두부와 채소를 넣어 끓인 된장찌개의 일종입니다.",
    },
    {
        "id": 57,
        "query": "된장국 끓이는 법",
        "intent": "recipe",
        "category": "한식_국",
        "difficulty": "easy",
        "expected_keywords": ["된장", "국", "멸치육수"],
        "ground_truth": "된장국은 멸치육수에 된장을 풀고 두부나 호박을 넣어 끓인 국입니다.",
    },
    {
        "id": 58,
        "query": "시래기국 레시피",
        "intent": "recipe",
        "category": "한식_국",
        "difficulty": "medium",
        "expected_keywords": ["시래기", "국", "된장"],
        "ground_truth": "시래기국은 불린 시래기를 된장 육수에 넣고 끓여 구수하게 만든 국입니다.",
    },
    {
        "id": 59,
        "query": "조개탕 만들기",
        "intent": "recipe",
        "category": "한식_찌개",
        "difficulty": "medium",
        "expected_keywords": ["조개", "탕", "시원"],
        "ground_truth": "조개탕은 조개를 육수에 넣고 끓인 뒤 파와 마늘로 간을 한 시원한 국물 요리입니다.",
    },
    {
        "id": 60,
        "query": "황태국 끓이는 법",
        "intent": "recipe",
        "category": "한식_국",
        "difficulty": "medium",
        "expected_keywords": ["황태", "국", "해장"],
        "ground_truth": "황태국은 황태를 찢어 육수에 넣고 끓인 뒤 계란을 풀어 만드는 해장국입니다.",
    },

    # === 추가 한식 구이/볶음 (10개) ===
    {
        "id": 61,
        "query": "새우볶음 만드는 법",
        "intent": "recipe",
        "category": "한식_볶음",
        "difficulty": "medium",
        "expected_keywords": ["새우", "볶음", "채소"],
        "ground_truth": "새우볶음은 손질한 새우를 채소와 함께 센 불에서 빠르게 볶아 만듭니다.",
    },
    {
        "id": 62,
        "query": "버섯볶음 레시피",
        "intent": "recipe",
        "category": "한식_볶음",
        "difficulty": "easy",
        "expected_keywords": ["버섯", "볶음", "간장"],
        "ground_truth": "버섯볶음은 여러 종류의 버섯을 간장과 마늘로 볶아 만드는 반찬입니다.",
    },
    {
        "id": 63,
        "query": "고추장삼겹살 만들기",
        "intent": "recipe",
        "category": "한식_구이",
        "difficulty": "medium",
        "expected_keywords": ["삼겹살", "고추장", "구이"],
        "ground_truth": "고추장삼겹살은 삼겹살을 고추장 양념에 재워 구워 먹는 요리입니다.",
    },
    {
        "id": 64,
        "query": "연어구이 하는 법",
        "intent": "recipe",
        "category": "한식_구이",
        "difficulty": "easy",
        "expected_keywords": ["연어", "구이", "소금"],
        "ground_truth": "연어구이는 연어에 소금을 뿌려 팬이나 오븐에서 구워 만듭니다.",
    },
    {
        "id": 65,
        "query": "쭈꾸미볶음 레시피",
        "intent": "recipe",
        "category": "한식_볶음",
        "difficulty": "medium",
        "expected_keywords": ["쭈꾸미", "고추장", "볶음"],
        "ground_truth": "쭈꾸미볶음은 쭈꾸미를 고추장 양념에 버무려 채소와 함께 볶아 만드는 매운 요리입니다.",
    },
    {
        "id": 66,
        "query": "삼치구이 만드는 법",
        "intent": "recipe",
        "category": "한식_구이",
        "difficulty": "easy",
        "expected_keywords": ["삼치", "구이", "소금"],
        "ground_truth": "삼치구이는 손질한 삼치에 소금을 뿌려 팬에서 구워 만듭니다.",
    },
    {
        "id": 67,
        "query": "가지볶음 레시피",
        "intent": "recipe",
        "category": "한식_볶음",
        "difficulty": "easy",
        "expected_keywords": ["가지", "볶음", "간장"],
        "ground_truth": "가지볶음은 가지를 썰어 간장과 참기름으로 볶아 만드는 반찬입니다.",
    },
    {
        "id": 68,
        "query": "LA갈비 양념 만들기",
        "intent": "recipe",
        "category": "한식_구이",
        "difficulty": "medium",
        "expected_keywords": ["LA갈비", "양념", "간장"],
        "ground_truth": "LA갈비 양념은 간장, 설탕, 다진 마늘, 참기름, 배즙을 섞어 만듭니다.",
    },
    {
        "id": 69,
        "query": "꽁치구이 하는 법",
        "intent": "recipe",
        "category": "한식_구이",
        "difficulty": "easy",
        "expected_keywords": ["꽁치", "구이", "소금"],
        "ground_truth": "꽁치구이는 손질한 꽁치에 소금을 뿌려 팬이나 그릴에서 구워 만듭니다.",
    },
    {
        "id": 70,
        "query": "멸치볶음 간장 양념",
        "intent": "recipe",
        "category": "한식_반찬",
        "difficulty": "easy",
        "expected_keywords": ["멸치", "간장", "볶음"],
        "ground_truth": "멸치볶음 양념은 간장, 설탕, 물엿, 참기름을 섞어 만들어 볶은 멸치에 버무립니다.",
    },

    # === 추가 한식 밥/면 (10개) ===
    {
        "id": 71,
        "query": "새우볶음밥 만들기",
        "intent": "recipe",
        "category": "한식_밥",
        "difficulty": "medium",
        "expected_keywords": ["새우", "볶음밥", "밥"],
        "ground_truth": "새우볶음밥은 새우와 채소를 볶다가 밥을 넣고 함께 볶아 만듭니다.",
    },
    {
        "id": 72,
        "query": "야채볶음밥 레시피",
        "intent": "recipe",
        "category": "한식_밥",
        "difficulty": "easy",
        "expected_keywords": ["야채", "볶음밥", "간단"],
        "ground_truth": "야채볶음밥은 여러 채소를 잘게 썰어 밥과 함께 볶아 간단히 만듭니다.",
    },
    {
        "id": 73,
        "query": "막국수 만드는 법",
        "intent": "recipe",
        "category": "한식_면",
        "difficulty": "medium",
        "expected_keywords": ["막국수", "메밀", "시원"],
        "ground_truth": "막국수는 메밀면을 삶아 차가운 양념장에 비비거나 육수에 말아 먹는 국수입니다.",
    },
    {
        "id": 74,
        "query": "쫄면 만들기",
        "intent": "recipe",
        "category": "한식_면",
        "difficulty": "easy",
        "expected_keywords": ["쫄면", "고추장", "비빔"],
        "ground_truth": "쫄면은 쫄면을 삶아 고추장 양념과 채소를 넣고 비벼 먹는 면 요리입니다.",
    },
    {
        "id": 75,
        "query": "비빔냉면 양념장",
        "intent": "recipe",
        "category": "한식_면",
        "difficulty": "medium",
        "expected_keywords": ["냉면", "비빔", "양념장"],
        "ground_truth": "비빔냉면 양념장은 고추장, 고춧가루, 식초, 설탕, 참기름을 섞어 만듭니다.",
    },
    {
        "id": 76,
        "query": "우동 만드는 법",
        "intent": "recipe",
        "category": "글로벌_요리",
        "difficulty": "easy",
        "expected_keywords": ["우동", "면", "육수"],
        "ground_truth": "우동은 우동면을 삶아 멸치육수나 가쓰오부시 육수에 담고 고명을 얹어 만듭니다.",
    },
    {
        "id": 77,
        "query": "라면 맛있게 끓이는 법",
        "intent": "recipe",
        "category": "한식_면",
        "difficulty": "easy",
        "expected_keywords": ["라면", "간단", "끓이기"],
        "ground_truth": "라면은 물을 끓인 뒤 면과 스프를 넣고 계란이나 파를 추가해 끓입니다.",
    },
    {
        "id": 78,
        "query": "덮밥 만들기",
        "intent": "recipe",
        "category": "한식_밥",
        "difficulty": "medium",
        "expected_keywords": ["덮밥", "밥", "토핑"],
        "ground_truth": "덮밥은 밥 위에 볶은 고기나 채소를 얹고 소스를 뿌려 만드는 한 그릇 요리입니다.",
    },
    {
        "id": 79,
        "query": "삼각김밥 만드는 법",
        "intent": "recipe",
        "category": "한식_밥",
        "difficulty": "easy",
        "expected_keywords": ["삼각김밥", "김", "밥"],
        "ground_truth": "삼각김밥은 밥에 속재료를 넣고 김으로 삼각형 모양으로 싸서 만듭니다.",
    },
    {
        "id": 80,
        "query": "물냉면 육수 만들기",
        "intent": "recipe",
        "category": "한식_면",
        "difficulty": "hard",
        "expected_keywords": ["냉면", "육수", "시원"],
        "ground_truth": "물냉면 육수는 사골이나 동치미 국물을 베이스로 식초와 설탕으로 간을 맞춰 만듭니다.",
    },

    # === 추가 한식 반찬/찜/조림 (10개) ===
    {
        "id": 81,
        "query": "깻잎찜 만드는 법",
        "intent": "recipe",
        "category": "한식_찜",
        "difficulty": "medium",
        "expected_keywords": ["깻잎", "찜", "양념"],
        "ground_truth": "깻잎찜은 깻잎에 양념을 켜켜이 발라 쌓은 뒤 찜통에 쪄 만듭니다.",
    },
    {
        "id": 82,
        "query": "무조림 레시피",
        "intent": "recipe",
        "category": "한식_반찬",
        "difficulty": "easy",
        "expected_keywords": ["무", "조림", "간장"],
        "ground_truth": "무조림은 무를 간장, 설탕으로 달콤짭짤하게 조려 만드는 반찬입니다.",
    },
    {
        "id": 83,
        "query": "고구마줄기볶음 만들기",
        "intent": "recipe",
        "category": "한식_반찬",
        "difficulty": "medium",
        "expected_keywords": ["고구마줄기", "볶음", "나물"],
        "ground_truth": "고구마줄기볶음은 삶은 고구마줄기를 간장과 참기름으로 볶아 만듭니다.",
    },
    {
        "id": 84,
        "query": "호박나물 무치는 법",
        "intent": "recipe",
        "category": "한식_반찬",
        "difficulty": "easy",
        "expected_keywords": ["호박", "나물", "무침"],
        "ground_truth": "호박나물은 볶은 호박을 참기름과 마늘로 무쳐 만드는 반찬입니다.",
    },
    {
        "id": 85,
        "query": "두부조림 만드는 법",
        "intent": "recipe",
        "category": "한식_반찬",
        "difficulty": "easy",
        "expected_keywords": ["두부", "조림", "간장"],
        "ground_truth": "두부조림은 부친 두부를 간장 양념으로 조려 만드는 반찬입니다.",
    },
    {
        "id": 86,
        "query": "연근조림 레시피",
        "intent": "recipe",
        "category": "한식_반찬",
        "difficulty": "medium",
        "expected_keywords": ["연근", "조림", "간장"],
        "ground_truth": "연근조림은 삶은 연근을 간장, 설탕, 물엿으로 조려 만듭니다.",
    },
    {
        "id": 87,
        "query": "오이무침 만들기",
        "intent": "recipe",
        "category": "한식_반찬",
        "difficulty": "easy",
        "expected_keywords": ["오이", "무침", "새콤"],
        "ground_truth": "오이무침은 오이를 썰어 고춧가루, 식초, 설탕으로 새콤하게 무친 반찬입니다.",
    },
    {
        "id": 88,
        "query": "장조림 만드는 법",
        "intent": "recipe",
        "category": "한식_반찬",
        "difficulty": "medium",
        "expected_keywords": ["장조림", "소고기", "간장"],
        "ground_truth": "장조림은 소고기를 간장 양념에 푹 조려 부드럽고 짭짤하게 만든 반찬입니다.",
    },
    {
        "id": 89,
        "query": "가지나물 만들기",
        "intent": "recipe",
        "category": "한식_반찬",
        "difficulty": "easy",
        "expected_keywords": ["가지", "나물", "참기름"],
        "ground_truth": "가지나물은 찐 가지를 찢어 참기름과 마늘로 무친 나물입니다.",
    },
    {
        "id": 90,
        "query": "미나리무침 레시피",
        "intent": "recipe",
        "category": "한식_반찬",
        "difficulty": "easy",
        "expected_keywords": ["미나리", "무침", "고춧가루"],
        "ground_truth": "미나리무침은 데친 미나리를 고춧가루와 참기름으로 무친 반찬입니다.",
    },

    # === 추가 글로벌 요리 & 특수 케이스 (10개) ===
    {
        "id": 91,
        "query": "토마토 파스타 만들기",
        "intent": "recipe",
        "category": "글로벌_요리",
        "difficulty": "medium",
        "expected_keywords": ["토마토", "파스타", "소스"],
        "ground_truth": "토마토 파스타는 토마토 소스를 만들어 삶은 면에 버무려 만드는 이탈리아 요리입니다.",
    },
    {
        "id": 92,
        "query": "볶음우동 레시피",
        "intent": "recipe",
        "category": "글로벌_요리",
        "difficulty": "easy",
        "expected_keywords": ["우동", "볶음", "간단"],
        "ground_truth": "볶음우동은 우동면과 채소, 고기를 함께 볶아 간장으로 간을 한 요리입니다.",
    },
    {
        "id": 93,
        "query": "샌드위치 만드는 법",
        "intent": "recipe",
        "category": "글로벌_요리",
        "difficulty": "easy",
        "expected_keywords": ["샌드위치", "빵", "간단"],
        "ground_truth": "샌드위치는 빵 사이에 채소, 고기, 치즈 등을 넣어 만드는 간편식입니다.",
    },
    {
        "id": 94,
        "query": "김밥 만드는 법",
        "intent": "recipe",
        "category": "한식_밥",
        "difficulty": "medium",
        "expected_keywords": ["김밥", "김", "밥"],
        "ground_truth": "김밥은 밥과 여러 재료를 김에 올려 말아 만드는 한식 요리입니다.",
    },
    {
        "id": 95,
        "query": "핫도그 만들기",
        "intent": "recipe",
        "category": "글로벌_요리",
        "difficulty": "easy",
        "expected_keywords": ["핫도그", "소시지", "빵"],
        "ground_truth": "핫도그는 빵에 소시지와 채소를 넣고 소스를 뿌려 만듭니다.",
    },
    {
        "id": 96,
        "query": "치킨샐러드 레시피",
        "intent": "recipe",
        "category": "글로벌_요리",
        "difficulty": "medium",
        "expected_keywords": ["치킨", "샐러드", "채소"],
        "ground_truth": "치킨샐러드는 익힌 닭고기와 여러 채소를 섞어 드레싱을 뿌려 만듭니다.",
    },
    {
        "id": 97,
        "query": "떡볶이 만드는 법",
        "intent": "recipe",
        "category": "한식_볶음",
        "difficulty": "easy",
        "expected_keywords": ["떡볶이", "고추장", "어묵"],
        "ground_truth": "떡볶이는 떡과 어묵을 고추장 양념에 볶아 만드는 한식 간식입니다.",
    },
    {
        "id": 98,
        "query": "순대 맛있게 먹는 법",
        "intent": "dish_overview",
        "category": "한식_기타",
        "difficulty": "easy",
        "expected_keywords": ["순대", "찌기", "먹는법"],
        "ground_truth": "순대는 찜통에 쪄서 소금이나 양념장에 찍어 먹거나 순대국밥으로 끓여 먹습니다.",
    },
    {
        "id": 99,
        "query": "수제버거 만들기",
        "intent": "recipe",
        "category": "글로벌_요리",
        "difficulty": "hard",
        "expected_keywords": ["버거", "패티", "빵"],
        "ground_truth": "수제버거는 패티를 구워 빵에 채소와 함께 끼워 소스를 뿌려 만듭니다.",
    },
    {
        "id": 100,
        "query": "초밥 만드는 방법",
        "intent": "recipe",
        "category": "글로벌_요리",
        "difficulty": "hard",
        "expected_keywords": ["초밥", "회", "밥"],
        "ground_truth": "초밥은 초밥용 밥을 만들어 회나 토핑을 올려 만드는 일본 요리입니다.",
    },
]


def run_rag_evaluation(test_cases: List[Dict]) -> Dataset:
    """RAG 파이프라인을 실행해 RAGAS용 데이터셋 생성."""
    questions: list[str] = []
    contexts_list: list[list[str]] = []
    answers: list[str] = []
    ground_truths: list[str] = []

    print("\n" + "=" * 80)
    print("Intent RAG 테스트 케이스 실행 시작")
    print("=" * 80)

    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        ground_truth = test_case["ground_truth"]

        print(f"\n[{i}/{len(test_cases)}] 처리 중: {query}")

        try:
            request = AskRequest(query=query, k=8, enable_rewrite=True, model="gpt-4o")
            response = run_pipeline(request)

            intent = response.get("intent", "")
            branch = response.get("branch", "")

            # Clarify / OOD 응답은 평가에서 제외
            if intent != "recipe" or str(branch).startswith("clarify"):
                print(f"   스킵: intent={intent}, branch={branch} (평가 제외)")
                continue

            answer = response.get("answer", "")

            # ✅ NEW: Use the actual context_text from pipeline response
            # This ensures we evaluate what was actually used for generation
            context_text = response.get("context_text", "")

            # RAGAS expects a list of context strings
            if context_text:
                contexts = [context_text]
            else:
                # If no context was used, indicate that clearly
                contexts = ["컨텍스트가 비어있습니다 (no_context_refusal 또는 필터링됨)"]

            questions.append(query)
            contexts_list.append(contexts)
            answers.append(answer)
            ground_truths.append(ground_truth)

            print(f"   ✅ 답변 길이: {len(answer)} chars")
            print(f"   ✅ 컨텍스트 길이: {len(context_text)} chars")
            print(f"   ✅ 사용된 문서 수: {response.get('used_docs', 0)}개")
            print(f"   ✅ intent={intent}, branch={branch}")

        except Exception as e:
            print(f"   ❌ 오류 발생: {e}")
            questions.append(query)
            contexts_list.append(["오류로 인해 검색 실패"])
            answers.append("답변 생성 실패")
            ground_truths.append(ground_truth)

    dataset_dict = {
        "question": questions,
        "contexts": contexts_list,
        "answer": answers,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(dataset_dict)

    print("\n" + "=" * 80)
    print(f"최종 RAGAS 데이터셋 크기: {len(dataset)}")
    print("=" * 80)

    return dataset


def evaluate_with_ragas(dataset: Dataset):
    """RAGAS 평가 실행."""
    print("\n" + "=" * 80)
    print("RAGAS 평가 시작")
    print("=" * 80)

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, answer_relevancy, faithfulness],
        llm=llm,
    )

    print("\n✅ RAGAS 평가 완료")
    return result


def save_results(result, dataset: Dataset, output_dir: str = "ragas_results"):
    """평가 결과 저장."""
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def extract_score(v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, list):
            vals = [x for x in v if isinstance(x, (int, float))]
            return float(sum(vals) / len(vals)) if vals else 0.0
        return 0.0

    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        metrics = {
            "context_precision": extract_score(df["context_precision"].mean())
            if "context_precision" in df.columns
            else 0.0,
            "context_recall": extract_score(df["context_recall"].mean())
            if "context_recall" in df.columns
            else 0.0,
            "answer_relevancy": extract_score(df["answer_relevancy"].mean())
            if "answer_relevancy" in df.columns
            else 0.0,
            "faithfulness": extract_score(df["faithfulness"].mean())
            if "faithfulness" in df.columns
            else 0.0,
        }
    else:
        metrics = {
            "context_precision": extract_score(result["context_precision"]),
            "context_recall": extract_score(result["context_recall"]),
            "answer_relevancy": extract_score(result["answer_relevancy"]),
            "faithfulness": extract_score(result["faithfulness"]),
        }

    summary = {"timestamp": ts, "metrics": metrics, "dataset_size": len(dataset)}

    with open(out_dir / f"ragas_summary_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    try:
        df = result.to_pandas()
        df.to_csv(out_dir / f"ragas_detailed_{ts}.csv", index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"⚠️ 상세 CSV 저장 실패: {e}")

    report = f"""
{'=' * 80}
RAGAS 평가 결과 요약
{'=' * 80}

평가 시각: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
평가 케이스 수: {len(dataset)}
{'=' * 80}
지표
{'=' * 80}

1. Context Precision : {metrics['context_precision']:.4f}
2. Context Recall    : {metrics['context_recall']:.4f}
3. Answer Relevancy  : {metrics['answer_relevancy']:.4f}
4. Faithfulness      : {metrics['faithfulness']:.4f}

평균 점수: {sum(metrics.values()) / 4:.4f}
"""

    with open(out_dir / f"ragas_report_{ts}.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print("✅ 결과 저장 완료:")
    print(f"  - {out_dir / f'ragas_summary_{ts}.json'}")
    print(f"  - {out_dir / f'ragas_detailed_{ts}.csv'}")
    print(f"  - {out_dir / f'ragas_report_{ts}.txt'}")


def main():
    print("\n" + "=" * 80)
    print("Intent RAG + RAGAS 평가")
    print("=" * 80)
    print(f"테스트 케이스 수: {len(TEST_CASES)}")

    dataset = run_rag_evaluation(TEST_CASES)
    result = evaluate_with_ragas(dataset)
    save_results(result, dataset)

    print("\n" + "=" * 80)
    print("✅ 모든 평가 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
