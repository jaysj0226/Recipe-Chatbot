"""Allergy detection utilities for routing/rewrite.

This module provides lightweight keyword/regex-based detection of allergy
and substitution intents, plus extraction of common allergens from text.

Notes:
- Keep strings in UTF-8. Avoid heavy NLP deps; regex only.
- Cover both Korean and English variants to increase recall.
"""

from __future__ import annotations

import re
from typing import Dict, List, Set


# Trigger phrases indicating allergy, intolerance, avoidance, or substitution intent
TRIGGER_PATTERN = re.compile(
    r"(알레르|알러지|알레르겐|알러젠|과민|민감|불내증|못\s*먹|먹지\s*못|금기|피하|제외|빼(?:고|줘)?|제거|대체|대신|치환|\bsubstitut\w*\b|\ballerg\w*\b|\bintoleran\w*\b|\bavoid\b|can'?t\s*eat|without)",
    re.IGNORECASE,
)


# Canonical allergen map -> synonyms/variants (Korean + English)
ALLERGEN_SYNONYMS: Dict[str, List[str]] = {
    # meats
    "pork": ["돼지고기", "돼지", "pork"],
    "beef": ["소고기", "소", "beef"],
    "chicken": ["닭고기", "닭", "chicken"],
    # egg, dairy, soy
    "egg": ["계란", "달걀", "egg", "eggs"],
    "milk": ["우유", "유제품", "치즈", "버터", "milk", "dairy", "cheese", "butter", "lactose"],
    "soy": ["대두", "콩", "두부", "soy", "soybean", "tofu"],
    # wheat / gluten
    "wheat_gluten": ["밀", "밀가루", "글루텐", "wheat", "flour", "gluten"],
    # nuts and seeds
    "peanut": ["땅콩", "peanut", "peanuts"],
    "tree_nut": ["견과", "아몬드", "호두", "캐슈", "피칸", "헤이즐넛", "nut", "nuts", "almond", "walnut", "cashew", "pecan", "hazelnut"],
    "sesame": ["참깨", "들깨", "깨", "sesame", "perilla"],
    # seafood
    "crustacean": ["갑각류", "새우", "게", "랍스터", "가재", "crustacean", "shrimp", "prawn", "crab", "lobster"],
    "shellfish": ["조개류", "홍합", "바지락", "조개", "굴", "전복", "가리비", "shellfish", "clam", "mussel", "oyster", "scallop"],
    "fish": ["생선", "참치", "연어", "대구", "고등어", "fish", "salmon", "tuna", "cod", "mackerel"],
    # others (examples)
    "celery": ["셀러리", "celery"],
    "mustard": ["겨자", "머스타드", "mustard"],
    "tomato": ["토마토", "tomato"],
}


def normalize(text: str) -> str:
    return (text or "").lower().strip()


def detect_triggers(text: str) -> bool:
    """Return True if the text likely expresses allergy/substitution intent."""
    if not text:
        return False
    return bool(TRIGGER_PATTERN.search(text))


def extract_allergens(text: str) -> Set[str]:
    """Extract canonical allergen keys present in the text via substring match.

    This is deliberately simple and recall-oriented. It scans for any synonym
    occurrence and returns the corresponding canonical keys.
    """
    found: Set[str] = set()
    if not text:
        return found
    t = normalize(text)
    for canon, syns in ALLERGEN_SYNONYMS.items():
        for s in syns:
            if s and normalize(s) in t:
                found.add(canon)
                break
    return found


def build_constraint_text(allergens: Set[str]) -> str:
    """Build a short Korean constraint string for rewrite prompt augmentation."""
    if not allergens:
        return ""
    # Present canonical keys as readable tokens; mix with Korean labels where obvious
    readable = ", ".join(sorted(allergens))
    return f"제약: 알레르기/제외 대상 [{readable}] 제외, 적절한 대체재를 반영해 검색 최적화."

