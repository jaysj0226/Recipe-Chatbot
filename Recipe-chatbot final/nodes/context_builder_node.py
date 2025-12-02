"""Context Builder Node - Build Context from Retrieved Docs"""
from typing import List, Tuple

from utils.text_formatter import format_markdown_content


def build_context_node(docs: List[str], max_docs: int = 5, max_length: int = 6000) -> str:
    """
    Context Builder Node: 검색된 문서들을 컨텍스트로 구성
    
    Args:
        docs: 검색된 문서 리스트
        max_docs: 최대 문서 수
        max_length: 최대 컨텍스트 길이
        
    Returns:
        str: 포맷팅된 컨텍스트 문자열
    """
    if not docs:
        return ""
    
    contexts = []
    seen_content = set()
    
    for content in docs:
        # 짧은 문서 필터링 (20자 이상 허용)
        if not content or len(content) < 20:
            continue
        
        # 중복 제거 (앞 200자 기준 해싱)
        content_hash = hash(content[:200])
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)
        
        # 마크다운 포맷팅
        formatted = format_markdown_content(content)
        contexts.append(formatted)
        
        # 최대 문서 수 제한
        if len(contexts) >= max_docs:
            break
    
    # 구분자로 연결 후 길이 제한
    context_text = "\n\n---\n\n".join(contexts)
    return context_text[:max_length]


def build_context_with_images(
    docs: List[str],
    images: List[str] | None = None,
    max_docs: int = 5,
    max_length: int = 6000
) -> Tuple[str, List[str], List[str]]:
    """
    Build context text and select aligned images for the docs that are actually
    used in the final context. This keeps image URLs consistent with the text
    grounding used for answer generation.

    Args:
        docs: Retrieved document texts (ordered by relevance).
        images: Optional list of image URLs aligned index-wise with docs.
        max_docs: Maximum number of documents to include in context.
        max_length: Max context length after concatenation.

    Returns:
        (context_text, selected_image_urls, selected_doc_texts)
    """
    if not docs:
        return "", [], []

    contexts: List[str] = []
    selected_images: List[str] = []
    selected_docs: List[str] = []
    seen_content = set()
    images = images or []

    for idx, content in enumerate(docs):
        # ✅ 최소 길이를 20자로 완화하여 더 많은 레시피 포함 (제목만 있는 짧은 레시피도 허용)
        if not content or len(content) < 20:
            continue

        content_hash = hash(content[:200])
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)

        formatted = format_markdown_content(content)
        contexts.append(formatted)
        selected_docs.append(content)

        # pick aligned image if present and looks valid
        url = images[idx] if idx < len(images) else ""
        if isinstance(url, str) and url.startswith("http"):
            selected_images.append(url)

        if len(contexts) >= max_docs:
            break

    context_text = "\n\n---\n\n".join(contexts)
    return context_text[:max_length], selected_images, selected_docs
