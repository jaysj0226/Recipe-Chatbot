"""Text Formatting Utilities"""
import re


def format_markdown_content(content: str) -> str:
    """
    마크다운 형식을 사용자 친화적으로 변환합니다.
    Args:
        content: 원본 마크다운 텍스트
    Returns:
        str: 포맷된 텍스트
    """
    # 제목 변환
    content = re.sub(r'^# (.+)$', r'[제목] \1', content, flags=re.MULTILINE)

    # 섹션 제목 변환
    content = re.sub(r'^## Ingredients$', '[재료]', content, flags=re.MULTILINE)
    content = re.sub(r'^## Steps$', '[조리]', content, flags=re.MULTILINE)

    # 메타 라인 제거
    content = re.sub(r'^Source:.*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'^Image:.*$', '', content, flags=re.MULTILINE)

    # 과도한 개행 제거
    content = re.sub(r'\n{3,}', '\n\n', content)

    return content.strip()


def clean_newlines(text: str) -> str:
    """
    과도한 개행을 정리합니다.
    Args:
        text: 원본 텍스트
    Returns:
        str: 정리된 텍스트
    """
    return re.sub(r"\n{3,}", "\n\n", text)

