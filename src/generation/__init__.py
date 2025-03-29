"""
텍스트 생성 모듈

이 모듈은 RAG 파이프라인의 생성(Generation) 부분을 담당합니다.
검색된 문서를 바탕으로 OpenAI 또는 Hugging Face 모델을 통해 텍스트를 생성합니다.
"""

from .generation import generate_text, generate_text_with_retrieved_context

__all__ = ["generate_text", "generate_text_with_retrieved_context"] 