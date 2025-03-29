"""
텍스트 생성 모듈

이 모듈은 OpenAI API 또는 Hugging Face의 모델을 사용하여 텍스트를 생성합니다.
RAG(Retrieval-Augmented Generation) 파이프라인의 일부로 사용됩니다.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union

# OpenAI 관련 임포트
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Hugging Face 관련 임포트
try:
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

logger = logging.getLogger(__name__)

# 기본 설정
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
DEFAULT_HF_MODEL = "google/flan-t5-base"

def generate_text(
    prompt: str,
    provider: str = "openai",
    model: Optional[str] = None,
    max_tokens: int = 500,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """
    제공된 프롬프트를 바탕으로 텍스트를 생성합니다.

    Args:
        prompt (str): 생성을 위한 프롬프트/질문
        provider (str, optional): 사용할 제공자 ("openai" 또는 "huggingface"). 기본값은 "openai".
        model (str, optional): 사용할 특정 모델. 지정되지 않으면 기본 모델이 사용됩니다.
        max_tokens (int, optional): 생성할 최대 토큰 수. 기본값은 500.
        temperature (float, optional): 온도 파라미터 (높을수록 더 창의적). 기본값은 0.7.
        **kwargs: 추가 모델별 파라미터

    Returns:
        str: 생성된 텍스트

    Raises:
        ValueError: 제공자가 지원되지 않거나 필요한 라이브러리가 설치되지 않은 경우
    """
    if provider.lower() == "openai":
        return _generate_with_openai(prompt, model, max_tokens, temperature, **kwargs)
    elif provider.lower() == "huggingface":
        return _generate_with_huggingface(prompt, model, max_tokens, temperature, **kwargs)
    else:
        raise ValueError(f"지원되지 않는 제공자: {provider}. 'openai' 또는 'huggingface'를 사용하세요.")

def generate_text_with_retrieved_context(
    query: str,
    contexts: List[str],
    provider: str = "openai",
    model: Optional[str] = None,
    max_tokens: int = 500,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """
    검색된 컨텍스트를 바탕으로 RAG(Retrieval-Augmented Generation)를 수행합니다.

    Args:
        query (str): 사용자 질문
        contexts (List[str]): 검색된 관련 문서/컨텍스트 목록
        provider (str, optional): 사용할 제공자 ("openai" 또는 "huggingface"). 기본값은 "openai".
        model (str, optional): 사용할 특정 모델. 지정되지 않으면 기본 모델이 사용됩니다.
        max_tokens (int, optional): 생성할 최대 토큰 수. 기본값은 500.
        temperature (float, optional): 온도 파라미터 (높을수록 더 창의적). 기본값은 0.7.
        **kwargs: 추가 모델별 파라미터

    Returns:
        str: 생성된 텍스트
    """
    # 검색된 컨텍스트를 바탕으로 프롬프트 생성
    prompt = _create_rag_prompt(query, contexts)
    
    # 텍스트 생성
    return generate_text(
        prompt=prompt,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )

def _create_rag_prompt(query: str, contexts: List[str]) -> str:
    """
    검색된 컨텍스트를 바탕으로 RAG 프롬프트를 생성합니다.

    Args:
        query (str): 사용자 질문
        contexts (List[str]): 검색된 관련 문서/컨텍스트 목록

    Returns:
        str: RAG 프롬프트
    """
    context_text = "\n\n".join([f"문서 {i+1}:\n{context}" for i, context in enumerate(contexts)])
    
    return f"""다음 정보를 바탕으로 질문에 답변해주세요:

{context_text}

질문: {query}

답변:"""

def _generate_with_openai(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 500,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """
    OpenAI API를 사용하여 텍스트를 생성합니다.

    Args:
        prompt (str): 프롬프트
        model (str, optional): 사용할 OpenAI 모델
        max_tokens (int, optional): 생성할 최대 토큰 수
        temperature (float, optional): 온도 파라미터
        **kwargs: 추가 OpenAI 파라미터

    Returns:
        str: 생성된 텍스트
    """
    if not OPENAI_AVAILABLE:
        raise ImportError("OpenAI 패키지가 설치되지 않았습니다. 'pip install openai'를 실행하세요.")
    
    # API 키 확인
    api_key = os.environ.get("OPENAI_API_KEY") or kwargs.get("api_key")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경 변수를 설정하거나 api_key 파라미터를 전달해야 합니다.")
    
    # 모델 설정
    model = model or DEFAULT_OPENAI_MODEL
    
    # OpenAI 클라이언트 초기화
    client = OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            max_tokens=max_tokens,
            temperature=temperature,
            **{k: v for k, v in kwargs.items() if k not in ["api_key"]}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API 호출 중 오류 발생: {e}")
        return f"오류: {str(e)}"

def _generate_with_huggingface(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 500,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """
    Hugging Face 모델을 사용하여 텍스트를 생성합니다.

    Args:
        prompt (str): 프롬프트
        model (str, optional): 사용할 Hugging Face 모델
        max_tokens (int, optional): 생성할 최대 토큰 수
        temperature (float, optional): 온도 파라미터
        **kwargs: 추가 Hugging Face 파라미터

    Returns:
        str: 생성된 텍스트
    """
    if not HUGGINGFACE_AVAILABLE:
        raise ImportError("Transformers 패키지가 설치되지 않았습니다. 'pip install transformers torch'를 실행하세요.")
    
    # 모델 설정
    model_name = model or DEFAULT_HF_MODEL
    
    try:
        # 텍스트 생성 파이프라인 설정
        generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            **{k: v for k, v in kwargs.items() if k not in ["api_key"]}
        )
        
        # 텍스트 생성
        response = generator(
            prompt,
            max_length=len(prompt.split()) + max_tokens,
            temperature=temperature,
            num_return_sequences=1
        )
        
        # 생성된 텍스트에서 프롬프트 제거
        generated_text = response[0]["generated_text"]
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    except Exception as e:
        logger.error(f"Hugging Face 모델 호출 중 오류 발생: {e}")
        return f"오류: {str(e)}" 