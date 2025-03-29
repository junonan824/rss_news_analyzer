"""
텍스트 생성 모듈

이 모듈은 OpenAI API 또는 Hugging Face의 모델을 사용하여 텍스트를 생성합니다.
RAG(Retrieval-Augmented Generation) 파이프라인의 일부로 사용됩니다.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Union, Tuple

import openai
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# .env 파일 로드
load_dotenv()

# 로깅 설정
logger = logging.getLogger(__name__)

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

class TextGenerator:
    """텍스트 생성 클래스"""
    
    def __init__(
        self, 
        default_provider: str = "openai",
        default_model: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ):
        """
        Args:
            default_provider: 기본 제공자 ('openai' 또는 'huggingface')
            default_model: 기본 모델 이름 (None이면 제공자의 기본 모델 사용)
            max_tokens: 최대 토큰 수
            temperature: 생성 온도 (높을수록 창의적, 낮을수록 보수적)
        """
        self.default_provider = default_provider
        self.default_model = default_model or self._get_default_model(default_provider)
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # HuggingFace 모델 캐시
        self.hf_models = {}
        self.hf_tokenizers = {}
        
    def _get_default_model(self, provider: str) -> str:
        """제공자별 기본 모델 반환"""
        if provider == "openai":
            return "gpt-3.5-turbo"
        elif provider == "huggingface":
            return "meta-llama/Llama-2-7b-chat-hf"
        else:
            raise ValueError(f"Unknown provider: {provider}")
            
    def generate_text(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """텍스트 생성
        
        Args:
            prompt: 생성 프롬프트
            provider: 제공자 (openai 또는 huggingface)
            model: 모델 이름
            max_tokens: 최대 토큰 수
            temperature: 생성 온도
            **kwargs: 추가 파라미터
            
        Returns:
            생성된 텍스트
        """
        provider = provider or self.default_provider
        model = model or self.default_model
        max_tokens = max_tokens or self.max_tokens
        temperature = temperature or self.temperature
        
        if provider == "openai":
            return self._generate_with_openai(prompt, model, max_tokens, temperature, **kwargs)
        elif provider == "huggingface":
            return self._generate_with_huggingface(prompt, model, max_tokens, temperature, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {provider}")
            
    async def generate_text_async(
        self,
        prompt: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """비동기 텍스트 생성
        
        Args:
            prompt: 생성 프롬프트
            provider: 제공자 (openai 또는 huggingface)
            model: 모델 이름
            max_tokens: 최대 토큰 수
            temperature: 생성 온도
            **kwargs: 추가 파라미터
            
        Returns:
            생성된 텍스트
        """
        import asyncio
        
        # 동기 함수를 스레드 풀에서 실행
        return await asyncio.to_thread(
            self.generate_text,
            prompt=prompt,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
    
    def generate_text_with_retrieved_context(
        self, 
        query: str, 
        contexts: List[str],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """검색된 컨텍스트로 텍스트 생성 (RAG)
        
        Args:
            query: 사용자 쿼리
            contexts: 검색된 컨텍스트 목록
            provider: 제공자 (openai 또는 huggingface)
            model: 모델 이름
            max_tokens: 최대 토큰 수
            temperature: 생성 온도
            **kwargs: 추가 파라미터
            
        Returns:
            생성된 텍스트
        """
        # RAG 프롬프트 생성
        prompt = self._create_rag_prompt(query, contexts)
        
        # 텍스트 생성
        return self.generate_text(
            prompt=prompt,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
    async def generate_text_with_retrieved_context_async(
        self, 
        query: str, 
        contexts: List[str],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """비동기 RAG 텍스트 생성
        
        Args:
            query: 사용자 쿼리
            contexts: 검색된 컨텍스트 목록
            provider: 제공자 (openai 또는 huggingface)
            model: 모델 이름
            max_tokens: 최대 토큰 수
            temperature: 생성 온도
            **kwargs: 추가 파라미터
            
        Returns:
            생성된 텍스트
        """
        import asyncio
        
        # RAG 프롬프트 생성
        prompt = self._create_rag_prompt(query, contexts)
        
        # 비동기 텍스트 생성
        return await self.generate_text_async(
            prompt=prompt,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )

    def _create_rag_prompt(self, query: str, contexts: List[str]) -> str:
        """RAG 프롬프트 생성
        
        Args:
            query: 사용자 쿼리
            contexts: 컨텍스트 목록
            
        Returns:
            RAG 프롬프트
        """
        context_text = "\n\n".join([f"Document {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""You are an AI assistant providing information based on the retrieved documents.

Relevant documents:
{context_text}

Based on the above documents, answer the following question:
{query}

Instructions:
1. Provide a detailed answer based on relevant information from the documents.
2. If the documents don't contain all the necessary information, clearly state what isn't covered.
3. If the question is in Korean, respond in Korean - even if the documents are in English.
4. Keep your response friendly and natural in tone, especially in Korean.
5. If needed, cite your sources at the end of your answer (e.g., "출처: CNN 기사").
6. If information is not available in the provided documents, be honest about it.

Your answer:
"""
        
        return prompt

    def _generate_with_openai(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """OpenAI API를 사용하여 텍스트 생성
        
        Args:
            prompt: 프롬프트
            model: 사용할 OpenAI 모델
            max_tokens: 생성할 최대 토큰 수
            temperature: 온도 파라미터
            **kwargs: 추가 OpenAI 파라미터
            
        Returns:
            생성된 텍스트
        """
        # API 키 확인
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            error_msg = "OPENAI_API_KEY 환경 변수가 설정되지 않았습니다."
            logger.error(error_msg)
            return f"오류: {error_msg}"
        
        # 모델 설정
        model = model or self.default_model
        
        try:
            # 직접 requests를 사용하여 API 호출 (OpenAI 클라이언트 사용 안 함)
            import requests
            
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # 허용된 파라미터만 전달
            allowed_params = ["frequency_penalty", "presence_penalty", "stop", "top_p"]
            for param in allowed_params:
                if param in kwargs:
                    payload[param] = kwargs[param]
            
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                response_data = response.json()
                return response_data["choices"][0]["message"]["content"].strip()
            else:
                error_msg = f"OpenAI API 오류 (상태 코드: {response.status_code}): {response.text}"
                logger.error(error_msg)
                return f"오류: {error_msg}"
            
        except Exception as e:
            logger.error(f"OpenAI API 호출 중 오류 발생: {e}")
            return f"죄송합니다. API 호출 중 오류가 발생했습니다: {str(e)}"
    
    def _generate_with_huggingface(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Hugging Face 모델 사용 텍스트 생성
        
        Args:
            prompt: 프롬프트
            model: 사용할 Hugging Face 모델
            max_tokens: 생성할 최대 토큰 수
            temperature: 온도 파라미터
            **kwargs: 추가 HF 파라미터
            
        Returns:
            생성된 텍스트
        """
        # 모델 설정
        model_name = model or self.default_model
        
        try:
            # 모델과 토크나이저 로드 (캐싱)
            if model_name not in self.hf_models:
                self.hf_tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
                self.hf_models[model_name] = AutoModelForCausalLM.from_pretrained(model_name)
            
            tokenizer = self.hf_tokenizers[model_name]
            model = self.hf_models[model_name]
            
            # 텍스트 생성 파이프라인 설정
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # 텍스트 생성
            response = generator(
                prompt,
                max_length=len(tokenizer.encode(prompt)) + max_tokens,
                temperature=temperature,
                num_return_sequences=1,
                **kwargs
            )
            
            # 생성된 텍스트에서 프롬프트 제거
            generated_text = response[0]["generated_text"]
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Hugging Face 모델 호출 중 오류 발생: {e}")
            return f"오류: {str(e)}"

# 기존 함수들은 TextGenerator 클래스의 메서드를 호출하도록 수정

def generate_text(
    prompt: str,
    provider: str = "openai",
    model: Optional[str] = None,
    max_tokens: int = 1000,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """텍스트 생성 함수
    
    Args:
        prompt: 생성 프롬프트
        provider: 제공자 (openai 또는 huggingface)
        model: 모델 이름
        max_tokens: 최대 토큰 수
        temperature: 생성 온도
        **kwargs: 추가 파라미터
        
    Returns:
        생성된 텍스트
    """
    generator = TextGenerator(
        default_provider=provider,
        default_model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return generator.generate_text(prompt, **kwargs)

def generate_text_with_retrieved_context(
    query: str, 
    contexts: List[str],
    provider: str = "openai",
    model: Optional[str] = None,
    max_tokens: int = 1000, 
    temperature: float = 0.7,
    **kwargs
) -> str:
    """검색된 컨텍스트로 텍스트 생성 (RAG)
    
    Args:
        query: 사용자 쿼리
        contexts: 검색된 컨텍스트 목록
        provider: 제공자 (openai 또는 huggingface)
        model: 모델 이름
        max_tokens: 최대 토큰 수
        temperature: 생성 온도
        **kwargs: 추가 파라미터
        
    Returns:
        생성된 텍스트
    """
    generator = TextGenerator(
        default_provider=provider,
        default_model=model,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return generator.generate_text_with_retrieved_context(
        query=query,
        contexts=contexts,
        **kwargs
    )

# 헬퍼 함수들
def _create_rag_prompt(query: str, contexts: List[str]) -> str:
    """RAG 프롬프트 생성
    
    Args:
        query: 사용자 쿼리
        contexts: 컨텍스트 목록
        
    Returns:
        RAG 프롬프트
    """
    context_text = "\n\n".join([f"Document {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
    
    prompt = f"""You are an AI assistant providing information based on the retrieved documents.

Relevant documents:
{context_text}

Based on the above documents, answer the following question:
{query}

Instructions:
1. Provide a detailed answer based on relevant information from the documents.
2. If the documents don't contain all the necessary information, clearly state what isn't covered.
3. If the question is in Korean, respond in Korean - even if the documents are in English.
4. Keep your response friendly and natural in tone, especially in Korean.
5. If needed, cite your sources at the end of your answer (e.g., "출처: CNN 기사").
6. If information is not available in the provided documents, be honest about it.

Your answer:
"""
    
    return prompt

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
    
    try:
        # 직접 requests를 사용하여 API 요청 (OpenAI 클라이언트 생성 없이)
        import requests
        import json
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"].strip()
        else:
            error_msg = f"API 오류 (상태 코드: {response.status_code}): {response.text}"
            logger.error(error_msg)
            return f"오류: {error_msg}"
            
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
        # 텍스트 생성 파이프라인 설정 - 기본 인자만 사용
        generator = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name
        )
        
        # 텍스트 생성 - 기본 인자만 사용
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