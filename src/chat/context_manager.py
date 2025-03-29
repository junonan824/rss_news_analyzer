"""
대화 컨텍스트 관리 모듈.

대화 내용을 기반으로 컨텍스트를 구성하고 관리합니다.
"""

import logging
from typing import List, Dict, Any

# 로깅 설정
logger = logging.getLogger(__name__)


class ContextManager:
    """대화 컨텍스트 관리 클래스"""
    
    def __init__(self, max_context_length: int = 5):
        """
        Args:
            max_context_length: 컨텍스트로 사용할 최대 메시지 수
        """
        self.max_context_length = max_context_length
        
    def build_prompt_with_context(self, messages, query: str) -> str:
        """이전 대화 내용을 포함한 프롬프트 생성
        
        Args:
            messages: 이전 대화 메시지 목록 (session.messages)
            query: 현재 질문
            
        Returns:
            컨텍스트가 포함된 프롬프트
        """
        # 최근 N개의 메시지만 사용
        recent_messages = messages[-self.max_context_length:] if messages and len(messages) > 0 else []
        
        # 메시지 포맷팅
        context_str = ""
        for msg in recent_messages:
            role = "User" if msg.role == "user" else "Assistant"
            content = getattr(msg, 'content', '') or ''
            context_str += f"{role}: {content}\n"
        
        # 최종 프롬프트 생성
        prompt = f"""Previous conversation:
{context_str}
User: {query}

Please respond to the user's most recent question based on the context of the conversation.
"""
        return prompt
    
    def extract_entities_from_context(self, messages) -> List[str]:
        """대화 컨텍스트에서 중요 엔티티 추출
        
        Args:
            messages: 이전 대화 메시지 목록 (session.messages)
            
        Returns:
            추출된 엔티티 목록
        """
        # 간단한 키워드 추출 구현
        entities = []
        
        if not messages:
            return entities
            
        for msg in messages:
            if hasattr(msg, 'role') and msg.role == "user" and hasattr(msg, 'content'):
                content = getattr(msg, 'content', '') or ''
                # 단어 기반 키워드 추출 (매우 기본적인 방식)
                words = content.lower().split()
                # 3글자 이상 단어만 추출 (불용어 제외 필요)
                for word in words:
                    if len(word) > 3 and word not in ['what', 'when', 'where', 'which', 'who', 'whom', 'whose']:
                        entities.append(word)
        
        # 중복 제거
        return list(set(entities))
    
    def get_relevant_context(self, messages, query: str) -> Dict[str, Any]:
        """질문과 관련된 컨텍스트 정보를 추출
        
        Args:
            messages: 이전 대화 메시지 목록 (session.messages)
            query: 현재 질문
            
        Returns:
            관련 컨텍스트 정보를 담은 딕셔너리
        """
        entities = self.extract_entities_from_context(messages)
        
        return {
            "entities": entities,
            "recent_topics": self._extract_topics(messages),
            "query_terms": query.lower().split()
        }
    
    def _extract_topics(self, messages) -> List[str]:
        """대화에서 주요 토픽 추출 (간단한 구현)"""
        topics = []
        
        if not messages:
            return topics
            
        # 마지막 3개 메시지의 키워드만 확인
        recent_msgs = messages[-3:] if len(messages) > 3 else messages
        
        for msg in recent_msgs:
            if hasattr(msg, 'role') and msg.role == "user" and hasattr(msg, 'content'):
                content = getattr(msg, 'content', '') or ''
                words = content.lower().split()
                # 명사 위주로 추출해야 하지만 간단한 구현으로 대체
                for word in words:
                    if len(word) > 4:  # 좀 더 긴 단어만 토픽으로 간주
                        topics.append(word)
        
        return list(set(topics)) 