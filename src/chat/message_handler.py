"""
채팅 메시지 처리 모듈.

사용자 메시지를 처리하고 응답을 생성합니다.
"""

import logging
from typing import List, Dict, Any, Optional

from .context_manager import ContextManager
from src.embeddings.vector_db import VectorDB
from src.generation.generation import TextGenerator

# 로깅 설정
logger = logging.getLogger(__name__)


class MessageHandler:
    """채팅 메시지 처리 클래스"""
    
    def __init__(
        self, 
        vector_db: Optional[VectorDB] = None,
        generator: Optional[TextGenerator] = None,
        collection_name: str = "rss_articles"
    ):
        """
        Args:
            vector_db: 벡터 데이터베이스 핸들러
            generator: 텍스트 생성기
            collection_name: 사용할 컬렉션 이름
        """
        self.vector_db = vector_db
        self.generator = generator
        self.collection_name = collection_name
        self.context_manager = ContextManager()
    
    def _load_dependencies(self):
        """필요한 의존성 로드"""
        if self.vector_db is None:
            from src.embeddings.vector_db import VectorDB
            try:
                self.vector_db = VectorDB(collection_name=self.collection_name)
                logger.info("Vector DB handler initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Vector DB: {e}")
                self.vector_db = None
        
        if self.generator is None:
            from src.generation.generation import TextGenerator
            try:
                self.generator = TextGenerator()
                logger.info("Text generator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Text Generator: {e}")
                self.generator = None
    
    async def handle_message(self, session, message: str) -> str:
        """메시지 처리
        
        Args:
            session: 채팅 세션 객체
            message: 사용자 메시지
            
        Returns:
            생성된 응답
        """
        self._load_dependencies()
        
        # 세션에서 컨텍스트 가져오기
        context = session.get_context()
        
        # 응답 처리
        try:
            # 벡터 DB가 없으면 일반 응답 생성
            if self.vector_db is None or self.generator is None:
                logger.warning("Vector DB or Generator not available, using fallback response")
                return self._generate_fallback_response(message)
            
            # 관련 문서 검색
            query_with_context = self._build_enhanced_query(context, message)
            relevant_docs = self.vector_db.search(
                query_with_context, 
                n_results=3
            )
            
            # 검색 결과 구조 로깅 추가
            logger.debug(f"검색 결과 구조: {relevant_docs.keys()}")
            
            # 문제가 되는 조건문 수정
            if not relevant_docs or len(relevant_docs.get("documents", [[]])[0]) == 0:
                logger.info("No relevant documents found, using direct generation")
                return await self._generate_direct_response(context, message)
            
            # RAG 기반 응답 생성
            prompt = self.context_manager.build_prompt_with_context(context, message)
            response = self._generate_rag_response(prompt, relevant_docs, context)
            
            return response
        
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return "죄송합니다. 메시지를 처리하는 동안 오류가 발생했습니다."
    
    def _build_enhanced_query(self, context, message: str) -> str:
        """컨텍스트를 고려한 향상된 검색 쿼리 생성"""
        # 관련 엔티티 추출
        entities = self.context_manager.extract_entities_from_context(context)
        
        # 중요 엔티티가 있으면 원본 쿼리에 추가
        if entities:
            top_entities = entities[:3]  # 최대 3개만 사용
            enhanced_query = f"{message} {' '.join(top_entities)}"
            logger.debug(f"Enhanced query: {enhanced_query} (original: {message})")
            return enhanced_query
        
        return message
    
    async def _generate_direct_response(self, context, message: str) -> str:
        """일반 응답 생성"""
        if self.generator:
            prompt = self.context_manager.build_prompt_with_context(context, message)
            try:
                response = self.generator.generate_text(prompt)
                return response
            except Exception as e:
                logger.error(f"Error generating direct response: {e}")
        
        return self._generate_fallback_response(message)
    
    def _generate_rag_response(self, prompt: str, relevant_docs, context) -> str:
        """RAG 기반 응답 생성"""
        if not self.generator:
            return self._generate_fallback_response(prompt)
        
        try:
            # 관련 문서 내용 추출 - documents 구조에 맞게 수정
            documents = relevant_docs.get("documents", [[]])[0]
            metadatas = relevant_docs.get("metadatas", [[]])[0]
            
            doc_texts = []
            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                title = meta.get("title", "제목 없음")
                # 출처 정보 추가
                source = meta.get("source", "") or meta.get("field", "")
                published = meta.get("published", "")
                doc_texts.append(f"문서 {i+1}:\n제목: {title}\n출처: {source}\n발행일: {published}\n내용: {doc}")
            
            # 문서 컨텍스트 구성
            docs_context = "\n\n".join(doc_texts)
            
            # RAG 프롬프트 구성 - 지시사항 명확화
            rag_prompt = f"""다음 정보를 바탕으로 사용자의 최근 질문에 답변해주세요:

검색된 관련 문서:
{docs_context}

대화 컨텍스트:
{prompt}

지침:
1. 제공된 문서에 관련 정보가 있으면 그 정보를 바탕으로 답변하세요.
2. 정보가 불충분하거나 관련이 없으면 솔직하게 모른다고 답변하세요.
3. 답변은 친절하고 자연스러운 한국어로 작성하세요.

답변:"""
            
            # 응답 생성
            response = self.generator.generate_text(rag_prompt)
            return response
        
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, message: str) -> str:
        """폴백 응답 생성"""
        return "메시지를 받았습니다. 하지만 현재 적절한 응답을 생성할 수 없습니다." 