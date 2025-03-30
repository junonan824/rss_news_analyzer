"""
채팅 메시지 처리 모듈.

사용자 메시지를 처리하고 응답을 생성합니다.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio

from .context_manager import ContextManager
from src.embeddings.vector_db import VectorDB
from src.generation.generation import TextGenerator

# RSS 피드 관련 임포트 추가
from src.rss_fetch.rss_fetch import process_rss_feed, find_relevant_rss_feeds
from src.embeddings.embedding import process_rss_data

import os
import json
import tempfile

# 로깅 설정
logger = logging.getLogger(__name__)

# 유사도 임계값 및 최소 문서 수 설정
SIMILARITY_THRESHOLD = 15.0
MIN_DOCUMENTS_REQUIRED = 1

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
        self.data_dir = os.path.join("data", "on_demand")
        os.makedirs(self.data_dir, exist_ok=True)
    
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
            logger.info(f"검색 쿼리: '{query_with_context}'")
            
            relevant_docs = self.vector_db.search(
                query_with_context, 
                n_results=3
            )
            
            # 검색 결과 로깅
            doc_count = len(relevant_docs.get("documents", [[]])[0]) if relevant_docs and "documents" in relevant_docs else 0
            logger.info(f"검색 결과: {doc_count}개 문서 찾음")
            
            if doc_count > 0:
                distances = relevant_docs.get("distances", [[]])[0]
                logger.info(f"유사도 점수: {distances}")
                
                # 유사도 점수 확인 - 너무 낮으면 새 데이터 가져오기
                has_high_quality_results = any(distance < (1.0/SIMILARITY_THRESHOLD) for distance in distances)
                if doc_count < MIN_DOCUMENTS_REQUIRED or not has_high_quality_results:
                    logger.info(f"검색 결과 품질이 낮음, 새로운 RSS 피드 가져오기")
                    
                    # 새 RSS 피드 가져와서 처리
                    updated = await self._fetch_and_update_rss(query_with_context)
                    
                    if updated:
                        # 다시 검색
                        relevant_docs = self.vector_db.search(
                            query_with_context, 
                            n_results=3
                        )
                        
                        doc_count = len(relevant_docs.get("documents", [[]])[0]) if relevant_docs else 0
                        logger.info(f"업데이트 후 검색 결과: {doc_count}개 문서 찾음")
            
            # 문서가 여전히 없으면 일반 응답 생성
            if not relevant_docs or doc_count == 0:
                logger.info("No relevant documents found, using direct generation")
                return await self._generate_direct_response(context, message)
            
            # RAG 기반 응답 생성
            response = await self._generate_rag_response(context, message, relevant_docs)
            
            return response
        
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return "죄송합니다. 메시지를 처리하는 동안 오류가 발생했습니다."
    
    async def _fetch_and_update_rss(self, query: str) -> bool:
        """쿼리와 관련된 RSS 피드를 가져와 벡터 DB 업데이트
        
        Args:
            query: 검색 쿼리
            
        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            # 1. 키워드에 관련된 RSS 피드 URL 찾기
            relevant_feeds = find_relevant_rss_feeds(query)
            
            if not relevant_feeds:
                logger.info("쿼리와 관련된 RSS 피드를 찾을 수 없습니다.")
                return False
            
            # 2. 각 피드 처리 및 벡터 DB 업데이트
            for feed_url in relevant_feeds[:3]:  # 최대 3개 피드만 처리
                # 임시 파일 사용
                with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
                    rss_json_path = temp_file.name
                
                # RSS 피드 처리
                logger.info(f"RSS 피드 처리 중: {feed_url}")
                await asyncio.to_thread(process_rss_feed, feed_url, rss_json_path)
                
                # 임베딩 생성
                embedding_json_path = os.path.join(self.data_dir, f"embedding_{os.path.basename(rss_json_path)}")
                await asyncio.to_thread(process_rss_data, rss_json_path, output_path=embedding_json_path)
                
                # 벡터 DB 업데이트
                with open(rss_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 기존 컬렉션에 추가
                articles = data.get('articles', [])
                
                if articles:
                    # 임베딩 데이터 로드
                    with open(embedding_json_path, 'r', encoding='utf-8') as f:
                        embedding_data = json.load(f)
                        
                    texts = embedding_data.get('texts', [])
                    metadata = embedding_data.get('metadata', [])
                    
                    # 임베딩 파일 경로
                    embedding_file = embedding_data.get('embedding_file')
                    embeddings = None
                    
                    if embedding_file and os.path.exists(embedding_file):
                        import numpy as np
                        embeddings = np.load(embedding_file)
                    
                    # 벡터 DB 업데이트
                    self.vector_db.add_texts(
                        texts=texts,
                        metadatas=metadata,
                        embeddings=embeddings.tolist() if embeddings is not None else None
                    )
                    
                    logger.info(f"{len(texts)}개의 새 문서가 벡터 DB에 추가되었습니다.")
                
                # 임시 파일 정리
                try:
                    os.remove(rss_json_path)
                    if os.path.exists(embedding_file):
                        os.remove(embedding_file)
                except:
                    pass
            
            return True
        
        except Exception as e:
            logger.error(f"RSS 피드 업데이트 중 오류 발생: {e}")
            return False
    
    def _build_enhanced_query(self, context, message: str) -> str:
        """컨텍스트를 고려한 향상된 검색 쿼리 생성"""
        # 관련 엔티티 추출
        entities = self.context_manager.extract_entities_from_context(context)
        
        # 특정 키워드가 포함된 경우 관련 키워드 추가
        specific_keywords = {
            "인도네시아": ["indonesia", "earthquake", "sumatra", "java"],
            "지진": ["earthquake", "tremor", "seismic", "magnitude"],
            "태풍": ["typhoon", "hurricane", "storm"],
            "홍수": ["flood", "flooding", "inundation"],
        }
        
        additional_keywords = []
        for keyword, related in specific_keywords.items():
            if keyword.lower() in message.lower():
                additional_keywords.extend(related)
        
        # 중요 엔티티가 있으면 원본 쿼리에 추가
        if entities or additional_keywords:
            top_entities = entities[:3] if entities else []  # 최대 3개만 사용
            enhanced_query = f"{message} {' '.join(top_entities)} {' '.join(additional_keywords)}"
            logger.debug(f"Enhanced query: {enhanced_query} (original: {message})")
            return enhanced_query
        
        return message
    
    async def _generate_direct_response(self, context, message: str) -> str:
        """직접 응답 생성 (관련 문서 없음)
        
        Args:
            context: 대화 컨텍스트
            message: 사용자 메시지
            
        Returns:
            생성된 응답
        """
        # 컨텍스트 기반 프롬프트 구성
        prompt = self.context_manager.build_prompt_with_context(context, message)
        
        # 직접 응답 생성
        try:
            return await self.generator.generate_text_async(
                prompt=prompt, 
                max_tokens=500,
                temperature=0.7
            )
        except Exception as e:
            logger.error(f"Error generating direct response: {e}")
            return "죄송합니다. 현재 이 정보에 대한 응답을 생성할 수 없습니다."
            
    async def _generate_rag_response(self, context, message: str, relevant_docs: Dict) -> str:
        """RAG 기반 응답 생성
        
        Args:
            context: 대화 컨텍스트
            message: 사용자 메시지
            relevant_docs: 검색 결과
            
        Returns:
            생성된 응답
        """
        try:
            # 관련 문서 추출
            documents = relevant_docs.get("documents", [[]])[0]
            metadatas = relevant_docs.get("metadatas", [[]])[0]
            
            # 문서와 메타데이터 결합
            contexts = []
            for doc, meta in zip(documents, metadatas):
                context_entry = f"제목: {meta.get('title', '제목 없음')}\n\n{doc}"
                if 'url' in meta:
                    context_entry += f"\n\nURL: {meta.get('url')}"
                contexts.append(context_entry)
            
            # RAG 응답 생성
            response = await self.generator.generate_text_with_retrieved_context_async(
                query=message,
                contexts=contexts,
                max_tokens=800,
                temperature=0.7
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return "죄송합니다. 관련 정보를 기반으로 응답을 생성하는 중 오류가 발생했습니다."
    
    def _generate_fallback_response(self, message: str) -> str:
        """폴백 응답 생성
        
        Args:
            message: 사용자 메시지
            
        Returns:
            생성된 응답
        """
        return "죄송합니다. 현재 검색 시스템이 이용 불가능합니다. 나중에 다시 시도해주세요." 