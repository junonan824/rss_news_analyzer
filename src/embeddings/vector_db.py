"""
Vector Database Integration Module

이 모듈은 ChromaDB를 사용하여 텍스트 임베딩을 저장하고 검색하는 기능을 제공합니다.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import chromadb
from chromadb.utils import embedding_functions

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorDB:
    """ChromaDB를 사용한 벡터 데이터베이스 클래스"""
    
    def __init__(self, 
                collection_name: str = "rss_articles",
                persist_directory: Optional[str] = "data/chroma_db",
                embedding_model: str = "all-MiniLM-L6-v2"):
        """
        VectorDB 초기화
        
        Args:
            collection_name (str): ChromaDB 컬렉션 이름
            persist_directory (str, optional): 데이터베이스 저장 디렉토리
            embedding_model (str): 임베딩 모델 이름 (직접 임베딩을 생성하는 경우 사용)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # 저장 디렉토리가 지정된 경우 생성
        if persist_directory and not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
        
        # ChromaDB 클라이언트 초기화
        logger.info(f"ChromaDB 클라이언트 초기화 중... (저장 경로: {persist_directory})")
        try:
            if persist_directory:
                self.client = chromadb.PersistentClient(path=persist_directory)
            else:
                self.client = chromadb.Client()
                
            # 임베딩 함수 설정 (직접 임베딩을 제공하는 경우에는 사용하지 않음)
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
            
            # 컬렉션 생성 또는 가져오기
            try:
                self.collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"기존 컬렉션 '{collection_name}'을 불러왔습니다.")
            except Exception:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_function
                )
                logger.info(f"새 컬렉션 '{collection_name}'을 생성했습니다.")
                
        except Exception as e:
            logger.error(f"ChromaDB 초기화 중 오류 발생: {str(e)}")
            raise
    
    def add_texts(self, 
                 texts: List[str], 
                 metadatas: Optional[List[Dict[str, Any]]] = None,
                 ids: Optional[List[str]] = None,
                 embeddings: Optional[List[List[float]]] = None) -> List[str]:
        """
        텍스트를 벡터 DB에 추가
        
        Args:
            texts (List[str]): 추가할 텍스트 목록
            metadatas (List[Dict], optional): 각 텍스트의 메타데이터
            ids (List[str], optional): 각 텍스트의 고유 ID (없으면 자동 생성)
            embeddings (List[List[float]], optional): 사전 계산된 임베딩 벡터
            
        Returns:
            List[str]: 추가된 텍스트의 ID 목록
        """
        if not texts:
            logger.warning("추가할 텍스트가 없습니다.")
            return []
        
        # ID가 없는 경우 자동 생성
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # 메타데이터가 없는 경우 빈 딕셔너리 생성
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        
        logger.info(f"{len(texts)}개의 텍스트를 벡터 DB에 추가 중...")
        
        try:
            # 임베딩이 제공된 경우 직접 사용
            if embeddings is not None:
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
            else:
                # 임베딩이 없는 경우 ChromaDB의 임베딩 함수 사용
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
            
            logger.info(f"{len(texts)}개의 텍스트가 성공적으로 추가되었습니다.")
            return ids
        except Exception as e:
            logger.error(f"텍스트 추가 중 오류 발생: {str(e)}")
            raise
    
    def search(self, 
              query: str, 
              n_results: int = 5,
              where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        쿼리와 유사한 텍스트 검색
        
        Args:
            query (str): 검색 쿼리
            n_results (int): 반환할 결과 수
            where (Dict, optional): 메타데이터 필터링 조건
                  예시: {"$and": [{"field": "title"}, {"source": "article"}]}
                  다중 조건은 $and 또는 $or 연산자를 사용해야 함
            
        Returns:
            Dict: 검색 결과
        """
        logger.info(f"쿼리 '{query}'로 검색 중...")
        
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where
            )
            
            logger.info(f"{len(results.get('documents', [''])[0])}개의 결과를 찾았습니다.")
            return results
        except Exception as e:
            logger.error(f"검색 중 오류 발생: {str(e)}")
            raise
    
    def search_by_vector(self, 
                       vector: List[float], 
                       n_results: int = 5,
                       where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        임베딩 벡터와 유사한 텍스트 검색
        
        Args:
            vector (List[float]): 검색할 임베딩 벡터
            n_results (int): 반환할 결과 수
            where (Dict, optional): 메타데이터 필터링 조건
                  예시: {"$and": [{"field": "title"}, {"source": "article"}]}
                  다중 조건은 $and 또는 $or 연산자를 사용해야 함
            
        Returns:
            Dict: 검색 결과
        """
        logger.info("벡터 기반 검색 중...")
        
        try:
            results = self.collection.query(
                query_embeddings=[vector],
                n_results=n_results,
                where=where
            )
            
            logger.info(f"{len(results.get('documents', [''])[0])}개의 결과를 찾았습니다.")
            return results
        except Exception as e:
            logger.error(f"벡터 검색 중 오류 발생: {str(e)}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        컬렉션 통계 반환
        
        Returns:
            Dict: 컬렉션 통계 정보
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "count": count,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"컬렉션 통계 조회 중 오류 발생: {str(e)}")
            raise

def load_rss_to_vectordb(
    json_path: str,
    collection_name: str = "rss_articles",
    persist_directory: str = "data/chroma_db",
    embedding_path: Optional[str] = None
) -> VectorDB:
    """
    RSS 데이터를 벡터 DB에 로드
    
    Args:
        json_path (str): RSS 데이터가 저장된 JSON 파일 경로
        collection_name (str): 사용할 컬렉션 이름
        persist_directory (str): DB 저장 경로
        embedding_path (str, optional): 사전 계산된 임베딩 파일 경로
        
    Returns:
        VectorDB: 초기화된 VectorDB 인스턴스
    """
    # JSON 파일 로드
    logger.info(f"JSON 데이터 로드 중: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"JSON 파일 로드 중 오류 발생: {str(e)}")
        raise
    
    articles = data.get('articles', [])
    logger.info(f"{len(articles)}개의 기사를 찾았습니다.")
    
    # 임베딩 로드 (있는 경우)
    embeddings = None
    if embedding_path and os.path.exists(embedding_path):
        try:
            embedding_data = json.load(open(embedding_path, 'r', encoding='utf-8'))
            texts = embedding_data.get('texts', [])
            metadata = embedding_data.get('metadata', [])
            embedding_file = embedding_data.get('embedding_file')
            
            if embedding_file and os.path.exists(embedding_file):
                embeddings = np.load(embedding_file)
                logger.info(f"임베딩 로드 완료: {embedding_file}")
            else:
                logger.warning(f"임베딩 파일을 찾을 수 없습니다: {embedding_file}")
                from src.embeddings.embedding import TextEmbedder
                
                # 임베딩 생성
                embedder = TextEmbedder()
                texts = []
                metadata = []
                
                for article in articles:
                    # 제목과 본문 모두 저장
                    if article.get('title'):
                        texts.append(article['title'])
                        metadata.append({
                            'source': 'article',
                            'field': 'title',
                            'article_id': article.get('id', ''),
                            'title': article.get('title', ''),
                            'link': article.get('link', ''),
                            'published': article.get('published', '')
                        })
                    
                    if article.get('content'):
                        texts.append(article['content'])
                        metadata.append({
                            'source': 'article',
                            'field': 'content',
                            'article_id': article.get('id', ''),
                            'title': article.get('title', ''),
                            'link': article.get('link', ''),
                            'published': article.get('published', '')
                        })
                
                embeddings = embedder.embed_texts(texts)
            
        except Exception as e:
            logger.error(f"임베딩 로드 중 오류 발생: {str(e)}")
            logger.info("임베딩을 새로 생성합니다.")
            embeddings = None
    
    # 임베딩이 없는 경우 생성
    if embeddings is None:
        from src.embeddings.embedding import TextEmbedder
        
        # 임베딩 생성
        embedder = TextEmbedder()
        texts = []
        metadata = []
        
        for article in articles:
            # 제목과 본문 모두 저장
            if article.get('title'):
                texts.append(article['title'])
                metadata.append({
                    'source': 'article',
                    'field': 'title',
                    'article_id': article.get('id', ''),
                    'title': article.get('title', ''),
                    'link': article.get('link', ''),
                    'published': article.get('published', '')
                })
            
            if article.get('content'):
                texts.append(article['content'])
                metadata.append({
                    'source': 'article',
                    'field': 'content',
                    'article_id': article.get('id', ''),
                    'title': article.get('title', ''),
                    'link': article.get('link', ''),
                    'published': article.get('published', '')
                })
        
        embeddings = embedder.embed_texts(texts)
    
    # VectorDB 초기화
    vector_db = VectorDB(
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    # UUID 생성 (고유 ID)
    import uuid
    ids = [str(uuid.uuid4()) for _ in range(len(texts))]
    
    # 벡터 DB에 추가
    vector_db.add_texts(
        texts=texts,
        metadatas=metadata,
        ids=ids,
        embeddings=embeddings.tolist() if embeddings is not None else None
    )
    
    return vector_db

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RSS 데이터를 벡터 DB에 로드합니다.')
    parser.add_argument('json_path', help='RSS 데이터가 저장된 JSON 파일 경로')
    parser.add_argument('-c', '--collection', default='rss_articles', help='사용할 컬렉션 이름')
    parser.add_argument('-d', '--db_dir', default='data/chroma_db', help='DB 저장 경로')
    parser.add_argument('-e', '--embeddings', help='사전 계산된 임베딩 파일 경로')
    parser.add_argument('-q', '--query', help='검색할 쿼리 (선택사항)')
    parser.add_argument('-n', '--num_results', type=int, default=5, help='반환할 결과 수')
    
    args = parser.parse_args()
    
    try:
        # 벡터 DB에 RSS 데이터 로드
        db = load_rss_to_vectordb(
            json_path=args.json_path,
            collection_name=args.collection,
            persist_directory=args.db_dir,
            embedding_path=args.embeddings
        )
        
        print(f"벡터 DB에 데이터가 성공적으로 로드되었습니다.")
        stats = db.get_collection_stats()
        print(f"컬렉션 '{stats['collection_name']}'에 {stats['count']}개의 항목이 있습니다.")
        
        # 쿼리가 제공된 경우 검색 수행
        if args.query:
            print(f"\n쿼리: '{args.query}'에 대한 상위 {args.num_results}개 결과:")
            results = db.search(args.query, args.num_results)
            
            # 결과 출력
            documents = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            distances = results.get('distances', [[]])[0]
            
            for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
                print(f"\n{i+1}. 유사도: {1 - dist:.4f}")
                print(f"제목: {meta.get('title', 'N/A')}")
                print(f"출처: {meta.get('field', 'N/A')}")
                print(f"링크: {meta.get('link', 'N/A')}")
                print(f"내용: {doc[:200]}..." if len(doc) > 200 else f"내용: {doc}")
                
    except Exception as e:
        logger.error(f"벡터 DB 처리 중 오류 발생: {str(e)}")
        print(f"오류 발생: {str(e)}") 