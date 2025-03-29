"""
Text Embedding Module

이 모듈은 텍스트 데이터를 벡터로 변환하여 임베딩하는 기능을 제공합니다.
sentence-transformers를 사용하여 텍스트를 고차원 벡터 공간에 매핑합니다.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Union, Any, Optional
from sentence_transformers import SentenceTransformer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextEmbedder:
    """텍스트를 임베딩 벡터로 변환하는 클래스"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        TextEmbedder 초기화
        
        Args:
            model_name (str): 사용할 sentence-transformers 모델 이름
                            기본값은 'all-MiniLM-L6-v2'로 속도와 품질의 균형이 좋은 모델
        """
        logger.info(f"임베딩 모델 '{model_name}' 로딩 중...")
        try:
            self.model = SentenceTransformer(model_name)
            self.vector_size = self.model.get_sentence_embedding_dimension()
            logger.info(f"모델 로딩 완료. 벡터 크기: {self.vector_size}")
        except Exception as e:
            logger.error(f"모델 로딩 중 오류 발생: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        단일 텍스트를 임베딩 벡터로 변환
        
        Args:
            text (str): 임베딩할 텍스트
            
        Returns:
            np.ndarray: 임베딩 벡터
        """
        if not text or not text.strip():
            logger.warning("빈 텍스트는 임베딩할 수 없습니다.")
            # 빈 벡터 반환
            return np.zeros(self.vector_size)
        
        return self.model.encode(text)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        여러 텍스트를 임베딩 벡터로 변환
        
        Args:
            texts (List[str]): 임베딩할 텍스트 리스트
            
        Returns:
            np.ndarray: 임베딩 벡터 배열
        """
        if not texts:
            logger.warning("임베딩할 텍스트가 없습니다.")
            return np.array([])
        
        # 빈 텍스트 필터링
        valid_texts = [text for text in texts if text and text.strip()]
        if len(valid_texts) < len(texts):
            logger.warning(f"{len(texts) - len(valid_texts)}개의 빈 텍스트가 무시되었습니다.")
        
        if not valid_texts:
            return np.array([])
        
        return self.model.encode(valid_texts)
    
    def embed_articles(self, articles: List[Dict[str, Any]], 
                      text_fields: List[str] = ['title', 'content']) -> Dict[str, Any]:
        """
        기사 데이터에서 지정된 필드를 임베딩
        
        Args:
            articles (List[Dict]): 기사 데이터 리스트
            text_fields (List[str]): 임베딩할 텍스트 필드 목록
                                   기본값은 ['title', 'content']
                                   
        Returns:
            Dict: 임베딩 데이터를 포함한 결과
                {
                    'embeddings': 임베딩 벡터 리스트,
                    'texts': 임베딩된 텍스트 리스트,
                    'metadata': 각 텍스트의 메타데이터 리스트
                }
        """
        texts = []
        metadata = []
        
        for article in articles:
            for field in text_fields:
                if field in article and article[field]:
                    texts.append(article[field])
                    # 메타데이터 생성
                    meta = {
                        'source': 'article',
                        'field': field,
                        'article_id': article.get('id', ''),
                        'title': article.get('title', ''),
                        'link': article.get('link', ''),
                        'published': article.get('published', '')
                    }
                    metadata.append(meta)
        
        logger.info(f"{len(texts)}개의 텍스트 임베딩 중...")
        embeddings = self.embed_texts(texts)
        
        return {
            'embeddings': embeddings,
            'texts': texts,
            'metadata': metadata
        }

def process_rss_data(json_path: str, 
                    model_name: str = 'all-MiniLM-L6-v2',
                    output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    RSS JSON 데이터를 로드하여 텍스트 임베딩 생성
    
    Args:
        json_path (str): RSS 데이터가 저장된 JSON 파일 경로
        model_name (str): 사용할 임베딩 모델 이름
        output_path (str, optional): 임베딩 결과를 저장할 파일 경로
        
    Returns:
        Dict: 임베딩 결과
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
    
    # 임베더 초기화
    embedder = TextEmbedder(model_name)
    
    # 임베딩 생성
    result = embedder.embed_articles(articles)
    
    # 결과 저장 (선택사항)
    if output_path:
        # 임베딩은 크기가 크므로 numpy 배열로 별도 저장
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        base_name = os.path.splitext(output_path)[0]
        
        # 임베딩 저장
        np.save(f"{base_name}_embeddings.npy", result['embeddings'])
        
        # 텍스트와 메타데이터 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'texts': result['texts'],
                'metadata': result['metadata'],
                'embedding_file': f"{base_name}_embeddings.npy"
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"임베딩 데이터가 {output_path}에 저장되었습니다.")
    
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RSS 데이터에서 텍스트 임베딩을 생성합니다.')
    parser.add_argument('json_path', help='RSS 데이터가 저장된 JSON 파일 경로')
    parser.add_argument('-m', '--model', default='all-MiniLM-L6-v2', help='사용할 임베딩 모델 이름')
    parser.add_argument('-o', '--output', help='임베딩 결과를 저장할 파일 경로')
    
    args = parser.parse_args()
    
    try:
        process_rss_data(args.json_path, args.model, args.output)
    except Exception as e:
        logger.error(f"임베딩 처리 중 오류 발생: {str(e)}")
        print(f"오류 발생: {str(e)}") 