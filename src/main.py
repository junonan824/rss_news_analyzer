"""
RSS 피드 처리 및 벡터 DB 검색 통합 스크립트

이 스크립트는 RSS 피드 수집, 임베딩 생성, 벡터 DB 저장, 검색까지의
전체 과정을 수행합니다.
"""

import os
import argparse
import logging
from src.rss_fetch.rss_fetch import process_rss_feed
from src.embeddings.embedding import process_rss_data
from src.embeddings.vector_db import load_rss_to_vectordb

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='RSS 피드 처리 및 벡터 DB 검색')
    parser.add_argument('url', help='RSS 피드 URL')
    parser.add_argument('-o', '--output_dir', default='data', help='출력 디렉토리')
    parser.add_argument('-c', '--collection', default='rss_articles', help='벡터 DB 컬렉션 이름')
    parser.add_argument('-q', '--query', help='검색 쿼리 (선택사항)')
    parser.add_argument('-n', '--num_results', type=int, default=5, help='검색 결과 수')
    
    args = parser.parse_args()
    
    try:
        # 출력 디렉토리 생성
        os.makedirs(args.output_dir, exist_ok=True)
        
        # RSS 피드 처리
        rss_json_path = os.path.join(args.output_dir, 'rss_data.json')
        logger.info(f"RSS 피드 처리 중: {args.url}")
        process_rss_feed(args.url, rss_json_path)
        
        # 임베딩 생성
        embedding_json_path = os.path.join(args.output_dir, 'embeddings.json')
        logger.info("임베딩 생성 중...")
        process_rss_data(rss_json_path, output_path=embedding_json_path)
        
        # 벡터 DB에 로드
        db_path = os.path.join(args.output_dir, 'chroma_db')
        logger.info("벡터 DB에 로드 중...")
        db = load_rss_to_vectordb(
            json_path=rss_json_path,
            collection_name=args.collection,
            persist_directory=db_path,
            embedding_path=embedding_json_path
        )
        
        stats = db.get_collection_stats()
        print(f"\n벡터 DB 로드 완료: 컬렉션 '{stats['collection_name']}'에 {stats['count']}개의 항목이 있습니다.")
        
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
        logger.error(f"처리 중 오류 발생: {str(e)}")
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 