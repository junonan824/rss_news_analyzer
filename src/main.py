"""
RSS 피드 처리 및 벡터 DB 검색 통합 스크립트

이 스크립트는 RSS 피드 수집, 임베딩 생성, 벡터 DB 저장, 검색, 지식 그래프 생성까지의
전체 과정을 수행합니다.
"""

import os
import argparse
import logging
from src.rss_fetch.rss_fetch import process_rss_feed
from src.embeddings.embedding import process_rss_data
from src.embeddings.vector_db import load_rss_to_vectordb
from src.knowledge_graph.entity_extractor import EntityExtractor
from src.knowledge_graph.graph_builder import KnowledgeGraph

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='RSS 피드 처리 및 벡터 DB 검색, 지식 그래프 생성')
    parser.add_argument('url', help='RSS 피드 URL')
    parser.add_argument('-o', '--output_dir', default='data', help='출력 디렉토리')
    parser.add_argument('-c', '--collection', default='rss_articles', help='벡터 DB 컬렉션 이름')
    parser.add_argument('-q', '--query', help='검색 쿼리 (선택사항)')
    parser.add_argument('-n', '--num_results', type=int, default=5, help='검색 결과 수')
    parser.add_argument('-g', '--graph', action='store_true', help='지식 그래프 생성')
    parser.add_argument('-v', '--visualize', action='store_true', help='그래프 시각화')
    parser.add_argument('--ner_model', default='en_core_web_sm', help='NER에 사용할 spaCy 모델')
    
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
        
        # 지식 그래프 생성 (선택사항)
        if args.graph:
            # 개체 추출
            logger.info("NER을 사용하여 개체 추출 중...")
            import json
            with open(rss_json_path, 'r', encoding='utf-8') as f:
                rss_data = json.load(f)
            
            extractor = EntityExtractor(model_name=args.ner_model)
            articles_with_entities = extractor.extract_entities_batch(rss_data.get('articles', []))
            
            # 추출된 개체 저장
            entities_json_path = os.path.join(args.output_dir, 'entities.json')
            with open(entities_json_path, 'w', encoding='utf-8') as f:
                json.dump({"articles": articles_with_entities}, f, ensure_ascii=False, indent=2)
            
            logger.info(f"개체 추출 결과가 {entities_json_path}에 저장되었습니다.")
            
            # 그래프 생성
            logger.info("지식 그래프 생성 중...")
            graph = KnowledgeGraph()
            graph.build_from_articles(articles_with_entities)
            
            # 그래프 저장
            graph_file_path = os.path.join(args.output_dir, 'knowledge_graph.json')
            graph.save(graph_file_path)
            
            # 그래프 통계 출력
            node_types = {}
            for _, attr in graph.graph.nodes(data=True):
                node_type = attr.get('type', 'UNKNOWN')
                node_types[node_type] = node_types.get(node_type, 0) + 1
            
            edge_types = {}
            for _, _, attr in graph.graph.edges(data=True):
                edge_type = attr.get('type', 'UNKNOWN')
                edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
            
            print("\n그래프 통계:")
            print(f"- 총 노드 수: {graph.graph.number_of_nodes()}")
            print(f"- 총 엣지 수: {graph.graph.number_of_edges()}")
            
            print("\n노드 유형별 통계:")
            for node_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {node_type}: {count}")
            
            print("\n엣지 유형별 통계:")
            for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {edge_type}: {count}")
            
            # 그래프 시각화 (선택사항)
            if args.visualize:
                graph_image_path = os.path.join(args.output_dir, 'knowledge_graph.png')
                print(f"\n그래프 시각화 중... (이미지: {graph_image_path})")
                graph.visualize(graph_image_path)
                
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}")
        print(f"오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 