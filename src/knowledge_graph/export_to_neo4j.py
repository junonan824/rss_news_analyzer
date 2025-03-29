"""
Neo4j 내보내기 스크립트

JSON 형식의 지식 그래프를 Neo4j 데이터베이스로 내보냅니다.
"""

import os
import sys
import argparse
import logging
from src.knowledge_graph.graph_builder import KnowledgeGraph
from src.knowledge_graph.neo4j_adapter import Neo4jAdapter

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='지식 그래프를 Neo4j 데이터베이스로 내보내기')
    parser.add_argument('graph_file', help='지식 그래프 JSON 파일 경로')
    parser.add_argument('--uri', default=os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
                        help='Neo4j 서버 URI')
    parser.add_argument('--user', default=os.environ.get('NEO4J_USER', 'neo4j'),
                        help='Neo4j 사용자명')
    parser.add_argument('--password', default=os.environ.get('NEO4J_PASSWORD', 'password'),
                        help='Neo4j 비밀번호')
    parser.add_argument('--clear', action='store_true',
                        help='내보내기 전에 Neo4j 데이터베이스 초기화')
    
    args = parser.parse_args()
    
    try:
        # 1. 그래프 파일 로드
        if not os.path.exists(args.graph_file):
            logger.error(f"파일을 찾을 수 없습니다: {args.graph_file}")
            return 1
        
        logger.info(f"지식 그래프 파일 로드 중: {args.graph_file}")
        kg = KnowledgeGraph.load(args.graph_file)
        
        # 2. Neo4j 연결 테스트
        logger.info(f"Neo4j 연결 중: {args.uri}")
        adapter = Neo4jAdapter(args.uri, args.user, args.password)
        if not adapter.connect():
            logger.error("Neo4j 연결 실패. URI, 사용자명, 비밀번호를 확인하세요.")
            return 1
        
        # 3. 데이터베이스 초기화 (선택적)
        if args.clear:
            logger.info("Neo4j 데이터베이스 초기화 중...")
            adapter.clear_database()
        
        # 4. 그래프 내보내기
        logger.info(f"{kg.graph.number_of_nodes()}개 노드, {kg.graph.number_of_edges()}개 엣지를 Neo4j로 내보내는 중...")
        result = adapter.import_graph(kg.graph)
        
        if result:
            # 5. 통계 출력
            stats = adapter.get_stats()
            logger.info("Neo4j 내보내기 완료!")
            logger.info(f"Neo4j 통계: {stats['node_count']}개 노드, {stats['relationship_count']}개 관계")
            
            # 노드 유형별 통계
            logger.info("노드 유형별 통계:")
            for node_type, count in stats.get('node_types', {}).items():
                logger.info(f"  {node_type}: {count}")
            
            # 관계 유형별 통계
            logger.info("관계 유형별 통계:")
            for rel_type, count in stats.get('relationship_types', {}).items():
                logger.info(f"  {rel_type}: {count}")
                
            logger.info(f"Neo4j 브라우저에서 그래프를 확인하세요: http://localhost:7474")
            
            # 샘플 Cypher 쿼리 제안
            logger.info("샘플 Cypher 쿼리:")
            logger.info("  MATCH (n) RETURN n LIMIT 100")
            logger.info("  MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100")
            logger.info("  MATCH (a:ARTICLE)-[:MENTIONS]->(p:PERSON) RETURN a, p")
            
            return 0
        else:
            logger.error("Neo4j 내보내기 실패")
            return 1
    
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        return 1
    finally:
        if 'adapter' in locals():
            adapter.close()

if __name__ == "__main__":
    sys.exit(main()) 