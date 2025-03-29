"""
Neo4j 어댑터 모듈

NetworkX 그래프를 Neo4j 데이터베이스로 내보내는 기능을 제공합니다.
"""

import os
import logging
from typing import Dict, Any, Optional
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)

class Neo4jAdapter:
    """
    NetworkX 그래프를 Neo4j로 변환하는 어댑터 클래스
    """
    
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None):
        """
        Neo4j 연결 초기화
        
        Args:
            uri: Neo4j 서버 URI (기본값: 환경 변수 또는 "bolt://localhost:7687")
            user: Neo4j 사용자명 (기본값: 환경 변수 또는 "neo4j")
            password: Neo4j 비밀번호 (기본값: 환경 변수 또는 "password")
        """
        self.uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD", "password")
        self.driver = None
        
    def connect(self) -> bool:
        """
        Neo4j 데이터베이스에 연결
        
        Returns:
            bool: 연결 성공 여부
        """
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # 연결 테스트
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Neo4j 연결 성공: {self.uri}")
            return True
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Neo4j 연결 실패: {str(e)}")
            return False
            
    def close(self):
        """연결 종료"""
        if self.driver:
            self.driver.close()
            
    def clear_database(self):
        """데이터베이스 초기화 (모든 노드와 관계 삭제)"""
        if not self.driver:
            if not self.connect():
                return False
                
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            logger.info("Neo4j 데이터베이스 초기화 완료")
            return True
        except Exception as e:
            logger.error(f"Neo4j 데이터베이스 초기화 실패: {str(e)}")
            return False
            
    def import_graph(self, graph):
        """
        NetworkX 그래프를 Neo4j로 내보내기
        
        Args:
            graph: NetworkX 그래프 객체
            
        Returns:
            bool: 성공 여부
        """
        if not self.driver:
            if not self.connect():
                return False
                
        try:
            # 1. 노드 생성
            for node_id, attrs in graph.nodes(data=True):
                self._create_node(node_id, attrs)
                
            # 2. 엣지(관계) 생성
            for source, target, attrs in graph.edges(data=True):
                self._create_relationship(source, target, attrs)
                
            logger.info(f"Neo4j 가져오기 완료: {graph.number_of_nodes()} 노드, {graph.number_of_edges()} 관계")
            return True
        except Exception as e:
            logger.error(f"Neo4j 가져오기 실패: {str(e)}")
            return False
            
    def _create_node(self, node_id: str, attributes: Dict[str, Any]):
        """
        Neo4j에 노드 생성
        
        Args:
            node_id: 노드 ID
            attributes: 노드 속성 딕셔너리
        """
        # 노드 타입 추출 (Neo4j 레이블로 사용)
        node_type = attributes.get('type', 'Entity')
        
        # 노드 속성에서 특수 문자 처리
        cleaned_attrs = {k: v for k, v in attributes.items() if k != 'type'}
        
        # Cypher 쿼리 생성 및 실행
        query = (
            f"MERGE (n:{node_type} {{id: $id}}) "
            "SET n += $props "
            "RETURN n"
        )
        
        with self.driver.session() as session:
            session.run(query, id=node_id, props=cleaned_attrs)
            
    def _create_relationship(self, source_id: str, target_id: str, attributes: Dict[str, Any]):
        """
        Neo4j에 관계 생성
        
        Args:
            source_id: 소스 노드 ID
            target_id: 타겟 노드 ID
            attributes: 관계 속성 딕셔너리
        """
        # 관계 타입 추출
        rel_type = attributes.get('type', 'RELATED_TO').upper()
        
        # 관계 속성 (타입 제외)
        cleaned_attrs = {k: v for k, v in attributes.items() if k != 'type'}
        
        # Cypher 쿼리 생성 및 실행
        query = (
            "MATCH (source {id: $source_id}) "
            "MATCH (target {id: $target_id}) "
            f"MERGE (source)-[r:{rel_type}]->(target) "
            "SET r += $props "
            "RETURN r"
        )
        
        with self.driver.session() as session:
            session.run(query, source_id=source_id, target_id=target_id, props=cleaned_attrs)
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Neo4j 데이터베이스 통계 조회
        
        Returns:
            Dict: 통계 정보 딕셔너리
        """
        if not self.driver:
            if not self.connect():
                return {"error": "Neo4j 연결 실패"}
                
        try:
            with self.driver.session() as session:
                # 노드 수 쿼리
                node_count_result = session.run("MATCH (n) RETURN count(n) AS count").single()
                node_count = node_count_result["count"] if node_count_result else 0
                
                # 관계 수 쿼리
                rel_count_result = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()
                rel_count = rel_count_result["count"] if rel_count_result else 0
                
                # 노드 레이블(타입) 통계
                label_stats_result = session.run(
                    "MATCH (n) "
                    "WITH labels(n) AS labels "
                    "UNWIND labels AS label "
                    "RETURN label, count(label) AS count "
                    "ORDER BY count DESC"
                )
                label_stats = {record["label"]: record["count"] for record in label_stats_result}
                
                # 관계 타입 통계
                rel_stats_result = session.run(
                    "MATCH ()-[r]->() "
                    "WITH type(r) AS rel_type "
                    "RETURN rel_type, count(rel_type) AS count "
                    "ORDER BY count DESC"
                )
                rel_stats = {record["rel_type"]: record["count"] for record in rel_stats_result}
                
                return {
                    "node_count": node_count,
                    "relationship_count": rel_count,
                    "node_types": label_stats,
                    "relationship_types": rel_stats
                }
        except Exception as e:
            logger.error(f"Neo4j 통계 조회 실패: {str(e)}")
            return {"error": str(e)} 