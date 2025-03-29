"""
Knowledge Graph Builder Module

이 모듈은 추출된 개체(Entity)들을 사용하여 지식 그래프를 구축하는 기능을 제공합니다.
NetworkX를 사용하여 그래프를 생성하고 조작합니다.
"""

import os
import json
import logging
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Set, Optional, Union
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """NetworkX를 사용한 지식 그래프 클래스"""
    
    def __init__(self):
        """KnowledgeGraph 초기화"""
        self.graph = nx.MultiDiGraph()
        logger.info("지식 그래프 초기화 완료")
    
    def add_entity(self, entity_id: str, entity_text: str, entity_type: str, **attributes) -> None:
        """
        그래프에 개체(Entity) 노드 추가
        
        Args:
            entity_id (str): 개체 고유 ID
            entity_text (str): 개체 텍스트
            entity_type (str): 개체 유형(PERSON, ORG, GPE 등)
            **attributes: 추가 속성
        """
        # 이미 있는 노드인지 확인
        if self.graph.has_node(entity_id):
            # 기존 노드 업데이트
            for key, value in attributes.items():
                self.graph.nodes[entity_id][key] = value
        else:
            # 새 노드 추가
            attributes.update({
                'text': entity_text,
                'type': entity_type,
                'created_at': datetime.now().isoformat()
            })
            self.graph.add_node(entity_id, **attributes)
    
    def add_relation(self, source_id: str, target_id: str, relation_type: str, **attributes) -> None:
        """
        그래프에 관계(Relation) 엣지 추가
        
        Args:
            source_id (str): 출발 개체 ID
            target_id (str): 도착 개체 ID
            relation_type (str): 관계 유형
            **attributes: 추가 속성
        """
        attributes.update({
            'type': relation_type,
            'created_at': datetime.now().isoformat()
        })
        self.graph.add_edge(source_id, target_id, **attributes)
    
    def add_entity_from_dict(self, entity: Dict[str, Any], entity_id: Optional[str] = None) -> str:
        """
        딕셔너리 형태의 개체 정보로 노드 추가
        
        Args:
            entity (Dict[str, Any]): 개체 정보 딕셔너리
            entity_id (str, optional): 개체 ID (없으면 자동 생성)
            
        Returns:
            str: 개체 ID
        """
        if entity_id is None:
            # 고유 ID 생성 (텍스트와 유형으로)
            entity_id = f"{entity['text']}_{entity['label']}".lower().replace(' ', '_')
        
        self.add_entity(
            entity_id=entity_id,
            entity_text=entity['text'],
            entity_type=entity['label'],
            **{k: v for k, v in entity.items() if k not in ['text', 'label']}
        )
        
        return entity_id
    
    def add_mentions_relation(self, article_id: str, entity_ids: List[str], source: str) -> None:
        """
        기사와 개체 사이에 '언급됨' 관계 추가
        
        Args:
            article_id (str): 기사 ID
            entity_ids (List[str]): 개체 ID 목록
            source (str): 기사의 어느 부분에서 언급되었는지(title/content)
        """
        for entity_id in entity_ids:
            self.add_relation(
                source_id=article_id,
                target_id=entity_id,
                relation_type='mentions',
                source=source
            )
    
    def add_co_occurrence_relations(self, entity_ids: List[str], context_id: str, weight: float = 1.0) -> None:
        """
        동일 문맥(기사)에 등장한 개체들 사이에 '함께 등장' 관계 추가
        
        Args:
            entity_ids (List[str]): 개체 ID 목록
            context_id (str): 문맥(기사) ID
            weight (float): 관계 가중치
        """
        # 모든 가능한 개체 페어에 대해 관계 추가
        for i in range(len(entity_ids)):
            for j in range(i + 1, len(entity_ids)):
                source_id = entity_ids[i]
                target_id = entity_ids[j]
                
                # 양방향 관계 추가
                self.add_relation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type='co_occurs_with',
                    context=context_id,
                    weight=weight
                )
                
                self.add_relation(
                    source_id=target_id,
                    target_id=source_id,
                    relation_type='co_occurs_with',
                    context=context_id,
                    weight=weight
                )
    
    def build_from_articles(self, articles: List[Dict[str, Any]]) -> None:
        """
        기사 목록에서 지식 그래프 구축
        
        Args:
            articles (List[Dict[str, Any]]): 개체가 포함된 기사 목록
        """
        for article in articles:
            if 'entities' not in article:
                logger.warning("기사에 개체 정보가 없습니다. 개체 추출이 필요합니다.")
                continue
            
            # 기사 노드 추가
            article_id = article.get('id', f"article_{hash(article['title'])}")
            self.add_entity(
                entity_id=article_id,
                entity_text=article['title'],
                entity_type='ARTICLE',
                link=article.get('link', ''),
                published=article.get('published', '')
            )
            
            # 기사에서 추출된 개체 처리
            title_entity_ids = []
            content_entity_ids = []
            
            for entity in article['entities']:
                # 개체 노드 추가
                entity_id = self.add_entity_from_dict(entity)
                
                # 개체가 제목에서 추출된 경우
                if entity.get('source') == 'title':
                    title_entity_ids.append(entity_id)
                # 개체가 본문에서 추출된 경우
                elif entity.get('source') == 'content':
                    content_entity_ids.append(entity_id)
            
            # 기사-개체 간 '언급됨' 관계 추가
            if title_entity_ids:
                self.add_mentions_relation(article_id, title_entity_ids, 'title')
            
            if content_entity_ids:
                self.add_mentions_relation(article_id, content_entity_ids, 'content')
            
            # 동일 기사 내 개체 간 '함께 등장' 관계 추가
            all_entity_ids = title_entity_ids + content_entity_ids
            if len(all_entity_ids) > 1:
                self.add_co_occurrence_relations(all_entity_ids, article_id)
        
        logger.info(f"그래프 구축 완료: {self.graph.number_of_nodes()}개 노드, {self.graph.number_of_edges()}개 엣지")
    
    def get_entity_by_text(self, text: str, entity_type: Optional[str] = None) -> List[str]:
        """
        텍스트로 개체 노드 검색
        
        Args:
            text (str): 검색할 개체 텍스트
            entity_type (str, optional): 개체 유형 필터
            
        Returns:
            List[str]: 일치하는 개체 ID 목록
        """
        result = []
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get('text', '').lower() == text.lower():
                if entity_type is None or attrs.get('type') == entity_type:
                    result.append(node_id)
        
        return result
    
    def get_relations(self, entity_id: str, relation_type: Optional[str] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        특정 개체의 관계 조회
        
        Args:
            entity_id (str): 개체 ID
            relation_type (str, optional): 관계 유형 필터
            
        Returns:
            List[Tuple[str, str, Dict[str, Any]]]: (source_id, target_id, 속성) 튜플 목록
        """
        result = []
        
        # 출발 관계 조회
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            if relation_type is None or data.get('type') == relation_type:
                result.append((entity_id, target, data))
        
        # 도착 관계 조회
        for source, _, data in self.graph.in_edges(entity_id, data=True):
            if relation_type is None or data.get('type') == relation_type:
                result.append((source, entity_id, data))
        
        return result
    
    def get_related_entities(self, entity_id: str, relation_type: Optional[str] = None) -> List[str]:
        """
        특정 개체와 관련된 다른 개체들 조회
        
        Args:
            entity_id (str): 개체 ID
            relation_type (str, optional): 관계 유형 필터
            
        Returns:
            List[str]: 관련 개체 ID 목록
        """
        relations = self.get_relations(entity_id, relation_type)
        related_entities = set()
        
        for source, target, _ in relations:
            if source == entity_id:
                related_entities.add(target)
            else:
                related_entities.add(source)
        
        return list(related_entities)
    
    def visualize(self, output_path: Optional[str] = None, entity_types: Optional[List[str]] = None) -> None:
        """
        그래프 시각화
        
        Args:
            output_path (str, optional): 이미지 저장 경로
            entity_types (List[str], optional): 표시할 개체 유형 목록
        """
        # 표시할 노드 필터링
        if entity_types:
            nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get('type') in entity_types]
            graph = self.graph.subgraph(nodes)
        else:
            graph = self.graph
            
        # 노드 색상 설정
        node_colors = []
        for _, attr in graph.nodes(data=True):
            if attr.get('type') == 'PERSON':
                node_colors.append('lightblue')
            elif attr.get('type') == 'ORG':
                node_colors.append('lightgreen')
            elif attr.get('type') == 'GPE':
                node_colors.append('salmon')
            elif attr.get('type') == 'ARTICLE':
                node_colors.append('yellow')
            else:
                node_colors.append('lightgray')
        
        # 그래프 그리기
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(graph, seed=42)
        
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, alpha=0.8, node_size=500)
        nx.draw_networkx_edges(graph, pos, alpha=0.5, arrows=True)
        
        # 노드 라벨
        labels = {n: attr.get('text', n)[:20] for n, attr in graph.nodes(data=True)}
        nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
        
        plt.axis('off')
        plt.tight_layout()
        
        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"그래프 이미지가 {output_path}에 저장되었습니다.")
        else:
            plt.show()
    
    def save(self, file_path: str) -> None:
        """
        그래프를 파일로 저장
        
        Args:
            file_path (str): 저장할 파일 경로
        """
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 그래프 저장
        if file_path.endswith('.graphml'):
            nx.write_graphml(self.graph, file_path)
        elif file_path.endswith('.gexf'):
            nx.write_gexf(self.graph, file_path)
        elif file_path.endswith('.json'):
            # JSON 형식으로 저장 (NetworkX의 node_link_data 사용)
            data = nx.node_link_data(self.graph)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            # 기본은 GraphML 형식
            nx.write_graphml(self.graph, file_path)
        
        logger.info(f"그래프가 {file_path}에 저장되었습니다.")
    
    @classmethod
    def load(cls, file_path: str) -> 'KnowledgeGraph':
        """
        파일에서 그래프 로드
        
        Args:
            file_path (str): 그래프 파일 경로
            
        Returns:
            KnowledgeGraph: 로드된 그래프 객체
        """
        graph_obj = cls()
        
        if file_path.endswith('.graphml'):
            graph_obj.graph = nx.read_graphml(file_path)
        elif file_path.endswith('.gexf'):
            graph_obj.graph = nx.read_gexf(file_path)
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            graph_obj.graph = nx.node_link_graph(data)
        else:
            # 기본은 GraphML 형식
            graph_obj.graph = nx.read_graphml(file_path)
        
        logger.info(f"그래프를 {file_path}에서 로드했습니다: {graph_obj.graph.number_of_nodes()}개 노드, {graph_obj.graph.number_of_edges()}개 엣지")
        return graph_obj

def build_graph_from_entities(articles_with_entities: List[Dict[str, Any]]) -> KnowledgeGraph:
    """
    개체가 포함된 기사 목록에서 지식 그래프 생성
    
    Args:
        articles_with_entities (List[Dict[str, Any]]): 개체가 포함된 기사 목록
        
    Returns:
        KnowledgeGraph: 생성된 지식 그래프
    """
    graph = KnowledgeGraph()
    graph.build_from_articles(articles_with_entities)
    return graph

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='개체 정보로 지식 그래프를 구축합니다.')
    parser.add_argument('entities_json', help='개체 정보가 포함된 JSON 파일 경로')
    parser.add_argument('-o', '--output', help='그래프를 저장할 파일 경로 (.graphml, .gexf, .json)')
    parser.add_argument('-v', '--visualize', help='그래프 시각화 이미지 저장 경로')
    parser.add_argument('-t', '--types', nargs='+', help='시각화할 개체 유형 목록 (예: PERSON ORG GPE)')
    
    args = parser.parse_args()
    
    try:
        # JSON 파일 로드
        with open(args.entities_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        articles = data.get('articles', [])
        logger.info(f"{len(articles)}개의 기사로 그래프 구축을 시작합니다.")
        
        # 그래프 구축
        graph = build_graph_from_entities(articles)
        
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
        
        # 그래프 저장
        if args.output:
            graph.save(args.output)
        
        # 그래프 시각화
        if args.visualize:
            graph.visualize(args.visualize, args.types)
    
    except Exception as e:
        logger.error(f"그래프 구축 중 오류 발생: {str(e)}")
        print(f"오류 발생: {str(e)}") 