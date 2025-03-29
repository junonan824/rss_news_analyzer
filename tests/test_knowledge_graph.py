"""
Knowledge Graph 모듈 테스트
"""
import os
import json
import pytest
import networkx as nx
from src.knowledge_graph.entity_extractor import EntityExtractor
from src.knowledge_graph.graph_builder import KnowledgeGraph

# 테스트용 텍스트
TEST_TEXTS = [
    "애플은 아이폰15를 출시했으며, 팀 쿡 CEO는 이번 제품이 혁신적이라고 말했다.",
    "마이크로소프트의 CEO 사티아 나델라는 AI 기술에 대규모 투자를 발표했다.",
    "구글과 메타는 인공지능 기술 개발에서 경쟁하고 있다."
]

def test_entity_extractor():
    """개체 추출 기능 테스트"""
    try:
        extractor = EntityExtractor()
        
        # 첫 번째 텍스트에서 개체 추출
        entities = extractor.extract_entities(TEST_TEXTS[0])
        
        # 개체가 추출되었는지 확인
        assert len(entities) > 0
        
        # 개체 구조 확인
        assert 'text' in entities[0]
        assert 'label' in entities[0]
        assert 'start_char' in entities[0]
        assert 'end_char' in entities[0]
        
        # 특정 개체가 추출되었는지 확인 (모델에 따라 결과가 다를 수 있음)
        entity_texts = [entity['text'] for entity in entities]
        assert any('애플' in text for text in entity_texts) or any('아이폰' in text for text in entity_texts)
        
    except Exception as e:
        if "E050" in str(e):  # spaCy 모델이 설치되지 않은 경우
            pytest.skip("spaCy 모델이 설치되지 않았습니다. 설치 방법: python -m spacy download en_core_web_sm")
        else:
            raise

def test_knowledge_graph():
    """지식 그래프 기능 테스트"""
    # 그래프 초기화
    graph = KnowledgeGraph()
    
    # 노드 추가
    graph.add_entity("apple", "Apple Inc.", "ORG", industry="Technology")
    graph.add_entity("iphone", "iPhone", "PRODUCT", manufacturer="Apple")
    graph.add_entity("tim_cook", "Tim Cook", "PERSON", role="CEO")
    
    # 관계 추가
    graph.add_relation("apple", "iphone", "manufactures")
    graph.add_relation("tim_cook", "apple", "works_for")
    
    # 노드와 엣지 수 확인
    assert graph.graph.number_of_nodes() == 3
    assert graph.graph.number_of_edges() == 2
    
    # 노드 속성 확인
    assert graph.graph.nodes["apple"]["text"] == "Apple Inc."
    assert graph.graph.nodes["apple"]["type"] == "ORG"
    assert graph.graph.nodes["apple"]["industry"] == "Technology"
    
    # 관계 조회
    relations = graph.get_relations("apple")
    assert len(relations) == 2
    
    # 관련 개체 조회
    related = graph.get_related_entities("apple")
    assert "iphone" in related
    assert "tim_cook" in related
    
    # 텍스트로 개체 검색
    found = graph.get_entity_by_text("Apple Inc.")
    assert "apple" in found

def test_graph_from_entities(tmp_path):
    """개체에서 그래프 생성 테스트"""
    # 테스트용 기사 데이터 생성
    articles = [
        {
            "title": "테크 기업 뉴스",
            "content": "애플은 아이폰15를 출시했으며, 팀 쿡 CEO는 이번 제품이 혁신적이라고 말했다.",
            "entities": [
                {"text": "애플", "label": "ORG", "source": "content"},
                {"text": "아이폰15", "label": "PRODUCT", "source": "content"},
                {"text": "팀 쿡", "label": "PERSON", "source": "content"}
            ]
        },
        {
            "title": "인공지능 발전",
            "content": "구글과 메타는 인공지능 기술 개발에서 경쟁하고 있다.",
            "entities": [
                {"text": "구글", "label": "ORG", "source": "content"},
                {"text": "메타", "label": "ORG", "source": "content"},
                {"text": "인공지능", "label": "TECHNOLOGY", "source": "content"}
            ]
        }
    ]
    
    # 그래프 생성
    graph = KnowledgeGraph()
    graph.build_from_articles(articles)
    
    # 그래프 검증
    node_count = graph.graph.number_of_nodes()
    edge_count = graph.graph.number_of_edges()
    
    # 기사 2개, 개체 6개 = 총 8개 노드
    assert node_count >= 8
    
    # 기사-개체 관계 6개, 개체 간 관계 최소 2개
    assert edge_count >= 8
    
    # 그래프 저장 및 로드
    graph_path = os.path.join(tmp_path, "test_graph.json")
    graph.save(graph_path)
    
    # 파일이 생성되었는지 확인
    assert os.path.exists(graph_path)
    
    # 그래프 로드
    loaded_graph = KnowledgeGraph.load(graph_path)
    
    # 로드된 그래프가 원본과 동일한지 확인
    assert loaded_graph.graph.number_of_nodes() == node_count
    assert loaded_graph.graph.number_of_edges() == edge_count

def test_integration():
    """개체 추출과 그래프 생성 통합 테스트"""
    try:
        # 개체 추출기 초기화
        extractor = EntityExtractor()
        
        # 테스트 기사 데이터
        articles = [
            {
                "title": "테크 뉴스",
                "content": TEST_TEXTS[0]
            },
            {
                "title": "AI 투자",
                "content": TEST_TEXTS[1]
            }
        ]
        
        # 개체 추출
        articles_with_entities = extractor.extract_entities_batch(articles)
        
        # 개체가 추출되었는지 확인
        assert 'entities' in articles_with_entities[0]
        assert len(articles_with_entities[0]['entities']) > 0
        
        # 그래프 생성
        graph = KnowledgeGraph()
        graph.build_from_articles(articles_with_entities)
        
        # 그래프가 생성되었는지 확인
        assert graph.graph.number_of_nodes() > 0
        assert graph.graph.number_of_edges() > 0
        
    except Exception as e:
        if "E050" in str(e):  # spaCy 모델이 설치되지 않은 경우
            pytest.skip("spaCy 모델이 설치되지 않았습니다. 설치 방법: python -m spacy download en_core_web_sm")
        else:
            raise

if __name__ == "__main__":
    # 테스트 실행
    pytest.main(["-xvs", __file__]) 