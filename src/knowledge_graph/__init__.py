"""
Knowledge Graph Module

이 패키지는 텍스트에서 개체(Entity)를 추출하고 지식 그래프를 구축하는 기능을 제공합니다.
"""

from .entity_extractor import EntityExtractor, extract_entities_from_texts
from .graph_builder import KnowledgeGraph, build_graph_from_entities

__all__ = [
    'EntityExtractor', 'extract_entities_from_texts',
    'KnowledgeGraph', 'build_graph_from_entities'
]
