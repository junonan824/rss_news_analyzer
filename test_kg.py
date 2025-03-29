import json
import os
from src.knowledge_graph.entity_extractor import EntityExtractor
from src.knowledge_graph.graph_builder import KnowledgeGraph

def main():
    # 데이터 파일 경로
    data_file = "data/test_rss.json"
    output_dir = "data"
    
    # 출력 파일 경로
    entities_json_path = os.path.join(output_dir, 'entities.json')
    graph_file_path = os.path.join(output_dir, 'knowledge_graph.json')
    graph_image_path = os.path.join(output_dir, 'knowledge_graph.png')
    
    # 데이터 로드
    print("Loading test data...")
    with open(data_file, 'r', encoding='utf-8') as f:
        rss_data = json.load(f)
    
    articles = rss_data.get('articles', [])
    print(f"Found {len(articles)} articles")
    
    # 개체 추출
    print("Extracting entities...")
    extractor = EntityExtractor(model_name='en_core_web_sm')
    articles_with_entities = extractor.extract_entities_batch(articles)
    
    # 추출된 개체 저장
    with open(entities_json_path, 'w', encoding='utf-8') as f:
        json.dump({"articles": articles_with_entities}, f, ensure_ascii=False, indent=2)
    
    print(f"Entities saved to {entities_json_path}")
    
    # 개체 결과 출력
    print("\nExtracted Entities:")
    for i, article in enumerate(articles_with_entities):
        print(f"Article {i+1}: {article['title']}")
        print(f"  Entities: {article.get('entities', [])}")
    
    # 그래프 생성
    print("\nBuilding knowledge graph...")
    graph = KnowledgeGraph()
    graph.build_from_articles(articles_with_entities)
    
    # 그래프 저장
    graph.save(graph_file_path)
    print(f"Graph saved to {graph_file_path}")
    
    # 그래프 통계 출력
    node_types = {}
    for _, attr in graph.graph.nodes(data=True):
        node_type = attr.get('type', 'UNKNOWN')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    edge_types = {}
    for _, _, attr in graph.graph.edges(data=True):
        edge_type = attr.get('type', 'UNKNOWN')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    print("\nGraph Statistics:")
    print(f"- Total nodes: {graph.graph.number_of_nodes()}")
    print(f"- Total edges: {graph.graph.number_of_edges()}")
    
    print("\nNode types:")
    for node_type, count in sorted(node_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {node_type}: {count}")
    
    print("\nEdge types:")
    for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {edge_type}: {count}")
    
    # 그래프 시각화
    print("\nVisualizing graph...")
    graph.visualize(graph_image_path)
    print(f"Graph visualization saved to {graph_image_path}")

if __name__ == "__main__":
    main() 