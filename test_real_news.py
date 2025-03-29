import os
import json
import logging
from src.rss_fetch.rss_fetch import process_rss_feed
from src.knowledge_graph.entity_extractor import EntityExtractor
from src.knowledge_graph.graph_builder import KnowledgeGraph

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # 설정
    rss_urls = [
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",  # 뉴욕 타임즈
        "https://feeds.bbci.co.uk/news/world/rss.xml",            # BBC
        "http://rss.cnn.com/rss/edition_world.rss",               # CNN
        "https://www.theguardian.com/world/rss"                   # 가디언
    ]
    
    output_dir = "data/real_news"
    os.makedirs(output_dir, exist_ok=True)
    
    # 선택한 RSS 피드 URL (0-3 중 선택)
    selected_feed = 1  # BBC 기사
    
    # RSS 피드 처리
    rss_json_path = os.path.join(output_dir, 'rss_data.json')
    url = rss_urls[selected_feed]
    
    print(f"1. RSS 피드 가져오기: {url}")
    process_rss_feed(url, rss_json_path)
    
    # 데이터 로드
    with open(rss_json_path, 'r', encoding='utf-8') as f:
        rss_data = json.load(f)
    
    articles = rss_data.get('articles', [])
    print(f"가져온 기사 수: {len(articles)}")
    
    # 처음 5개 기사 제목 출력
    print("\n처음 5개 기사 제목:")
    for i, article in enumerate(articles[:5]):
        print(f"{i+1}. {article.get('title', 'No Title')}")
    
    # 개체 추출
    print("\n2. 개체 추출 중...")
    entities_json_path = os.path.join(output_dir, 'entities.json')
    
    # 개체 추출 - 성능을 위해 처음 10개 기사만 사용
    limited_articles = articles[:10]
    print(f"처리할 기사 수: {len(limited_articles)}")
    
    extractor = EntityExtractor(model_name='en_core_web_sm')
    articles_with_entities = extractor.extract_entities_batch(limited_articles)
    
    # 추출된 개체 저장
    with open(entities_json_path, 'w', encoding='utf-8') as f:
        json.dump({"articles": articles_with_entities}, f, ensure_ascii=False, indent=2)
    
    print(f"개체 추출 결과가 {entities_json_path}에 저장되었습니다.")
    
    # 개체 추출 결과 출력 (처음 3개 기사)
    print("\n추출된 개체 (처음 3개 기사):")
    for i, article in enumerate(articles_with_entities[:3]):
        print(f"기사 {i+1}: {article['title']}")
        entities = article.get('entities', [])
        print(f"  추출된 개체 수: {len(entities)}")
        # 처음 5개 개체만 출력
        for j, entity in enumerate(entities[:5]):
            print(f"    - {entity['text']} ({entity['label']})")
        if len(entities) > 5:
            print(f"    ... 외 {len(entities)-5}개")
    
    # 그래프 생성
    print("\n3. 지식 그래프 생성 중...")
    graph_file_path = os.path.join(output_dir, 'knowledge_graph.json')
    graph_image_path = os.path.join(output_dir, 'knowledge_graph.png')
    
    graph = KnowledgeGraph()
    graph.build_from_articles(articles_with_entities)
    
    # 그래프 저장
    graph.save(graph_file_path)
    print(f"그래프가 {graph_file_path}에 저장되었습니다.")
    
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
    
    # 그래프 시각화
    print("\n4. 그래프 시각화 중...")
    graph.visualize(graph_image_path)
    print(f"그래프 시각화가 {graph_image_path}에 저장되었습니다.")

if __name__ == "__main__":
    main() 