"""
시각화 실행 스크립트.
임베딩 시각화와 벡터 검색 시각화를 실행합니다.
"""

import sys
import os
import argparse

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.embedding_visualizer import EmbeddingVisualizer
from src.visualization.search_visualizer import VectorSearchVisualizer

def main():
    parser = argparse.ArgumentParser(description='RSS 뉴스 분석기 시각화 실행')
    parser.add_argument('--type', choices=['embedding', 'search', 'all'], default='all',
                      help='실행할 시각화 유형 (embedding, search, all)')
    parser.add_argument('--embedding_file', default="data/embeddings_embeddings.npy", 
                      help='임베딩 파일 경로')
    parser.add_argument('--articles_file', default="data/rss_data.json", 
                      help='기사 데이터 파일 경로')
    parser.add_argument('--output_dir', default="visualization_results", 
                      help='시각화 결과 저장 디렉토리')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 임베딩 시각화
    if args.type in ['embedding', 'all']:
        print("\n=== 임베딩 시각화 실행 ===\n")
        embedding_output = os.path.join(args.output_dir, "embedding_visualization.png")
        visualizer = EmbeddingVisualizer(
            embedding_file=args.embedding_file,
            articles_file=args.articles_file,
            output_file=embedding_output
        )
        visualizer.visualize()
    
    # 벡터 검색 시각화
    if args.type in ['search', 'all']:
        print("\n=== 벡터 검색 시각화 실행 ===\n")
        search_output_dir = os.path.join(args.output_dir, "search_results")
        visualizer = VectorSearchVisualizer(output_dir=search_output_dir)
        visualizer.search_and_visualize([
            "기후변화", "climate change", "global warming", "AI", "artificial intelligence"
        ])
    
    print(f"\n모든 시각화 결과가 {args.output_dir} 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    main()
