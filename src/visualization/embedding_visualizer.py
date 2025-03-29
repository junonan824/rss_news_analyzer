"""
임베딩 시각화 모듈.
t-SNE를 사용하여 임베딩 벡터를 2D로 시각화합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import json
from datetime import datetime
import argparse

class EmbeddingVisualizer:
    def __init__(self, embedding_file, articles_file, output_file="embedding_result.png"):
        """
        임베딩 시각화 클래스 초기화
        
        Args:
            embedding_file (str): 임베딩 파일 경로
            articles_file (str): 기사 데이터 파일 경로
            output_file (str): 출력 이미지 파일 경로
        """
        self.embedding_file = embedding_file
        self.articles_file = articles_file
        self.output_file = output_file
        
        # 데이터 속성
        self.embeddings = None
        self.article_titles = []
        
    def load_data(self):
        """임베딩 및 기사 데이터 로드"""
        # 임베딩 데이터 로드
        if os.path.exists(self.embedding_file):
            self.embeddings = np.load(self.embedding_file)
            print(f"임베딩 데이터 로드 완료: 형태 {self.embeddings.shape}")
        else:
            raise FileNotFoundError(f"임베딩 파일 {self.embedding_file}을 찾을 수 없습니다.")
        
        # 기사 제목 로드
        if os.path.exists(self.articles_file):
            with open(self.articles_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for article in data.get('articles', []):
                    self.article_titles.append(article.get('title', 'No Title'))
            print(f"{len(self.article_titles)}개의 기사 제목을 로드했습니다.")
        else:
            self.article_titles = [f"문서 {i+1}" for i in range(len(self.embeddings))]
            print("기사 파일을 찾을 수 없어 기본 레이블을 사용합니다.")
            
        print(f"임베딩 데이터 개수: {len(self.embeddings)}, 기사 제목 개수: {len(self.article_titles)}")
        
        # 데이터 수 맞추기
        self._align_data_counts()
        
    def _align_data_counts(self):
        """임베딩 데이터와 기사 제목 수 일치시키기"""
        min_count = min(len(self.embeddings), len(self.article_titles))
        self.embeddings = self.embeddings[:min_count]
        if len(self.article_titles) > min_count:
            self.article_titles = self.article_titles[:min_count]
        print(f"조정 후: 임베딩 데이터 개수: {len(self.embeddings)}, 기사 제목 개수: {len(self.article_titles)}")
        
    def visualize(self):
        """t-SNE로 임베딩 시각화"""
        if self.embeddings is None:
            self.load_data()
            
        n_samples = len(self.embeddings)
        perplexity_value = min(30, n_samples - 1)  # 데이터 포인트가 적을 경우 조정
        print(f"t-SNE 차원 축소 시작 (perplexity={perplexity_value})...")
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
        embeddings_2d = tsne.fit_transform(self.embeddings)
        print("t-SNE 차원 축소 완료")
        
        # 시각화
        plt.figure(figsize=(14, 10))
        
        # 컬러맵 설정
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=range(n_samples), cmap='viridis', 
                             alpha=0.8, s=100)
        
        # 일부 포인트에 레이블 표시
        max_labels = min(12, n_samples)  # 최대 12개 레이블 표시
        indices = np.linspace(0, n_samples-1, max_labels).astype(int)
        
        for idx in indices:
            # 제목이 너무 길면 자르기
            title = self.article_titles[idx]
            if len(title) > 25:
                title = title[:22] + "..."
                
            plt.annotate(f"{idx+1}: {title}", 
                        xy=(embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, ha='left', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # 제목 및 레이블
        current_time = datetime.now().strftime("%Y-%m-%d")
        plt.title(f't-SNE를 이용한 문서 임베딩 시각화 ({n_samples}개 문서, {current_time})')
        plt.xlabel('t-SNE 차원 1')
        plt.ylabel('t-SNE 차원 2')
        plt.colorbar(scatter, label='문서 인덱스')
        plt.tight_layout()
        
        # 저장
        plt.savefig(self.output_file, dpi=300, bbox_inches='tight')
        print(f"시각화 이미지가 {self.output_file}에 저장되었습니다.")
        
        # 보여주기 (선택사항)
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='t-SNE를 사용하여 임베딩 벡터 시각화')
    parser.add_argument('--embedding', default="data/embeddings_embeddings.npy", help='임베딩 파일 경로')
    parser.add_argument('--articles', default="data/rss_data.json", help='기사 데이터 파일 경로')
    parser.add_argument('--output', default="embedding_result.png", help='출력 이미지 파일 경로')
    
    args = parser.parse_args()
    
    visualizer = EmbeddingVisualizer(
        embedding_file=args.embedding,
        articles_file=args.articles,
        output_file=args.output
    )
    
    visualizer.visualize()

if __name__ == "__main__":
    main() 