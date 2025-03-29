"""
벡터 검색 시각화 모듈.
ChromaDB를 사용하여 벡터 검색을 수행하고 결과를 시각화합니다.
"""

import chromadb
import json
from datetime import datetime
import os
import argparse

class VectorSearchVisualizer:
    def __init__(self, output_dir="search_results"):
        """
        벡터 검색 시각화 클래스 초기화
        
        Args:
            output_dir (str): 검색 결과 저장 디렉토리
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 현재 시간 (파일명용)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ChromaDB 클라이언트
        self.client = None
        self.collection = None
        
    def initialize_client(self):
        """ChromaDB 클라이언트 초기화"""
        self.client = chromadb.Client()
        print("ChromaDB 클라이언트 초기화 완료")
        
        # 사용 가능한 컬렉션 목록 출력
        collections = self.client.list_collections()
        collection_names = [col.name for col in collections]
        print(f"사용 가능한 컬렉션: {collection_names}")
        
        if not collection_names:
            raise ValueError("사용 가능한 컬렉션이 없습니다. ChromaDB에 데이터를 먼저 로드하세요.")
        
        # 첫 번째 컬렉션 사용 또는 사용자가 지정한 컬렉션 사용
        self.collection_name = collection_names[0]
        print(f"컬렉션 '{self.collection_name}' 사용")
        
        self.collection = self.client.get_collection(name=self.collection_name)
        
    def search_and_visualize(self, search_terms):
        """
        검색 수행 및 결과 시각화
        
        Args:
            search_terms (list): 검색어 목록
        """
        if self.client is None:
            self.initialize_client()
            
        # 검색 결과를 저장할 파일
        results_file = os.path.join(self.output_dir, f"vector_search_results_{self.timestamp}.json")
        all_results = {}
        
        print("\n=== 벡터 검색 결과 ===\n")
        
        for term in search_terms:
            print(f"\n검색어: '{term}'에 대한 결과")
            print("-" * 60)
            
            try:
                results = self.collection.query(
                    query_texts=[term],
                    n_results=5
                )
                
                # 콘솔에 결과 출력
                if results and results.get('documents') and results['documents'][0]:
                    formatted_results = []
                    
                    for i, (doc, metadata, distance) in enumerate(zip(
                        results.get('documents', [[]])[0],
                        results.get('metadatas', [[]])[0],
                        results.get('distances', [[]])[0]
                    )):
                        similarity = 1 - distance  # 거리를 유사도로 변환
                        
                        # 콘솔 출력용 포맷
                        print(f"{i+1}. {metadata.get('title', 'No Title')}")
                        print(f"   링크: {metadata.get('link', 'No Link')}")
                        print(f"   유사도: {similarity:.4f}")
                        content_preview = doc[:150] + "..." if len(doc) > 150 else doc
                        print(f"   내용: {content_preview}")
                        print()
                        
                        # JSON용 포맷
                        formatted_results.append({
                            "rank": i+1,
                            "similarity": similarity,
                            "title": metadata.get("title", ""),
                            "link": metadata.get("link", ""),
                            "content": content_preview,
                            "published": metadata.get("published", "")
                        })
                    
                    # 전체 결과에 추가
                    all_results[term] = {
                        "results": formatted_results,
                        "count": len(formatted_results)
                    }
                else:
                    print(f"'{term}'에 대한 검색 결과가 없습니다.")
                    all_results[term] = {"results": [], "count": 0}
                    
            except Exception as e:
                print(f"검색 중 오류 발생: {str(e)}")
                all_results[term] = {"error": str(e)}
        
        # 결과를 JSON 파일로 저장
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n모든 검색 결과가 {results_file}에 저장되었습니다.")
        
        # 특정 키워드 결과만 별도 파일로 저장
        for term in search_terms:
            if term in all_results:
                term_file = os.path.join(self.output_dir, f"{term}_search_result.json")
                with open(term_file, "w", encoding="utf-8") as f:
                    json.dump(all_results[term], f, ensure_ascii=False, indent=2)
                print(f"'{term}' 검색 결과가 {term_file}에 별도 저장되었습니다.")
                
        print("\n스크린샷을 위해 이 터미널 화면을 캡처하세요!")

def main():
    parser = argparse.ArgumentParser(description='ChromaDB를 사용하여 벡터 검색 수행 및 시각화')
    parser.add_argument('--output_dir', default="search_results", help='검색 결과 저장 디렉토리')
    parser.add_argument('--search_terms', nargs='+', 
                        default=["기후변화", "climate change", "AI", "artificial intelligence"],
                        help='검색어 목록')
    
    args = parser.parse_args()
    
    visualizer = VectorSearchVisualizer(output_dir=args.output_dir)
    visualizer.search_and_visualize(args.search_terms)

if __name__ == "__main__":
    main() 