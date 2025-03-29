"""
임베딩 및 벡터 DB 모듈 테스트
"""
import os
import json
import pytest
import numpy as np
from src.embeddings.embedding import TextEmbedder, process_rss_data
from src.embeddings.vector_db import VectorDB, load_rss_to_vectordb

# 테스트용 텍스트
TEST_TEXTS = [
    "인공지능 기술이 빠르게 발전하고 있습니다.",
    "딥러닝 모델은 대량의 데이터로 학습됩니다.",
    "자연어 처리는 컴퓨터가 인간 언어를 이해하는 기술입니다.",
    "임베딩은 텍스트를 벡터 공간에 매핑하는 과정입니다.",
    "벡터 데이터베이스는 유사도 검색에 최적화되어 있습니다."
]

# 테스트용 메타데이터
TEST_METADATA = [
    {"source": "test", "category": "AI", "id": "1"},
    {"source": "test", "category": "ML", "id": "2"},
    {"source": "test", "category": "NLP", "id": "3"},
    {"source": "test", "category": "Embedding", "id": "4"},
    {"source": "test", "category": "Database", "id": "5"}
]

def test_text_embedder():
    """TextEmbedder 클래스 테스트"""
    # 임베더 초기화
    embedder = TextEmbedder()
    
    # 단일 텍스트 임베딩
    single_vector = embedder.embed_text(TEST_TEXTS[0])
    assert single_vector.shape[0] == embedder.vector_size
    
    # 다중 텍스트 임베딩
    vectors = embedder.embed_texts(TEST_TEXTS)
    assert vectors.shape == (len(TEST_TEXTS), embedder.vector_size)
    
    # 빈 텍스트 처리
    empty_vector = embedder.embed_text("")
    assert empty_vector.shape[0] == embedder.vector_size
    assert np.all(empty_vector == 0)

def test_vector_db():
    """VectorDB 클래스 테스트"""
    # 테스트용 DB 경로
    test_db_path = "tests/test_db"
    
    # 기존 테스트 DB 삭제
    import shutil
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)
    
    # VectorDB 초기화
    db = VectorDB(
        collection_name="test_collection",
        persist_directory=test_db_path
    )
    
    # 임베딩 생성
    embedder = TextEmbedder()
    embeddings = embedder.embed_texts(TEST_TEXTS)
    
    # DB에 텍스트 추가
    ids = db.add_texts(
        texts=TEST_TEXTS,
        metadatas=TEST_METADATA,
        embeddings=embeddings.tolist()
    )
    
    assert len(ids) == len(TEST_TEXTS)
    
    # 통계 확인
    stats = db.get_collection_stats()
    assert stats["count"] == len(TEST_TEXTS)
    
    # 검색 테스트
    query = "인공지능과 딥러닝"
    results = db.search(query, n_results=2)
    
    assert len(results["documents"][0]) == 2
    assert results["documents"][0][0] in TEST_TEXTS
    
    # 메타데이터 필터링 검색
    filtered_results = db.search(
        query="인공지능",
        n_results=3,
        where={"$and": [{"source": "test"}, {"category": "AI"}]}
    )
    
    if filtered_results["documents"][0]:
        assert filtered_results["metadatas"][0][0]["category"] == "AI"
    
    # 테스트 후 정리
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)

def test_integration(tmp_path):
    """임베딩과 벡터 DB 통합 테스트"""
    # 테스트용 RSS 데이터 생성
    test_articles = [
        {
            "title": "인공지능 최신 동향",
            "content": "인공지능 기술이 빠르게 발전하고 있으며, 많은 기업들이 도입을 검토하고 있습니다.",
            "link": "https://example.com/ai-trends",
            "published": "2023-01-01T00:00:00"
        },
        {
            "title": "딥러닝 모델의 발전",
            "content": "GPT, BERT 등 자연어 처리 모델의 성능이 크게 향상되었습니다.",
            "link": "https://example.com/deep-learning",
            "published": "2023-01-02T00:00:00"
        }
    ]
    
    test_data = {
        "timestamp": "2023-01-03T00:00:00",
        "articles": test_articles
    }
    
    # 테스트 데이터 저장
    test_json_path = os.path.join(tmp_path, "test_rss.json")
    with open(test_json_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # 임베딩 생성
    embedding_result = process_rss_data(
        json_path=test_json_path,
        output_path=os.path.join(tmp_path, "test_embeddings.json")
    )
    
    assert "embeddings" in embedding_result
    assert "texts" in embedding_result
    assert "metadata" in embedding_result
    
    # 벡터 DB에 로드
    db = load_rss_to_vectordb(
        json_path=test_json_path,
        collection_name="test_rss",
        persist_directory=os.path.join(tmp_path, "test_db"),
        embedding_path=os.path.join(tmp_path, "test_embeddings.json")
    )
    
    # 검색 테스트
    results = db.search("인공지능 기술")
    
    assert len(results["documents"][0]) > 0
    assert "인공지능" in results["documents"][0][0] 