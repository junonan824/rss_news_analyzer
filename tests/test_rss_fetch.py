"""
RSS 피드 모듈 테스트
"""
import os
import json
import pytest
from src.rss_fetch import fetch_rss, extract_article_data, save_to_json, process_rss_feed

# 테스트용 RSS 피드 URL (실제 작동하는 피드)
TEST_FEED_URL = "https://news.google.com/rss"
TEST_OUTPUT_FILE = "tests/test_output.json"

def test_fetch_rss():
    """RSS 피드를 성공적으로 가져오는지 테스트합니다."""
    entries = fetch_rss(TEST_FEED_URL)
    assert len(entries) > 0
    assert 'title' in entries[0]
    assert 'link' in entries[0]

def test_extract_article_data():
    """기사 데이터를 정상적으로 추출하는지 테스트합니다."""
    entries = fetch_rss(TEST_FEED_URL)
    articles = extract_article_data(entries)
    
    assert len(articles) > 0
    assert 'title' in articles[0]
    assert 'link' in articles[0]
    assert 'published' in articles[0]
    assert 'content' in articles[0]

def test_save_to_json():
    """데이터를 JSON 파일로 저장하는 기능을 테스트합니다."""
    # 테스트용 데이터
    test_data = [{'title': 'Test Article', 'content': 'Test Content'}]
    
    # 파일 저장
    file_path = save_to_json(test_data, TEST_OUTPUT_FILE)
    
    # 파일이 존재하는지 확인
    assert os.path.exists(file_path)
    
    # 파일을 다시 로드하여 내용 확인
    with open(file_path, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
    
    assert 'articles' in loaded_data
    assert len(loaded_data['articles']) == 1
    assert loaded_data['articles'][0]['title'] == 'Test Article'
    
    # 테스트 후 파일 정리
    os.remove(file_path)

def test_process_rss_feed():
    """전체 프로세스가 정상적으로 동작하는지 테스트합니다."""
    file_path = process_rss_feed(TEST_FEED_URL, TEST_OUTPUT_FILE)
    
    # 파일이 존재하는지 확인
    assert os.path.exists(file_path)
    
    # 파일을 다시 로드하여 내용 확인
    with open(file_path, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
    
    assert 'articles' in loaded_data
    assert len(loaded_data['articles']) > 0
    
    # 테스트 후 파일 정리
    os.remove(file_path) 