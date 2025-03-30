"""
RSS Feed Fetcher Module

이 모듈은 RSS 피드 URL에서 기사 데이터를 가져와서 파싱하고,
JSON 파일 형태로 저장하는 기능을 제공합니다.
"""

import logging
import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import feedparser
import requests
from bs4 import BeautifulSoup
from newspaper import Article
from tqdm import tqdm
from urllib.parse import urlparse, urljoin

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 기본 RSS 피드 소스 목록
DEFAULT_RSS_FEEDS = {
    "news": [
        "https://news.google.com/rss",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://feeds.bbci.co.uk/news/world/rss.xml",
        "https://www.yahoo.com/news/rss/world"
    ],
    "tech": [
        "https://news.google.com/rss/search?q=technology&hl=en-US&gl=US&ceid=US:en",
        "https://techcrunch.com/feed/",
        "https://www.wired.com/feed/rss",
        "https://feeds.arstechnica.com/arstechnica/technology-lab"
    ],
    "science": [
        "https://news.google.com/rss/search?q=science&hl=en-US&gl=US&ceid=US:en",
        "https://www.sciencedaily.com/rss/all.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml"
    ],
    "business": [
        "https://news.google.com/rss/search?q=business&hl=en-US&gl=US&ceid=US:en",
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"
    ],
    "health": [
        "https://news.google.com/rss/search?q=health&hl=en-US&gl=US&ceid=US:en",
        "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml",
        "https://www.who.int/rss-feeds/news-english.xml"
    ],
    "disaster": [
        "https://news.google.com/rss/search?q=disaster+OR+earthquake+OR+hurricane&hl=en-US&gl=US&ceid=US:en"
    ],
    "general": [
        "https://news.google.com/rss",
        "https://www.reddit.com/r/news/.rss"
    ],
    # 한국어 RSS 피드
    "korean": [
        "https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko",
        "https://news.google.com/rss/search?q=최신뉴스&hl=ko&gl=KR&ceid=KR:ko"
    ],
    "korean_tech": [
        "https://news.google.com/rss/search?q=기술&hl=ko&gl=KR&ceid=KR:ko"
    ],
    "korean_science": [
        "https://news.google.com/rss/search?q=과학&hl=ko&gl=KR&ceid=KR:ko"
    ],
    "korean_disaster": [
        "https://news.google.com/rss/search?q=지진%20재난&hl=ko&gl=KR&ceid=KR:ko"
    ]
}

# 키워드 매핑
KEYWORD_TO_CATEGORY = {
    # 영어 키워드
    "tech": "tech",
    "technology": "tech",
    "digital": "tech",
    "software": "tech",
    "hardware": "tech",
    "ai": "tech",
    "artificial intelligence": "tech",
    
    "science": "science",
    "research": "science",
    "discovery": "science",
    "space": "science",
    
    "business": "business",
    "economy": "business",
    "finance": "business",
    "market": "business",
    "stocks": "business",
    
    "health": "health",
    "medical": "health",
    "medicine": "health",
    "disease": "health",
    "covid": "health",
    
    "disaster": "disaster",
    "earthquake": "disaster",
    "hurricane": "disaster",
    "storm": "disaster",
    "flooding": "disaster",
    "wildfire": "disaster",
    
    # 한국어 키워드
    "기술": "korean_tech",
    "테크": "korean_tech",
    "디지털": "korean_tech",
    "소프트웨어": "korean_tech",
    "하드웨어": "korean_tech",
    "인공지능": "korean_tech",
    
    "과학": "korean_science",
    "연구": "korean_science",
    "발견": "korean_science",
    "우주": "korean_science",
    
    "비즈니스": "business",
    "경제": "business",
    "금융": "business",
    "시장": "business",
    "주식": "business",
    
    "건강": "health",
    "의학": "health",
    "질병": "health",
    "코로나": "health",
    
    "재난": "korean_disaster",
    "지진": "korean_disaster",
    "태풍": "korean_disaster",
    "폭풍": "korean_disaster",
    "홍수": "korean_disaster",
    "산불": "korean_disaster",
    
    # 특정 지역 키워드
    "인도네시아": "disaster",
    "수마트라": "disaster",
    "자바": "disaster",
    
    # 일반 뉴스
    "news": "news",
    "뉴스": "korean",
    "최신": "korean"
}

def find_relevant_rss_feeds(query: str) -> List[str]:
    """
    쿼리와 관련된 RSS 피드 URL 목록을 반환
    
    Args:
        query: 검색 쿼리
        
    Returns:
        관련 RSS 피드 URL 목록
    """
    relevant_feeds = []
    query_lower = query.lower()
    matched_categories = set()
    
    # 한국어 -> 영어 키워드 번역 매핑
    kr_to_en_mapping = {
        "인도네시아": "indonesia",
        "지진": "earthquake",
        "태풍": "typhoon",
        "홍수": "flood",
        "재난": "disaster",
        "뉴스": "news"
    }
    
    # 영어 검색어 생성
    english_query = query_lower
    for kr_word, en_word in kr_to_en_mapping.items():
        if kr_word in query_lower:
            english_query = english_query.replace(kr_word, en_word)
    
    # 1. 키워드 매칭으로 카테고리 찾기
    for keyword, category in KEYWORD_TO_CATEGORY.items():
        if keyword.lower() in query_lower:
            matched_categories.add(category)
    
    # 2. 카테고리별 피드 추가
    for category in matched_categories:
        if category in DEFAULT_RSS_FEEDS:
            relevant_feeds.extend(DEFAULT_RSS_FEEDS[category])
    
    # 3. 카테고리 매칭이 없으면 한영 키워드로 Google News 검색 피드 생성
    if not matched_categories:
        # 영어 쿼리로 가정
        encoded_query = requests.utils.quote(query)
        google_news_feed = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        relevant_feeds.append(google_news_feed)
        
        # 한국어 피드 추가 (기본 포함)
        relevant_feeds.append("https://news.google.com/rss?hl=ko&gl=KR&ceid=KR:ko")
        
        # 한국어 버전으로도 검색해보기
        google_news_feed_ko = f"https://news.google.com/rss/search?q={encoded_query}&hl=ko&gl=KR&ceid=KR:ko"
        relevant_feeds.append(google_news_feed_ko)
    
    # 4. 한국어 키워드를 영어로 변환한 검색 쿼리 추가
    if english_query != query_lower:
        encoded_en_query = requests.utils.quote(english_query)
        google_news_feed_en = f"https://news.google.com/rss/search?q={encoded_en_query}&hl=en-US&gl=US&ceid=US:en"
        relevant_feeds.append(google_news_feed_en)
    
    # 5. 최종 피드 반환 (중복 제거)
    return list(set(relevant_feeds))

def _extract_content_with_newspaper(url: str) -> Tuple[str, str, Dict]:
    """newspaper3k를 사용하여 기사 내용을 추출
    
    Args:
        url: 기사 URL
        
    Returns:
        (텍스트, 요약, 메타데이터) 튜플
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()  # 요약 및 키워드 추출
        
        # 메타데이터 추출
        metadata = {
            "title": article.title,
            "publish_date": article.publish_date.isoformat() if article.publish_date else None,
            "authors": article.authors,
            "keywords": article.keywords,
            "summary": article.summary,
            "url": url
        }
        
        return article.text, article.summary, metadata
    except Exception as e:
        logger.error(f"Error extracting content from {url}: {e}")
        return "", "", {"url": url, "error": str(e)}

def fetch_rss(url: str) -> list:
    """
    RSS 피드 URL에서 기사 목록을 가져옵니다.
    
    Args:
        url (str): RSS 피드 URL
        
    Returns:
        list: 기사 항목 리스트
        
    Raises:
        Exception: RSS 피드를 파싱하는 중 오류가 발생한 경우
    """
    try:
        logger.info(f"피드 가져오기 시작: {url}")
        feed = feedparser.parse(url)
        
        if feed.bozo:
            logger.warning(f"피드 파싱 경고: {feed.bozo_exception}")
        
        logger.info(f"가져온 항목 수: {len(feed.entries)}")
        return feed.entries
    except Exception as e:
        logger.error(f"RSS 피드 파싱 중 오류 발생: {str(e)}")
        raise

def clean_html(html_text: str) -> str:
    """
    HTML 태그를 제거하고 순수 텍스트만 추출합니다.
    
    Args:
        html_text (str): HTML 포맷의 텍스트
        
    Returns:
        str: HTML 태그가 제거된 순수 텍스트
    """
    if not html_text:
        return ""
    
    soup = BeautifulSoup(html_text, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

def normalize_date(date_str):
    """날짜 문자열을 ISO 형식으로 변환합니다.
    
    Args:
        date_str: 날짜 문자열 또는 time_struct
        
    Returns:
        ISO 형식의 날짜 문자열 또는 None
    """
    if not date_str:
        return None
        
    try:
        # time_struct 형식인 경우 (feedparser가 반환하는 형식)
        if hasattr(date_str, '__len__') and len(date_str) >= 6:
            date_obj = datetime(*date_str[:6])
            return date_obj.isoformat()
        
        # 이미 문자열인 경우 파싱 시도
        return datetime.fromisoformat(date_str).isoformat()
    except (ValueError, TypeError, AttributeError):
        try:
            # 다양한 형식 지원
            for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S"]:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    return date_obj.isoformat()
                except (ValueError, TypeError):
                    continue
        except Exception:
            pass
    
    return None

def extract_article_data(entries: list) -> list:
    """
    RSS 피드 항목에서 필요한 기사 데이터를 추출합니다.
    
    Args:
        entries (list): feedparser로 파싱한 RSS 항목 리스트
        
    Returns:
        list: 정제된 기사 데이터 리스트
    """
    articles = []
    
    for entry in entries:
        article = {
            'title': entry.get('title', ''),
            'link': entry.get('link', ''),
            'published': normalize_date(entry.get('published_parsed')),
            'summary': clean_html(entry.get('summary', '')),
            'content': clean_html(entry.get('content', [{'value': ''}])[0]['value']),
            'author': entry.get('author', ''),
            'tags': [tag.get('term', '') for tag in entry.get('tags', [])]
        }
        
        # 본문이 없는 경우 요약을 본문으로 사용
        if not article['content'] and article['summary']:
            article['content'] = article['summary']
            
        articles.append(article)
    
    logger.info(f"추출된 기사 항목 수: {len(articles)}")
    return articles

def save_to_json(articles: list, output_file: str = 'rss_data.json') -> str:
    """
    기사 데이터를 JSON 파일로 저장합니다.
    
    Args:
        articles (list): 저장할 기사 목록
        output_file (str, optional): 출력 파일 경로. 기본값은 'rss_data.json'.
        
    Returns:
        str: 저장된 파일 경로
    """
    # 디렉토리 생성 (필요한 경우)
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 메타데이터와 함께 JSON 저장
    data = {
        'articles': articles,
        'count': len(articles),
        'timestamp': datetime.now().isoformat(),
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"{len(articles)}개 기사를 {output_file}에 저장했습니다.")
        return os.path.abspath(output_file)
    except Exception as e:
        logger.error(f"JSON 저장 실패: {e}")
        # 대체 파일 이름 생성
        alt_file = f"backup_{int(time.time())}.json"
        logger.info(f"대체 파일에 저장 시도: {alt_file}")
        with open(alt_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return os.path.abspath(alt_file)

def process_rss_feed(url: str, output_file: str = 'rss_data.json') -> str:
    """
    RSS 피드 URL에서 데이터를 가져와 처리하고 JSON으로 저장하는 전체 과정을 수행합니다.
    
    Args:
        url (str): RSS 피드 URL
        output_file (str, optional): 저장할 파일 경로. 기본값은 'rss_data.json'
        
    Returns:
        str: 저장된 파일의 절대 경로
    """
    logger.info(f"RSS 피드 처리 시작: {url}")
    
    # RSS 피드 가져오기
    entries = fetch_rss(url)
    
    # 기사 데이터 추출
    articles = extract_article_data(entries)
    
    # JSON 파일로 저장
    file_path = save_to_json(articles, output_file)
    
    logger.info(f"RSS 피드 처리 완료: {url}")
    return file_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='RSS 피드를 가져와 JSON 파일로 저장합니다.')
    parser.add_argument('url', help='RSS 피드 URL')
    parser.add_argument('-o', '--output', default='data/rss_data.json', help='저장할 JSON 파일 경로 (기본값: data/rss_data.json)')
    
    args = parser.parse_args()
    
    try:
        output_path = process_rss_feed(args.url, args.output)
        print(f"RSS 데이터가 {output_path}에 저장되었습니다.")
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        print(f"오류 발생: {str(e)}") 