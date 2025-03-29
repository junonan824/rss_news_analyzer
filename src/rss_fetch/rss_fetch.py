"""
RSS Feed Fetcher Module

이 모듈은 RSS 피드 URL에서 기사 데이터를 가져와서 파싱하고,
JSON 파일 형태로 저장하는 기능을 제공합니다.
"""

import os
import json
import datetime
import feedparser
from bs4 import BeautifulSoup
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def normalize_date(date_str: str) -> str:
    """
    다양한 형식의 날짜 문자열을 표준 형식으로 변환합니다.
    
    Args:
        date_str (str): 날짜 문자열
        
    Returns:
        str: 표준화된 날짜 문자열 (ISO 형식)
    """
    if not date_str:
        return datetime.datetime.now().isoformat()
    
    try:
        # feedparser가 대부분의 날짜를 datetime 객체로 변환해줌
        date_obj = datetime.datetime(*date_str[:6])
        return date_obj.isoformat()
    except (TypeError, ValueError):
        # 변환 실패 시 현재 시간 반환
        logger.warning(f"날짜 변환 실패: {date_str}, 현재 시간으로 대체")
        return datetime.datetime.now().isoformat()

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
        articles (list): 기사 데이터 리스트
        output_file (str, optional): 저장할 파일 경로. 기본값은 'rss_data.json'
        
    Returns:
        str: 저장된 파일의 절대 경로
    """
    # 디렉토리가 없으면 생성
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 데이터에 타임스탬프 추가
    data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'articles': articles
    }
    
    # JSON 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"데이터가 성공적으로 저장됨: {os.path.abspath(output_file)}")
    return os.path.abspath(output_file)

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