"""
Entity Extractor Module

이 모듈은 spaCy를 사용하여 텍스트에서 개체(Entity)를 추출하는 기능을 제공합니다.
"""

import logging
import spacy
from typing import List, Dict, Any, Tuple, Set, Optional
from collections import defaultdict

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EntityExtractor:
    """텍스트에서 개체(Entity)를 추출하는 클래스"""
    
    def __init__(self, model_name: str = "en_core_web_sm", load_korean: bool = False):
        """
        EntityExtractor 초기화
        
        Args:
            model_name (str): 사용할 spaCy 모델 이름 (기본값: "en_core_web_sm")
            load_korean (bool): 한국어 모델도 로드할지 여부 (기본값: False)
        """
        logger.info(f"spaCy 모델 '{model_name}' 로딩 중...")
        try:
            # 영어 모델 로드
            self.nlp = spacy.load(model_name)
            self.models = {"en": self.nlp}
            
            # 한국어 모델 로드 (선택사항)
            if load_korean:
                try:
                    self.korean_nlp = spacy.load("ko_core_news_sm")
                    self.models["ko"] = self.korean_nlp
                    logger.info("한국어 spaCy 모델 로드 완료")
                except Exception as e:
                    logger.warning(f"한국어 모델 로드 실패: {str(e)}")
                    logger.warning("한국어 모델을 사용하려면 다음 명령어를 실행하세요: python -m spacy download ko_core_news_sm")
            
            logger.info("spaCy 모델 로딩 완료")
        except Exception as e:
            logger.error(f"spaCy 모델 로딩 중 오류 발생: {str(e)}")
            logger.error("모델을 다운로드하려면 다음 명령어를 실행하세요: python -m spacy download en_core_web_sm")
            raise
    
    def extract_entities(self, text: str, lang: str = "en") -> List[Dict[str, Any]]:
        """
        텍스트에서 개체(인물, 조직, 장소 등) 추출
        
        Args:
            text (str): 분석할 텍스트
            lang (str): 텍스트의 언어 ("en" 또는 "ko")
            
        Returns:
            List[Dict[str, Any]]: 추출된 개체 목록
                [
                    {
                        "text": "개체 텍스트",
                        "label": "개체 유형(PERSON, ORG, GPE 등)",
                        "start_char": 시작 문자 인덱스,
                        "end_char": 종료 문자 인덱스
                    },
                    ...
                ]
        """
        if not text or not text.strip():
            logger.warning("빈 텍스트는 분석할 수 없습니다.")
            return []
        
        # 언어 모델 선택
        nlp = self.models.get(lang)
        if nlp is None:
            logger.warning(f"언어 '{lang}'에 대한 모델이 로드되지 않았습니다. 기본 모델을 사용합니다.")
            nlp = self.nlp
        
        # 텍스트 처리
        doc = nlp(text)
        
        # 개체 추출
        entities = []
        for ent in doc.ents:
            entity = {
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char
            }
            entities.append(entity)
        
        logger.info(f"{len(entities)}개의 개체를 추출했습니다.")
        return entities
    
    def extract_entities_from_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        기사 데이터에서 개체 추출
        
        Args:
            article (Dict[str, Any]): 기사 데이터
                {
                    "title": "기사 제목",
                    "content": "기사 본문",
                    ...
                }
            
        Returns:
            Dict[str, Any]: 개체가 추가된 기사 데이터
                {
                    "title": "기사 제목",
                    "content": "기사 본문",
                    "entities": [
                        {
                            "text": "개체 텍스트",
                            "label": "개체 유형",
                            "source": "title" 또는 "content",
                            ...
                        },
                        ...
                    ],
                    ...
                }
        """
        article_copy = article.copy()
        entities = []
        
        # 제목에서 개체 추출
        if "title" in article and article["title"]:
            title_entities = self.extract_entities(article["title"])
            for entity in title_entities:
                entity["source"] = "title"
                entities.append(entity)
        
        # 본문에서 개체 추출
        if "content" in article and article["content"]:
            content_entities = self.extract_entities(article["content"])
            for entity in content_entities:
                entity["source"] = "content"
                entities.append(entity)
        
        article_copy["entities"] = entities
        return article_copy
    
    def extract_entities_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        여러 기사에서 개체 추출
        
        Args:
            articles (List[Dict[str, Any]]): 기사 데이터 리스트
            
        Returns:
            List[Dict[str, Any]]: 개체가 추가된 기사 데이터 리스트
        """
        result = []
        for article in articles:
            article_with_entities = self.extract_entities_from_article(article)
            result.append(article_with_entities)
        
        return result

def extract_entities_from_texts(texts: List[str], model_name: str = "en_core_web_sm") -> List[List[Dict[str, Any]]]:
    """
    텍스트 리스트에서 개체를 추출하는 유틸리티 함수
    
    Args:
        texts (List[str]): 분석할 텍스트 리스트
        model_name (str): 사용할 spaCy 모델 이름
        
    Returns:
        List[List[Dict[str, Any]]]: 각 텍스트에서 추출된 개체 리스트
    """
    extractor = EntityExtractor(model_name)
    results = []
    
    for text in texts:
        entities = extractor.extract_entities(text)
        results.append(entities)
    
    return results

if __name__ == "__main__":
    import argparse
    import json
    import os
    
    parser = argparse.ArgumentParser(description='텍스트에서 개체를 추출합니다.')
    parser.add_argument('json_path', help='RSS 데이터가 저장된 JSON 파일 경로')
    parser.add_argument('-o', '--output', help='추출된 개체를 저장할 파일 경로')
    parser.add_argument('-m', '--model', default='en_core_web_sm', help='사용할 spaCy 모델 이름')
    parser.add_argument('-k', '--korean', action='store_true', help='한국어 모델도 사용')
    
    args = parser.parse_args()
    
    try:
        # JSON 파일 로드
        with open(args.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        articles = data.get('articles', [])
        logger.info(f"{len(articles)}개의 기사를 분석합니다.")
        
        # 개체 추출기 초기화
        extractor = EntityExtractor(args.model, args.korean)
        
        # 개체 추출
        articles_with_entities = extractor.extract_entities_batch(articles)
        
        # 결과 저장
        result_data = {
            "timestamp": data.get("timestamp", ""),
            "articles": articles_with_entities
        }
        
        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"개체 추출 결과가 {args.output}에 저장되었습니다.")
        else:
            # 개체 통계 출력
            entity_types = defaultdict(int)
            entity_counts = 0
            
            for article in articles_with_entities:
                for entity in article.get("entities", []):
                    entity_types[entity["label"]] += 1
                    entity_counts += 1
            
            print(f"\n추출된 개체 수: {entity_counts}")
            print("\n개체 유형별 통계:")
            for entity_type, count in sorted(entity_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {entity_type}: {count}")
    
    except Exception as e:
        logger.error(f"개체 추출 중 오류 발생: {str(e)}")
        print(f"오류 발생: {str(e)}") 