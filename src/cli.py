"""
텍스트 생성 모듈 테스트를 위한 CLI 스크립트

이 스크립트는 RAG(Retrieval-Augmented Generation) 파이프라인의 텍스트 생성 기능을
커맨드 라인에서 테스트할 수 있도록 해줍니다.
"""

import os
import sys
import argparse
import logging
from typing import List, Optional

# 프록시 관련 환경 변수 제거 (OpenAI 클라이언트에 영향을 주지 않도록)
for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'REQUESTS_CA_BUNDLE', 'no_proxy', 'NO_PROXY']:
    if proxy_var in os.environ:
        del os.environ[proxy_var]

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 상대 경로 임포트를 위한 모듈 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.generation import generate_text, generate_text_with_retrieved_context
from src.embeddings.vector_db import VectorDB

def parse_args():
    parser = argparse.ArgumentParser(description='텍스트 생성 및 RAG 테스트 CLI')
    
    subparsers = parser.add_subparsers(dest='command', help='실행할 명령')
    
    # 텍스트 생성 명령
    generate_parser = subparsers.add_parser('generate', help='텍스트 생성')
    generate_parser.add_argument('--prompt', required=True, help='생성을 위한 프롬프트')
    generate_parser.add_argument('--provider', default='openai', choices=['openai', 'huggingface'], help='사용할 텍스트 생성 제공자')
    generate_parser.add_argument('--model', help='사용할 모델 이름')
    generate_parser.add_argument('--max-tokens', type=int, default=500, help='생성할 최대 토큰 수')
    generate_parser.add_argument('--temperature', type=float, default=0.7, help='생성 온도 (높을수록 더 창의적)')
    generate_parser.add_argument('--api-key', help='API 키 (환경 변수로 설정하는 것이 더 안전함)')
    
    # RAG 명령
    rag_parser = subparsers.add_parser('rag', help='검색 기반 텍스트 생성 (RAG)')
    rag_parser.add_argument('--query', required=True, help='질문')
    rag_parser.add_argument('--collection', default='rss_articles', help='검색할 벡터 DB 컬렉션')
    rag_parser.add_argument('--db-dir', default='data/chroma_db', help='ChromaDB 디렉토리 경로')
    rag_parser.add_argument('--num-results', type=int, default=3, help='검색할 문서 수')
    rag_parser.add_argument('--provider', default='openai', choices=['openai', 'huggingface'], help='사용할 텍스트 생성 제공자')
    rag_parser.add_argument('--model', help='사용할 모델 이름')
    rag_parser.add_argument('--max-tokens', type=int, default=500, help='생성할 최대 토큰 수')
    rag_parser.add_argument('--temperature', type=float, default=0.7, help='생성 온도 (높을수록 더 창의적)')
    rag_parser.add_argument('--api-key', help='API 키 (환경 변수로 설정하는 것이 더 안전함)')
    
    return parser.parse_args()

def generate_command(args):
    """단순 텍스트 생성 명령 처리"""
    try:
        # API 키 설정
        if args.api_key and args.provider == 'openai':
            os.environ['OPENAI_API_KEY'] = args.api_key
        
        # 텍스트 생성 - proxies 필터링
        logger.info(f"프롬프트로 텍스트 생성 중: '{args.prompt[:50]}...'")
        
        # 기본 인자 추출
        generation_kwargs = {
            'prompt': args.prompt,
            'provider': args.provider,
            'model': args.model,
            'max_tokens': args.max_tokens,
            'temperature': args.temperature,
        }
        
        # API 키 추가 (OpenAI인 경우만)
        if args.provider == 'openai' and args.api_key:
            generation_kwargs['api_key'] = args.api_key
            
        # proxies 제외하고 추가 인자 전달
        result = generate_text(**generation_kwargs)
        
        # 결과 출력
        print("\n결과:")
        print("-" * 80)
        print(result)
        print("-" * 80)
        
    except Exception as e:
        logger.error(f"텍스트 생성 중 오류 발생: {str(e)}")
        print(f"오류: {str(e)}")

def rag_command(args):
    """RAG 파이프라인 실행 명령 처리"""
    try:
        # API 키 설정
        if args.api_key and args.provider == 'openai':
            os.environ['OPENAI_API_KEY'] = args.api_key
        
        # 벡터 DB 연결 및 검색
        logger.info(f"검색어로 문서 검색 중: '{args.query}'")
        db = VectorDB(
            collection_name=args.collection,
            persist_directory=args.db_dir
        )
        collection = db.collection
        
        if collection is None:
            print(f"컬렉션 '{args.collection}'을 찾을 수 없습니다.")
            return
        
        # 검색 수행
        results = db.search(args.query, args.num_results)
        
        if not results or not results.get('documents') or not results['documents'][0]:
            print(f"'{args.query}'에 대한 검색 결과가 없습니다.")
            return
        
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        
        # 검색 결과 출력
        print(f"\n'{args.query}'에 대한 검색 결과 {len(documents)}개:")
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            print(f"\n{i+1}. 제목: {meta.get('title', 'N/A')}")
            print(f"   출처: {meta.get('field', 'N/A')}")
            print(f"   내용: {doc[:100]}..." if len(doc) > 100 else f"   내용: {doc}")
        
        # RAG 수행 - proxies 필터링
        logger.info("검색 결과를 바탕으로 텍스트 생성 중...")
        
        # 기본 인자 추출
        generation_kwargs = {
            'query': args.query,
            'contexts': documents,
            'provider': args.provider,
            'model': args.model,
            'max_tokens': args.max_tokens,
            'temperature': args.temperature,
        }
        
        # API 키 추가 (OpenAI인 경우만)
        if args.provider == 'openai' and args.api_key:
            generation_kwargs['api_key'] = args.api_key
            
        # proxies 제외하고 추가 인자 전달
        result = generate_text_with_retrieved_context(**generation_kwargs)
        
        # 결과 출력
        print("\n생성된 답변:")
        print("-" * 80)
        print(result)
        print("-" * 80)
        
    except Exception as e:
        logger.error(f"RAG 실행 중 오류 발생: {str(e)}")
        print(f"오류: {str(e)}")

def main():
    args = parse_args()
    
    if args.command == 'generate':
        generate_command(args)
    elif args.command == 'rag':
        rag_command(args)
    else:
        print("명령을 지정해야 합니다. '--help'를 사용하여 사용법을 확인하세요.")

if __name__ == "__main__":
    main() 