# RSS 피드 처리 및 분석 프로젝트

이 프로젝트는 RSS 피드에서 데이터를 수집하고, 임베딩을 생성하여 지식 그래프로 분석하는 기능을 제공합니다. 또한 RAG(Retrieval-Augmented Generation) 기능을 통해 수집된 데이터를 기반으로 질문에 대한 답변을 생성합니다.

## 설치 방법

### 로컬 개발 환경 설정

```bash
# 가상 환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # macOS/Linux 또는 venv\Scripts\activate (Windows)

# 의존성 설치
pip install -r requirements.txt

# 언어 모델 다운로드
python -m spacy download en_core_web_sm

# 환경 변수 설정
cp .env.example .env
# .env 파일 편집하여 API 키 설정
```

### Docker를 사용한 설치

```bash
# 이미지 빌드 및 컨테이너 실행
docker-compose up -d
```

## 주요 기능

- RSS 피드에서 뉴스 기사 수집
- 텍스트 임베딩 생성 및 벡터 DB 저장
- 유사도 기반 문서 검색
- 개체명 인식(NER)을 통한 지식 그래프 구축
- Neo4j에 그래프 저장 및 분석
- OpenAI/HuggingFace 모델을 이용한 텍스트 생성
- RAG(Retrieval-Augmented Generation) 파이프라인
- 대화형 채팅 인터페이스

## 기본 사용법

### RSS 피드 수집 및 처리

```bash
# 기본 실행 - RSS 피드 수집 및 처리
python -m src.main https://news.google.com/rss

# 지식 그래프 생성 및 시각화 포함
python -m src.main https://news.google.com/rss -g -v
```

### 검색 및 RAG 사용

```bash
# 벡터 검색 실행
python -m src.embeddings.vector_db data/rss_data.json -q "인공지능 기술 동향" -n 5

# RAG 파이프라인 실행
python -m src.cli rag --query="우크라이나 전쟁 상황은 어떻게 되고 있나요?"
```

### 채팅 인터페이스 사용

```bash
# CLI 채팅 세션 시작
python -m src.cli chat --collection=news_articles

# 백엔드 API 서버 실행
python -m src.app
# 또는
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

### 웹 채팅 인터페이스 접속 방법

1. **백엔드 내장 채팅 인터페이스 (권장)**
   - 백엔드 서버를 실행한 후 브라우저에서 다음 주소로 접속:
   - http://localhost:8000/chat
   - 별도의 프론트엔드 설정 없이 바로 사용 가능

### 채팅 API 호출

```bash
# 채팅 설정: configs/chat_config.json 
# 채팅 API 사용
curl -X POST "http://localhost:8000/chat/message" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "user123", "message": "최근 뉴스 요약", "collection": "news_articles"}'
```

## API 서비스

```bash
# API 서버 실행
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
# Swagger 문서: http://localhost:8000/docs
```

### 주요 API 엔드포인트

- `/feed/process`: RSS 피드 처리
- `/search`: 벡터 검색
- `/generate-content`: 텍스트 생성
- `/rag/query`: RAG 쿼리 처리
- `/chat/message`: 채팅 메시지 전송
- `/graph/visualization`: 그래프 시각화

## LLM 통합

### 지원하는 LLM 모델

- **OpenAI**: GPT-3.5 Turbo, GPT-4, GPT-4 Turbo, GPT-4o
- **HuggingFace**: LLaMA 2, Falcon, Mistral, FLAN-T5
- **로컬 모델**: llama-cpp-python을 통한 로컬 모델 지원

### LLM 사용 예시

```bash
# 텍스트 생성
python -m src.cli generate --prompt="인공지능 기술의 최신 트렌드는?" --provider=openai --model=gpt-4

# 프롬프트 템플릿 사용
python -m src.cli generate-with-template --template="news_summary" --variables="topic=기후변화"
```

## RAG 워크플로우

RAG는 벡터 검색과 텍스트 생성을 결합하여 데이터 기반 응답을 생성하는 기술입니다:

1. 사용자 질의 → 임베딩 생성 → 벡터 DB 검색 → 관련 문서 검색
2. 프롬프트 구성 → LLM 응답 생성 → 사용자에게 응답 제공

```bash
# RAG 설정: configs/rag_config.json
# RAG 실행
python -m src.cli rag --query="최신 기술 트렌드는?" --collection=tech_news --num-results=5
```

## 도커 환경 접속 정보

- ChromaDB: http://localhost:8001
- Neo4j: http://localhost:7474 (사용자명: neo4j, 비밀번호: password)
- API 서버: http://localhost:8000