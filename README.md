# RSS 피드 처리 및 분석 프로젝트

이 프로젝트는 RSS 피드에서 데이터를 수집하고, 임베딩을 생성하여 지식 그래프로 분석하는 기능을 제공합니다.

## 설치 방법

1. 가상 환경 생성 및 활성화:

```bash
# 가상 환경 생성
python -m venv venv

# 가상 환경 활성화(Windows)
venv\Scripts\activate

# 가상 환경 활성화(macOS/Linux)
source venv/bin/activate
```

2. 의존성 설치:

```bash
pip install -r requirements.txt
```

3. spaCy 언어 모델 다운로드:

```bash
# 영어 모델 다운로드
python -m spacy download en_core_web_sm

# 한국어 모델 다운로드 (선택사항)
python -m spacy download ko_core_news_sm
```

README는 프로젝트의 문서이자 가이드라인으로, 프로젝트의 사용 맥락에 맞게 수정하시면 됩니다.

## 임베딩 & 검색 사용 방법

### 텍스트 임베딩 생성

RSS 데이터에서 텍스트 임베딩을 생성하려면:

```bash
python -m src.embeddings.embedding data/rss_data.json -o data/embeddings.json
```

### 벡터 DB에 데이터 저장

임베딩 데이터를 벡터 DB에 저장하려면:

```bash
python -m src.embeddings.vector_db data/rss_data.json -e data/embeddings.json
```

### 유사도 검색

특정 쿼리에 대한 유사한 문서를 검색하려면:

```bash
python -m src.embeddings.vector_db data/rss_data.json -q "인공지능 기술 동향" -n 5
```

## 지식 그래프 사용 방법

### 개체(Entity) 추출

RSS 데이터에서 개체(인물, 조직, 장소 등)를 추출하려면:

```bash
python -m src.knowledge_graph.entity_extractor data/rss_data.json -o data/entities.json
```

### 지식 그래프 생성

추출된 개체로 지식 그래프를 구축하려면:

```bash
python -m src.knowledge_graph.graph_builder data/entities.json -o data/knowledge_graph.json
```

### 그래프 시각화

지식 그래프를 이미지로 시각화하려면:

```bash
python -m src.knowledge_graph.graph_builder data/entities.json -v data/knowledge_graph.png
```

### 특정 개체 유형만 시각화

특정 유형의 개체(예: 인물, 조직)만 시각화하려면:

```bash
python -m src.knowledge_graph.graph_builder data/entities.json -v data/knowledge_graph.png -t PERSON ORG
```

## 통합 실행 (모든 기능을 한 번에)

### 명령행 인터페이스 (CLI)

RSS 피드에서 데이터 수집, 임베딩 생성, 벡터 DB 저장, 검색, 지식 그래프 생성까지 한 번에 실행:

```bash
# 기본 실행 - RSS 피드 수집 및 처리
python -m src.main https://news.google.com/rss

# 지식 그래프 생성 포함
python -m src.main https://news.google.com/rss -g

# 지식 그래프 시각화 포함
python -m src.main https://news.google.com/rss -g -v

# 검색 쿼리 포함
python -m src.main https://news.google.com/rss -q "최신 기술 트렌드" -g -v

# 전체 옵션 사용 예시
python -m src.main https://feeds.bbci.co.uk/news/world/rss.xml -o data/bbc -c bbc_news -q "climate change" -n 10 -g -v --ner_model en_core_web_sm
```

### 전체 옵션 설명

- `url`: RSS 피드 URL (필수)
- `-o, --output_dir`: 출력 디렉토리 (기본값: data)
- `-c, --collection`: 벡터 DB 컬렉션 이름 (기본값: rss_articles)
- `-q, --query`: 검색 쿼리 (선택사항)
- `-n, --num_results`: 검색 결과 수 (기본값: 5)
- `-g, --graph`: 지식 그래프 생성 (플래그)
- `-v, --visualize`: 그래프 시각화 (플래그)
- `--ner_model`: NER에 사용할 spaCy 모델 (기본값: en_core_web_sm)

## Web API (FastAPI 기반)

프로젝트는 웹 API를 통해 모든 기능을 사용할 수 있도록 FastAPI 기반 인터페이스를 제공합니다.

### API 서버 실행

```bash
# API 서버 시작
python -m src.app

# 또는 uvicorn 직접 사용
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000
```

API 서버는 기본적으로 http://localhost:8000 에서 실행됩니다.
Swagger 문서는 http://localhost:8000/docs 에서 확인할 수 있습니다.

### 주요 API 엔드포인트

#### RSS 피드 처리

```bash
curl -X POST "http://localhost:8000/feed/process" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://feeds.bbci.co.uk/news/world/rss.xml", "build_graph": true, "visualize_graph": true}'
```

#### 상태 확인

```bash
curl "http://localhost:8000/status"
```

#### 검색

```bash
curl "http://localhost:8000/search?query=climate%20change&num_results=5"
```

#### 그래프 통계

```bash
curl "http://localhost:8000/graph/stats"
```

#### 그래프 시각화 이미지 다운로드

웹 브라우저에서 http://localhost:8000/graph/visualization 접속

## 주요 기능

- 다양한 RSS 피드에서 기사 수집
- Sentence Transformers를 사용한 텍스트 임베딩
- ChromaDB를 활용한 효율적인 벡터 검색
- spaCy를 이용한 개체명 인식(NER)
- NetworkX 기반의 지식 그래프 구축
- 그래프 분석 및 시각화
- FastAPI 기반 웹 API 제공