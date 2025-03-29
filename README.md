# RSS 피드 처리 및 분석 프로젝트

이 프로젝트는 RSS 피드에서 데이터를 수집하고, 임베딩을 생성하여 지식 그래프로 분석하는 기능을 제공합니다.

## 설치 방법

### 로컬 개발 환경 설정

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

### Docker를 사용한 설치

전체 시스템을 Docker를 사용하여 설치하고 실행할 수 있습니다:

```bash
# 이미지 빌드 및 컨테이너 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

이 명령어는 다음 서비스를 실행합니다:
- RSS 뉴스 분석기 Python 애플리케이션 (FastAPI)
- ChromaDB 벡터 데이터베이스
- Neo4j 그래프 데이터베이스

## 프로젝트 구조

프로젝트는 다음과 같은 구조로 구성되어 있습니다:

```
rss_news_analyzer/
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   │
│   ├── rss_fetch/                    # RSS 피드 수집 모듈
│   │   ├── __init__.py
│   │   └── rss_fetch.py
│   │
│   ├── embeddings/                   # 벡터 임베딩 모듈
│   │   ├── __init__.py
│   │   ├── embedding.py
│   │   └── vector_db.py
│   │
│   ├── knowledge_graph/              # 지식 그래프 모듈
│   │   ├── __init__.py
│   │   ├── entity_extractor.py
│   │   ├── graph_builder.py
│   │   └── export_to_neo4j.py
│   │
│   ├── visualization/                # 시각화 모듈 (새로 추가)
│   │   ├── __init__.py
│   │   ├── embedding_visualizer.py   # (이전 visualize_embeddings.py)
│   │   └── search_visualizer.py      # (이전 vector_search_demo.py)
│   │
│   └── utils/                        # 유틸리티 함수 (새로 추가)
│       ├── __init__.py
│       ├── file_utils.py             # 파일 처리 유틸리티
│       └── logging_utils.py          # 로깅 유틸리티
│
├── scripts/                          # 실행 스크립트 (새로 추가)
│   ├── run_rss_fetch.py              # RSS 피드 수집 실행
│   ├── run_embedding.py              # 임베딩 생성 실행
│   ├── run_graph_builder.py          # 그래프 구축 실행
│   └── run_visualization.py          # 시각화 실행
│
├── data/                             # 데이터 저장 (기존 그대로)
│
├── tests/                            # 테스트 (기존 그대로)
│
└── configs/                          # 설정 파일 (새로 추가)
    ├── app_config.json               # 애플리케이션 설정
    ├── embedding_config.json         # 임베딩 모델 설정
    └── neo4j_config.json             # Neo4j 연결 설정
```

### 전체 시스템 아키텍처 다이어그램

```
+---------------------------+       +---------------------+
|                           |       |                     |
|   RSS Feed Sources        +------>+   RSS Fetch Module  |
|   (BBC, CNN, etc.)        |       |   (feedparser)      |
|                           |       |                     |
+---------------------------+       +----------+----------+
                                              |
                                              | JSON Data
                                              v
+---------------------------+       +---------------------+
|                           |       |                     |
|   ChromaDB                |<------+   Embedding Module  |
|   (Vector Database)       |       |   (Sentence-       |
|                           |       |    Transformers)    |
+-------------+-------------+       |                     |
              ^                     +---------------------+
              |                               |
              | Query/Results                 | Entities
              |                               v
+-------------+-------------+       +---------------------+
|                           |       |                     |
|   FastAPI                 |<----->+   Knowledge Graph   |
|   (Web API)               |       |   Module (spaCy,    |
|                           |       |   NetworkX)         |
+-------------+-------------+       |                     |
              ^                     +----------+----------+
              |                               |
              | HTTP Requests                 | Graph Data
              |                               v
+-------------+-------------+       +---------------------+
|                           |       |                     |
|   Client                  |       |   Neo4j             |
|   (Web Browser,           |       |   (Graph Database)  |
|   API Client)             |       |                     |
|                           |       |                     |
+---------------------------+       +---------------------+
```

### 데이터 흐름 다이어그램

```
[RSS 피드] --> [RSS 피드 파싱] --> [JSON 데이터] --> [텍스트 임베딩] --> [벡터 DB 저장]
                                           |
                                           v
                                   [개체명 인식(NER)]
                                           |
                                           v
                                   [지식 그래프 구축] --> [NetworkX 시각화]
                                           |
                                           v
                                   [Neo4j 내보내기] --> [Cypher 쿼리/분석]
```

### 주요 모듈 설명

- **rss_fetch**: RSS 피드에서 데이터를 수집하고 JSON 형식으로 저장
- **embeddings**: 텍스트 임베딩을 생성하고 ChromaDB에 저장/검색
- **knowledge_graph**: 개체 추출, 그래프 구축, Neo4j 연동
- **visualization**: 임베딩 시각화 및 벡터 검색 결과 시각화
- **utils**: 공통 유틸리티 함수

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

### Neo4j에 그래프 내보내기

만들어진 그래프를 Neo4j로 내보내려면:

```bash
python -m src.knowledge_graph.export_to_neo4j data/knowledge_graph.json
```

Neo4j 브라우저를 사용하여 그래프를 탐색할 수 있습니다: http://localhost:7474

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
# 로컬에서 API 서버 시작
python -m src.app

# 또는 uvicorn 직접 사용
uvicorn src.app:app --reload --host 0.0.0.0 --port 8000

# Docker로 실행 시
docker-compose up -d
```

API 서버는 기본적으로 http://localhost:8000 에서 실행됩니다.
Swagger 문서는 http://localhost:8000/docs 에서 확인할 수 있습니다.

### 주요 API 엔드포인트

#### RSS 피드 처리

```bash
curl -X POST "http://localhost:8000/feed/process" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://feeds.bbci.co.uk/news/world/rss.xml", "build_graph": true, "visualize_graph": true, "export_to_neo4j": true}'
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

#### Neo4j 통계 확인

```bash
curl "http://localhost:8000/neo4j/stats"
```

#### 그래프 시각화 이미지 다운로드

웹 브라우저에서 http://localhost:8000/graph/visualization 접속

## Docker 환경에서의 사용법

### 기본 사용법

```bash
# 전체 스택 시작
docker-compose up -d

# 특정 서비스만 재시작
docker-compose restart app

# 로그 확인
docker-compose logs -f app

# 전체 중지
docker-compose down
```

### Neo4j 브라우저 접속

1. 웹 브라우저에서 http://localhost:7474 접속
2. 로그인 정보:
   - 사용자명: neo4j
   - 비밀번호: password
3. 연결 설정:
   - 드롭다운에서 `bolt://` 선택
   - 주소란에 `localhost:7687` 입력

### Neo4j 유용한 쿼리

```cypher
// 모든 노드 보기
MATCH (n) RETURN n LIMIT 25

// 기사와 그 기사에서 언급된 엔티티 조회
MATCH (a:ARTICLE)-[r:MENTIONS]->(e) RETURN a, r, e LIMIT 10

// 특정 유형의 엔티티만 조회 (예: PERSON)
MATCH (p:PERSON) RETURN p

// 함께 등장한 엔티티 관계 조회
MATCH (e1)-[r:CO_OCCURS_WITH]->(e2) RETURN e1, r, e2 LIMIT 20

// 가장 많이 언급된 엔티티 찾기
MATCH (e)<-[r:MENTIONS]-(a) 
RETURN e.text AS entity, e.type AS type, COUNT(r) AS mentions 
ORDER BY mentions DESC LIMIT 10
```

### ChromaDB 접속

ChromaDB는 REST API로 접근 가능합니다: http://localhost:8001

예제 API 호출:
```bash
# 컬렉션 목록 조회
curl http://localhost:8001/api/v1/collections
```

### 트러블슈팅

#### Neo4j 연결 문제
- Neo4j 브라우저에서 연결할 때 URL을 `bolt://localhost:7687`로 지정
- 사용자명과 비밀번호가 docker-compose.yml 파일의 설정과 일치하는지 확인

#### huggingface_hub 호환성 문제
version 충돌이 발생하면 다음 명령으로 해결:
```bash
pip install huggingface_hub==0.17.3
```

#### Docker 네트워크 문제
```bash
# Docker 네트워크 상태 확인
docker network ls
docker network inspect rss_news_analyzer_rss_network
```

### 데이터 볼륨

- ChromaDB 데이터: `chroma_data` 볼륨
- Neo4j 데이터: `neo4j_data` 볼륨

Docker에서는 데이터가 볼륨에 저장되어 컨테이너가 재시작되어도 유지됩니다.

## 주요 기능

- 다양한 RSS 피드에서 기사 수집
- Sentence Transformers를 사용한 텍스트 임베딩
- ChromaDB를 활용한 효율적인 벡터 검색
- spaCy를 이용한 개체명 인식(NER)
- NetworkX 기반의 지식 그래프 구축
- Neo4j를 이용한 그래프 데이터 탐색
- 그래프 분석 및 시각화
- FastAPI 기반 웹 API 제공
- Docker 컨테이너화 지원