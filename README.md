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

### 통합 실행 (한 번에 모든 단계 실행)

RSS 피드에서 데이터 수집, 임베딩 생성, 벡터 DB 저장, 검색, 지식 그래프 생성까지 한 번에 실행:

```bash
python -m src.main https://news.google.com/rss -q "최신 기술 트렌드" -g -v
```

## 주요 기능

- 다양한 RSS 피드에서 기사 수집
- Sentence Transformers를 사용한 텍스트 임베딩
- ChromaDB를 활용한 효율적인 벡터 검색
- spaCy를 이용한 개체명 인식(NER)
- NetworkX 기반의 지식 그래프 구축
- 그래프 분석 및 시각화