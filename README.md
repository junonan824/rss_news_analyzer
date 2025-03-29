# RSS 피드 처리 및 분석 프로젝트

이 프로젝트는 RSS 피드에서 데이터를 수집하고, 임베딩을 생성하여 지식 그래프로 분석하는 기능을 제공합니다.

## 설치 방법

1. 의존성 설치:

```bash
pip install -r requirements.txt
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

### 통합 실행 (한 번에 모든 단계 실행)

RSS 피드에서 데이터 수집, 임베딩 생성, 벡터 DB 저장 및 검색까지 한 번에 실행:

```bash
python -m src.main https://news.google.com/rss -q "최신 기술 트렌드"
```

## 주요 기능

- 다양한 RSS 피드에서 기사 수집
- Sentence Transformers를 사용한 텍스트 임베딩
- ChromaDB를 활용한 효율적인 벡터 검색
- 메타데이터 필터링을 통한 고급 검색