# RSS News Analyzer

RSS 뉴스 피드를 수집하고, 텍스트 임베딩을 생성하며, 벡터 데이터베이스에 저장하고, Knowledge Graph를 구축하는 Python 프로젝트입니다.

## 주요 기능

- RSS 피드 수집 및 전처리
- 텍스트 임베딩 생성 (Sentence Transformers)
- 벡터 데이터베이스 저장 (ChromaDB)
- Knowledge Graph 구축 및 분석

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/rss_news_analyzer.git
cd rss_news_analyzer
```

2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

4. 환경 변수 설정
```bash
cp .env.example .env
# .env 파일을 수정하여 필요한 설정을 입력하세요
```

## 프로젝트 구조

```
rss_news_analyzer/
├── src/                    # 소스 코드
│   ├── config/            # 설정 파일
│   ├── data/              # 데이터 수집 및 전처리
│   ├── embedding/         # 임베딩 생성
│   ├── storage/           # 벡터 DB 저장
│   ├── knowledge_graph/   # Knowledge Graph 구축
│   └── utils/             # 유틸리티 함수
├── tests/                 # 테스트 코드
└── notebooks/            # 예제 노트북
```

## 사용 방법

1. RSS 피드 수집
```python
from src.data.collector import RSSCollector

collector = RSSCollector()
feeds = collector.collect_feeds()
```

2. 임베딩 생성
```python
from src.embedding.vectorizer import TextVectorizer

vectorizer = TextVectorizer()
embeddings = vectorizer.create_embeddings(feeds)
```

3. 벡터 DB 저장
```python
from src.storage.vector_store import VectorStore

store = VectorStore()
store.save_embeddings(embeddings)
```

4. Knowledge Graph 구축
```python
from src.knowledge_graph.builder import KnowledgeGraphBuilder

builder = KnowledgeGraphBuilder()
graph = builder.build_graph(feeds)
```

## 라이선스

MIT License

## 기여 방법

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 