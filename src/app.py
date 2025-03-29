"""
RSS 뉴스 분석기 API

이 모듈은 FastAPI를 사용하여 RSS 피드 처리, 벡터 검색, 지식 그래프 생성을 위한 
웹 API를 제공합니다.
"""

import os
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, HttpUrl

from src.rss_fetch.rss_fetch import process_rss_feed
from src.embeddings.embedding import process_rss_data
from src.embeddings.vector_db import load_rss_to_vectordb, VectorDB
from src.knowledge_graph.entity_extractor import EntityExtractor
from src.knowledge_graph.graph_builder import KnowledgeGraph
from src.knowledge_graph.neo4j_adapter import Neo4jAdapter
from src.generation.generation import generate_text, generate_text_with_retrieved_context

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="RSS 뉴스 분석기 API",
    description="RSS 피드를 처리하고, 벡터 검색 및 지식 그래프를 생성하는 API",
    version="1.0.0"
)

# 데이터 경로 설정
DATA_DIR = "data/api"
os.makedirs(DATA_DIR, exist_ok=True)

# 글로벌 변수 - 현재 로드된 데이터 상태 관리
current_feed = None
vector_db = None
knowledge_graph = None
neo4j_adapter = None

# 모델 정의
class FeedProcessRequest(BaseModel):
    url: HttpUrl
    collection_name: str = "rss_articles"
    ner_model: str = "en_core_web_sm"
    build_graph: bool = True
    visualize_graph: bool = True
    export_to_neo4j: bool = False

class ExportToNeo4jRequest(BaseModel):
    uri: str = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
    user: str = os.environ.get("NEO4J_USER", "neo4j")
    password: str = os.environ.get("NEO4J_PASSWORD", "password")
    clear_database: bool = True

class SearchQuery(BaseModel):
    query: str
    num_results: int = 5

class ProcessStatus(BaseModel):
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    results: List[Dict[str, Any]]
    count: int

class GenerateContentRequest(BaseModel):
    prompt: str
    provider: str = "openai"
    model: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.7

class GenerateSummaryRequest(BaseModel):
    query: str
    num_results: int = 3
    provider: str = "openai"
    model: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.7

# 백그라운드 작업
async def process_feed_background(url: str, collection_name: str, ner_model: str, build_graph: bool, 
                                  visualize_graph: bool, export_to_neo4j: bool):
    global current_feed, vector_db, knowledge_graph, neo4j_adapter
    
    try:
        # 파일 경로 설정
        rss_json_path = os.path.join(DATA_DIR, 'rss_data.json')
        embedding_json_path = os.path.join(DATA_DIR, 'embeddings.json')
        db_path = os.path.join(DATA_DIR, 'chroma_db')
        entities_json_path = os.path.join(DATA_DIR, 'entities.json')
        graph_json_path = os.path.join(DATA_DIR, 'knowledge_graph.json')
        graph_image_path = os.path.join(DATA_DIR, 'knowledge_graph.png')
        
        # 1. RSS 피드 처리
        process_rss_feed(url, rss_json_path)
        
        # 데이터 로드
        with open(rss_json_path, 'r', encoding='utf-8') as f:
            rss_data = json.load(f)
        
        # 2. 임베딩 생성 및 벡터 DB 저장
        process_rss_data(rss_json_path, output_path=embedding_json_path)
        vector_db = load_rss_to_vectordb(
            json_path=rss_json_path,
            collection_name=collection_name,
            persist_directory=db_path,
            embedding_path=embedding_json_path
        )
        
        # 3. 지식 그래프 생성 (선택적)
        if build_graph:
            # 개체 추출
            extractor = EntityExtractor(model_name=ner_model)
            articles_with_entities = extractor.extract_entities_batch(rss_data.get('articles', []))
            
            # 추출된 개체 저장
            with open(entities_json_path, 'w', encoding='utf-8') as f:
                json.dump({"articles": articles_with_entities}, f, ensure_ascii=False, indent=2)
            
            # 그래프 생성
            knowledge_graph = KnowledgeGraph()
            knowledge_graph.build_from_articles(articles_with_entities)
            
            # 그래프 저장
            knowledge_graph.save(graph_json_path)
            
            # 그래프 시각화 (선택적)
            if visualize_graph:
                knowledge_graph.visualize(graph_image_path)
            
            # Neo4j로 내보내기 (선택적)
            if export_to_neo4j:
                try:
                    # Neo4j 연결 설정은 기본값 또는 환경 변수에서 가져옴
                    neo4j_adapter = Neo4jAdapter()
                    if neo4j_adapter.connect():
                        neo4j_adapter.clear_database()
                        neo4j_adapter.import_graph(knowledge_graph.graph)
                except Exception as e:
                    print(f"Neo4j 내보내기 실패: {str(e)}")
        
        # 현재 피드 정보 업데이트
        current_feed = {
            "url": url,
            "collection_name": collection_name,
            "article_count": len(rss_data.get('articles', [])),
            "has_graph": build_graph,
            "has_visualization": build_graph and visualize_graph,
            "exported_to_neo4j": export_to_neo4j
        }
    
    except Exception as e:
        # 오류 발생 시 상태 초기화
        current_feed = {
            "url": url,
            "error": str(e),
            "status": "error"
        }
        raise e

# API 엔드포인트 정의
@app.get("/")
async def root():
    """API 상태 확인"""
    return {"status": "online", "message": "RSS 뉴스 분석기 API가 실행 중입니다."}

@app.post("/feed/process", response_model=ProcessStatus)
async def process_feed(request: FeedProcessRequest, background_tasks: BackgroundTasks):
    """
    RSS 피드를 처리하여 벡터 DB에 저장하고 지식 그래프 생성
    """
    background_tasks.add_task(
        process_feed_background,
        str(request.url),
        request.collection_name,
        request.ner_model,
        request.build_graph,
        request.visualize_graph,
        request.export_to_neo4j
    )
    
    return {
        "status": "processing",
        "message": f"RSS 피드 {request.url} 처리 중입니다. 상태를 확인하려면 /status를 호출하세요.",
        "details": {
            "url": request.url,
            "collection_name": request.collection_name,
            "build_graph": request.build_graph,
            "visualize_graph": request.visualize_graph,
            "export_to_neo4j": request.export_to_neo4j
        }
    }

@app.get("/status", response_model=ProcessStatus)
async def get_status():
    """
    현재 처리 상태 확인
    """
    if current_feed is None:
        return {
            "status": "idle",
            "message": "아직 처리된 피드가 없습니다.",
            "details": None
        }
    
    if "error" in current_feed:
        return {
            "status": "error",
            "message": f"마지막 처리 중 오류가 발생했습니다: {current_feed['error']}",
            "details": current_feed
        }
    
    return {
        "status": "ready",
        "message": "RSS 피드 처리가 완료되었습니다.",
        "details": current_feed
    }

@app.get("/search", response_model=SearchResult)
async def search(query: str = Query(..., description="검색 쿼리"), num_results: int = Query(5, ge=1, le=20)):
    """
    벡터 DB에서 의미적 검색 수행
    """
    if vector_db is None:
        raise HTTPException(status_code=400, detail="벡터 DB가 로드되지 않았습니다. 먼저 /feed/process를 호출하세요.")
    
    results = vector_db.search(query, num_results)
    
    documents = results.get('documents', [[]])[0]
    metadatas = results.get('metadatas', [[]])[0]
    distances = results.get('distances', [[]])[0]
    
    formatted_results = []
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        formatted_results.append({
            "rank": i + 1,
            "similarity": float(1 - dist),
            "title": meta.get('title', 'N/A'),
            "source": meta.get('field', 'N/A'),
            "link": meta.get('link', 'N/A'),
            "content": doc[:500] + "..." if len(doc) > 500 else doc,
            "published": meta.get('published', 'N/A')
        })
    
    return {
        "results": formatted_results,
        "count": len(formatted_results)
    }

@app.get("/graph/stats")
async def get_graph_stats():
    """
    지식 그래프 통계 정보 반환
    """
    if knowledge_graph is None:
        raise HTTPException(status_code=400, detail="지식 그래프가 생성되지 않았습니다. 먼저 /feed/process를 호출하세요.")
    
    node_types = {}
    for _, attr in knowledge_graph.graph.nodes(data=True):
        node_type = attr.get('type', 'UNKNOWN')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    edge_types = {}
    for _, _, attr in knowledge_graph.graph.edges(data=True):
        edge_type = attr.get('type', 'UNKNOWN')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    return {
        "graph_stats": {
            "node_count": knowledge_graph.graph.number_of_nodes(),
            "edge_count": knowledge_graph.graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types
        }
    }

@app.get("/graph/visualization")
async def get_graph_visualization():
    """
    지식 그래프 시각화 이미지 반환
    """
    graph_image_path = os.path.join(DATA_DIR, 'knowledge_graph.png')
    
    if not os.path.exists(graph_image_path):
        raise HTTPException(status_code=404, detail="그래프 시각화 이미지가 존재하지 않습니다.")
    
    return FileResponse(graph_image_path)

@app.post("/graph/export-to-neo4j")
async def export_graph_to_neo4j(request: ExportToNeo4jRequest):
    """
    지식 그래프를 Neo4j로 내보내기
    """
    if knowledge_graph is None:
        raise HTTPException(status_code=400, detail="지식 그래프가 생성되지 않았습니다. 먼저 /feed/process를 호출하세요.")
    
    result = knowledge_graph.export_to_neo4j(
        uri=request.uri,
        user=request.user,
        password=request.password
    )
    
    if result.get('success', False):
        return {
            "success": True,
            "message": "지식 그래프를 Neo4j로 성공적으로 내보냈습니다.",
            "details": result.get('stats', {})
        }
    else:
        raise HTTPException(
            status_code=500, 
            detail=f"Neo4j 내보내기 실패: {result.get('error', '알 수 없는 오류')}"
        )

@app.get("/neo4j/stats")
async def get_neo4j_stats():
    """
    Neo4j 데이터베이스 통계 반환
    """
    try:
        adapter = Neo4jAdapter()
        if adapter.connect():
            stats = adapter.get_stats()
            adapter.close()
            return {
                "success": True,
                "stats": stats
            }
        else:
            raise HTTPException(status_code=500, detail="Neo4j 연결 실패")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Neo4j 통계 조회 실패: {str(e)}")

@app.post("/generate-content")
async def generate_content(request: GenerateContentRequest):
    """
    제공된 프롬프트를 기반으로 텍스트 생성
    """
    try:
        result = generate_text(
            prompt=request.prompt,
            provider=request.provider,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return {
            "success": True,
            "generated_text": result,
            "prompt": request.prompt
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"텍스트 생성 중 오류 발생: {str(e)}")

@app.post("/generate-summary-of-search")
async def generate_summary_of_search(request: GenerateSummaryRequest):
    """
    검색 결과를 바탕으로 요약 생성 (RAG)
    """
    if vector_db is None:
        raise HTTPException(status_code=400, detail="벡터 DB가 로드되지 않았습니다. 먼저 /feed/process를 호출하세요.")
    
    # 검색 수행
    results = vector_db.search(request.query, request.num_results)
    
    if not results or not results.get('documents') or not results['documents'][0]:
        raise HTTPException(status_code=404, detail=f"'{request.query}'에 대한 검색 결과가 없습니다.")
    
    documents = results.get('documents', [[]])[0]
    metadatas = results.get('metadatas', [[]])[0]
    
    # 검색 결과 포맷팅
    context_docs = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        title = meta.get('title', 'N/A')
        context_docs.append(f"제목: {title}\n내용: {doc}")
    
    # RAG를 사용한 텍스트 생성
    try:
        summary = generate_text_with_retrieved_context(
            query=request.query,
            contexts=context_docs,
            provider=request.provider,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return {
            "success": True,
            "query": request.query,
            "summary": summary,
            "num_documents_used": len(context_docs),
            "documents": [
                {
                    "title": meta.get('title', 'N/A'),
                    "source": meta.get('field', 'N/A'),
                    "link": meta.get('link', 'N/A'),
                    "published": meta.get('published', 'N/A')
                }
                for meta in metadatas
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"요약 생성 중 오류 발생: {str(e)}")

# 서버 실행 코드
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 