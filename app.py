from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# main.py에서 컴파일된 LangGraph app 임포트
from main import app as agent_app
# 싱글톤 벡터 DB 임포트
from agent.vector_db import vdb

load_dotenv()

# FastAPI 인스턴스 생성
app = FastAPI(
    title="RAG & Vector DB Visualization Lab",
    description="한양대학교 기말 프로젝트 - RAG 및 벡터 DB 시각화 연구실",
    version="1.0.0",
    root_path="/rag"
)

# 서버 시작 시 샘플 데이터 적재
@app.on_event("startup")
def startup_event():
    try:
        if not vdb.documents:
            vdb.add_document(
                "LangGraph는 LangChain에서 만든 LLM 에이전트 및 멀티 에이전트 협업 시스템 구축을 위한 프레임워크입니다. 상태(State)를 보존하고 순환 그래프(Cyclic Graph) 구조를 지원하는 것이 핵심 특징입니다.",
                "LangGraph 공식문서"
            )
            vdb.add_document(
                "자가 수정 RAG(Corrective RAG)는 검색된 문서의 적합성을 스스로 판별하여 무의미한 문서일 경우 웹 검색을 통해 보완하고, 생성된 답변의 환각을 재평가하는 RAG 아키텍처입니다.",
                "RAG 논문 요약"
            )
            vdb.add_document(
                "한양대학교의 2026년 기말 프로젝트 제출 기한은 6월 중순이며, 보고서는 5페이지 이상으로 제출해야 합니다. LangChain/LangGraph 에이전트 소스코드도 함께 포함되어야 합니다.",
                "과제 공지사항"
            )
            print("[Startup] RAG 시각화용 샘플 문서 3건 임베딩 완료.")
    except Exception as e:
        print(f"[Startup Error] 샘플 문서 적재 중 에러 발생: {str(e)}")

# CORS 설정 (외부 자바스크립트 클라이언트 및 웹 UI 연동 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 요청 스키마 정의
class QueryRequest(BaseModel):
    question: str

class DocumentAddRequest(BaseModel):
    text: str
    source: str = "User Upload"

# 응답 스키마 정의
class QueryResponse(BaseModel):
    question: str
    generation: str
    documents_count: int
    loop_count: int

@app.get("/", response_class=HTMLResponse, summary="에이전트 웹 인터페이스")
def read_root():
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>RAG Visualization Lab API</h1><p>Template not found.</p>"

@app.get("/health", summary="서버 헬스 체크")
def health_check():
    return {"status": "healthy", "service": "Adaptive RAG Agent API"}

@app.post("/add-document", summary="지식 문서 추가 및 실시간 임베딩 생성")
def add_document(request: DocumentAddRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="문서 내용(text)은 비어있을 수 없습니다.")
    try:
        doc = vdb.add_document(request.text, request.source)
        return {"status": "success", "message": "문서가 성공적으로 임베딩되어 벡터 DB에 추가되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문서 추가 중 오류 발생: {str(e)}")

@app.post("/clear-documents", summary="벡터 DB의 모든 문서 삭제")
def clear_documents():
    vdb.clear()
    return {"status": "success", "message": "벡터 DB가 완전히 초기화되었습니다."}

@app.get("/documents", summary="현재 벡터 DB의 문서들 및 2D 투영 좌표 조회")
def get_documents():
    try:
        doc_coords, _ = vdb.get_visualization_coordinates()
        return {"documents": doc_coords}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"문서 조회 중 오류 발생: {str(e)}")

@app.post("/query-visualize", summary="질문 임베딩 생성, 유사도 검색, 2D 공간 투영 및 답변 도출")
def query_visualize(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="질문(question)은 비어있을 수 없습니다.")
        
    try:
        # 1. 벡터 DB에서 유사도 검색 및 질문 벡터 획득
        search_results, query_vector = vdb.similarity_search(request.question, k=3)
        
        # 2. 2D PCA 투영 좌표 계산
        doc_coords, query_coord = vdb.get_visualization_coordinates(query_vector)
        
        # 3. 유사도 순위 정보 조립
        rankings = []
        for r in search_results:
            rankings.append({
                "text": r["text"],
                "source": r["source"],
                "similarity": r["similarity"]
            })
            
        # 4. LangGraph 에이전트 실행 (이때 에이전트의 retrieve 노드도 싱글톤 vdb를 탐색함)
        inputs = {
            "question": request.question,
            "loop_count": 0,
            "documents": []
        }
        agent_result = agent_app.invoke(inputs)
        
        # 5. 프롬프트 조립본 시뮬레이션
        context = "\n\n".join([r["text"] for r in search_results]) if search_results else "제공된 정보 없음."
        constructed_prompt = (
            f"시스템: 당신은 고신뢰성 Q&A 비서입니다. 오직 제공된 문서(Context)에만 기반하여 질문에 답하십시오...\n\n"
            f"제공된 문서:\n{context}\n\n사용자 질문: {request.question}"
        )
        
        return {
            "generation": agent_result.get("generation", "답변을 생성하지 못했습니다."),
            "loop_count": agent_result.get("loop_count", 0),
            "documents_count": len(agent_result.get("documents", [])),
            "query_coord": query_coord,
            "doc_coords": doc_coords,
            "rankings": rankings,
            "constructed_prompt": constructed_prompt
        }
    except Exception as e:
        print(f"[API Error] Exception during visualization query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"에이전트 처리 및 시각화 연산 중 서버 오류 발생: {str(e)}")

# CLI 구동용과 FastAPI 기본 query도 호환되도록 유지
@app.post("/query", response_model=QueryResponse, summary="질문 처리 및 에이전트 실행 (기본 호환)")
async def run_query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="질문(question)은 비어있을 수 없습니다.")
    try:
        inputs = {"question": request.question, "loop_count": 0, "documents": []}
        result = agent_app.invoke(inputs)
        return QueryResponse(
            question=result.get("question", request.question),
            generation=result.get("generation", "답변을 생성하지 못했습니다."),
            documents_count=len(result.get("documents", [])),
            loop_count=result.get("loop_count", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
