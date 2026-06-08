import os
import time
import requests
from dotenv import load_dotenv
from .state import AgentState
from .chains import doc_grader_chain, generator_chain, query_rewriter_chain
from .vector_db import vdb

load_dotenv()

def invoke_with_retry(chain, inputs, max_retries=3, delay=1.0):
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[LLM API Warning] Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            else:
                raise e

# 임시 로컬 문서 데이터베이스 (RAG retrieve용)
LOCAL_DOCUMENTS = [
    {
        "content": "LangGraph는 LangChain에서 만든 LLM 에이전트 및 멀티 에이전트 협업 시스템 구축을 위한 프레임워크입니다. 상태(State)를 보존하고 순환 그래프(Cyclic Graph) 구조를 지원하는 것이 핵심 특징입니다.",
        "source": "LangGraph 공식문서"
    },
    {
        "content": "자가 수정 RAG(Corrective RAG)는 검색된 문서의 적합성을 스스로 판별하여 무의미한 문서일 경우 웹 검색을 통해 보완하고, 생성된 답변의 환각을 재평가하는 RAG 아키텍처입니다.",
        "source": "RAG 논문 요약"
    },
    {
        "content": "한양대학교의 2026년 기말 프로젝트 제출 기한은 6월 중순이며, 보고서는 5페이지 이상으로 제출해야 합니다. LangChain/LangGraph 에이전트 소스코드도 함께 포함되어야 합니다.",
        "source": "과제 공지사항"
    }
]

# ChatGoogleGenerativeAI의 content가 list 형식으로 오는 오류 방지용 헬퍼 함수
def get_message_content(message) -> str:
    """
    AIMessage의 content가 string이 아닌 list 형태로 반환되는 경우를 안전하게 파싱합니다.
    """
    if hasattr(message, "content"):
        content = message.content
    else:
        content = message
        
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                text_parts.append(part.get("text", ""))
              
            else:
                text_parts.append(str(part))
        return "".join(text_parts)
    return str(content)

# ----------------------------------------------------
# 1. Retrieve Node (문서 검색)
# ----------------------------------------------------
def retrieve(state: AgentState) -> dict:
    print("\n--- [NODE] Retrieve Documents ---")
    question = state["question"]
    
    # 동적 벡터 DB에서 유사도 검색 수행 (Top-3)
    search_results, _ = vdb.similarity_search(question, k=3)
    retrieved_docs = [r["text"] for r in search_results] if search_results else []
    
    if not retrieved_docs:
        print("벡터 DB에서 매칭되는 문서를 찾지 못했습니다.")
        
    print(f"벡터 DB 검색 완료: {len(retrieved_docs)}개 문서 검색됨.")
    return {"documents": retrieved_docs, "question": question, "loop_count": state.get("loop_count", 0)}


# ----------------------------------------------------
# 2. Grade Documents Node (문서 적합성 평가)
# ----------------------------------------------------
def grade_documents(state: AgentState) -> dict:
    print("\n--- [NODE] Grade Documents ---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    search_needed = True
    
    if not documents:
        print("평가할 문서가 없습니다. 웹 검색이 필요합니다.")
        return {"documents": [], "search_needed": True}
        
    for doc in documents:
        response = invoke_with_retry(doc_grader_chain, {"question": question, "document": doc})
        score = response.binary_score.lower().strip()
        
        if score == "yes":
            print("  - [관련 있음] 문서 승인")
            filtered_docs.append(doc)
            search_needed = False
        else:
            print("  - [관련 없음] 문서 제외")
            
    if not filtered_docs:
        search_needed = True
        print("유효한 문서가 없습니다. 추가 웹 검색을 진행합니다.")
    else:
        print(f"평가 완료: {len(filtered_docs)}개 문서 유효 판정.")
        
    return {"documents": filtered_docs, "search_needed": search_needed}


# ----------------------------------------------------
# 3. Transform Query Node (질문 재구성)
# ----------------------------------------------------
def transform_query(state: AgentState) -> dict:
    print("\n--- [NODE] Transform Query (Query Rewrite) ---")
    question = state["question"]
    
    rewritten_query = invoke_with_retry(query_rewriter_chain, {"question": question})
    query_text = get_message_content(rewritten_query).strip()
    
    print(f"원본 질문: '{question}'")
    print(f"재구성된 웹 검색 쿼리: '{query_text}'")
    
    return {"web_search_query": query_text}


# ----------------------------------------------------
# 4. Web Search Node (웹 검색 수행)
# ----------------------------------------------------
def web_search(state: AgentState) -> dict:
    print("\n--- [NODE] Web Search ---")
    query = state["web_search_query"]
    documents = state["documents"]
    
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    web_results = []
    
    if not tavily_api_key:
        print("[WARNING] TAVILY_API_KEY가 설정되지 않아 웹 검색을 수행할 수 없습니다.")
        web_results = ["웹 검색 API 키가 누락되어 검색 결과를 가져오지 못했습니다."]
    else:
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": tavily_api_key,
                "query": query,
                "search_depth": "advanced",
                "max_results": 3
            }
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code == 200:
                results = response.json().get("results", [])
                for r in results:
                    web_results.append(f"출처: {r['url']}\n내용: {r['content']}")
                print(f"Tavily 웹 검색 성공: {len(web_results)}개 검색 결과 수집됨.")
            else:
                print(f"Tavily API 에러: 상태 코드 {response.status_code}")
                web_results = ["Tavily 웹 검색 API 호출에 실패했습니다."]
        except Exception as e:
            print(f"웹 검색 중 예외 발생: {str(e)}")
            web_results = [f"웹 검색 오류 발생: {str(e)}"]
            
    combined_docs = list(documents) + web_results
    return {"documents": combined_docs}


# ----------------------------------------------------
# 5. Generate Node (답변 생성)
# ----------------------------------------------------
def generate(state: AgentState) -> dict:
    print("\n--- [NODE] Generate Answer ---")
    question = state["question"]
    documents = state["documents"]
    loop_count = state.get("loop_count", 0)
    
    context = "\n\n".join(documents) if documents else "제공된 정보 없음."
    
    generation_response = invoke_with_retry(generator_chain, {"context": context, "question": question})
    generation = get_message_content(generation_response).strip()
    
    print("답변 생성 완료.")
    return {"generation": generation, "loop_count": loop_count + 1}
