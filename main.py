import sys
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import retrieve, grade_documents, transform_query, web_search, generate
from agent.chains import hallucination_grader_chain, answer_grader_chain

load_dotenv()

# ----------------------------------------------------
# 조건부 라우팅 함수들 (Conditional Edges)
# ----------------------------------------------------

def decide_to_generate(state: AgentState) -> str:
    """
    문서 적합성 평가 결과를 보고, 생성을 시작할지 웹 검색을 추가할지 판단합니다.
    """
    print("\n--- [EDGE] Decide to Generate ---")
    if state["search_needed"]:
        print("결정: 적합한 문서 부족 -> 웹 검색으로 경로 분기")
        return "transform_query"
    else:
        print("결정: 적합한 문서 충분 -> 답변 생성으로 경로 분기")
        return "generate"


def grade_generation_v_documents_and_question(state: AgentState) -> str:
    """
    생성된 답변에 대한 환각 평가(Hallucination Check) 및 질문 답변 유용성(Answer Check)을 판정합니다.
    """
    print("\n--- [EDGE] Grade Generation vs Documents & Question ---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    loop_count = state.get("loop_count", 0)
    
    # 1. 환각 검증 (생성된 답변이 문서에 잘 근거하고 있는지)
    context = "\n\n".join(documents) if documents else ""
    hallucination_result = hallucination_grader_chain.invoke({
        "documents": context,
        "generation": generation
    })
    hallucination_score = hallucination_result.binary_score.lower().strip()
    
    if hallucination_score == "yes":
        print("검증 1: 답변이 문서에 근거함 (환각 없음) - 통과")
        
        # 2. 유용성 검증 (답변이 사용자 질문에 대한 유용한 해결책인지)
        answer_result = answer_grader_chain.invoke({
            "question": question,
            "generation": generation
        })
        answer_score = answer_result.binary_score.lower().strip()
        
        if answer_score == "yes":
            print("검증 2: 답변이 질문을 완전히 해결함 - 통과")
            return "useful"
        else:
            print("검증 2: 답변이 질문 해결에 부족함 - 재구성 및 재검색 시도")
            if loop_count >= 3:
                print("최대 루프 횟수(3회)에 도달하여 검증을 종료하고 답변을 반환합니다.")
                return "max_loops"
            return "not_useful"
    else:
        print("검증 1: 답변에서 환각(문서에 근거하지 않는 내용) 검출됨 - 답변 재생성 시도")
        if loop_count >= 3:
            print("최대 루프 횟수(3회)에 도달하여 검증을 종료하고 답변을 반환합니다.")
            return "max_loops"
        return "hallucination"


# ----------------------------------------------------
# LangGraph 컴파일 및 그래프 구축
# ----------------------------------------------------

workflow = StateGraph(AgentState)

# 1. 노드 추가
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

# 2. 시작 진입점 설정
workflow.set_entry_point("retrieve")

# 3. 엣지 연결
# Retrieve 이후 문서들을 평가합니다.
workflow.add_edge("retrieve", "grade_documents")

# Grade Documents의 결과에 따라 Generate 또는 Transform Query로 라우팅합니다.
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate"
    }
)

# 웹 검색을 위해 질문 재구성 후 웹 검색을 실행합니다.
workflow.add_edge("transform_query", "web_search")

# 웹 검색 후 다시 검색된 결과들을 평가하기 위해 grade_documents 노드로 되돌아갑니다.
workflow.add_edge("web_search", "grade_documents")

# Generate 이후 환각 여부와 질문 해결 여부를 검증합니다.
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "useful": END,
        "max_loops": END,
        "not_useful": "transform_query",
        "hallucination": "generate"
    }
)

# 그래프 컴파일
app = workflow.compile()


# ----------------------------------------------------
# 로컬 CLI 실행 루프 (인터렉티브 테스트 환경)
# ----------------------------------------------------
def run_cli():
    print("=========================================================")
    print("  자가 수정 및 적응형 RAG 에이전트 CLI 실행기")
    print("=========================================================")
    print("종료하려면 'exit' 또는 'quit'을 입력하세요.\n")
    
    while True:
        try:
            query = input("\n질문을 입력하세요: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("프로그램을 종료합니다.")
                break
                
            inputs = {
                "question": query,
                "loop_count": 0,
                "documents": []
            }
            
            # 그래프 실행 스트림 출력 및 로컬 상태 업데이트
            state = dict(inputs)
            for output in app.stream(inputs):
                for key, value in output.items():
                    print(f"\n=> [실행 단계 완료] Node: {key}")
                    state.update(value)
                    
            # 최종 결과 출력
            print("\n==================== [최종 답변] ====================")
            print(state.get("generation", "답변을 생성하지 못했습니다."))
            print("=====================================================")
        except KeyboardInterrupt:
            print("\n프로그램을 강제 종료합니다.")
            break
        except Exception as e:
            print(f"\n실행 중 에러가 발생했습니다: {str(e)}")

if __name__ == "__main__":
    run_cli()
