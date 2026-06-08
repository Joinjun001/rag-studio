from typing import List, TypedDict

class AgentState(TypedDict):
    """
    LangGraph 에이전트의 상태를 정의합니다.
    각 노드는 이 상태를 입력받아 처리한 후, 수정된 상태를 반환합니다.
    """
    question: str          # 사용자의 원본 질문
    generation: str        # 생성된 답변
    documents: List[str]   # 검색(Retrieve) 또는 웹 검색을 통해 수집된 문서 리스트
    search_needed: bool    # 추가적인 웹 검색이 필요한지 여부
    web_search_query: str  # 웹 검색에 사용할 재구성된 검색 쿼리
    loop_count: int        # 무한 루프 방지를 위한 노드 재시도 횟수 카운터
