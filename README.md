# 🧠 Adaptive & Self-Corrective RAG Agent

LangGraph와 FastAPI를 기반으로 구축된 **자가 수정 및 적응형 RAG(Adaptive & Self-Corrective RAG) 에이전트** 시스템입니다. 
로컬 지식베이스 검색 결과가 부족하거나 부적합할 때 스스로 질문을 재구성하여 웹 검색(Tavily API)을 수행하고, 생성된 답변의 환각(Hallucination) 및 유용성을 다단계로 검증하여 최적의 정확한 답변을 도출합니다.

---

## 💻 주요 기능 및 아키텍처

*   **상태 전이 기반 순환형 워크플로우(Stateful Cyclic Graph)**: LangGraph를 활용해 단순 체인형 RAG가 아닌, 평가 결과에 따라 경로를 수정하고 루프를 도는 동적 에이전트 구현
*   **자가 수정 RAG(Self-Corrective RAG)**: LLM 문서 평가기를 통해 검색된 정보의 관련성을 평가하고 무관한 정보는 필터링
*   **적응형 RAG(Adaptive RAG)**: 필요한 정보가 로컬 지식에 없을 경우, 최적화된 검색 쿼리로 질문을 변환하여 웹 검색 수행
*   **다단계 품질 보증 안전장치**: 생성된 답변의 환각 여부(Grounding Check)와 사용자의 원래 질문 해결 여부(Utility Check)를 순차적으로 검증
*   **FastAPI & Docker 가상화**: 서비스 상용화를 위해 FastAPI 서버 포장 및 Docker 컨테이너화 지원
*   **프리미엄 Web UI**: 사용자가 에이전트의 실시간 생각 흐름(Node 실행 로그)을 추적하고 간편하게 Q&A를 진행할 수 있는 웹 페이지 내장

---

## 📐 워크플로우 다이어그램

```mermaid
graph TD
    Start([시작: 사용자 질문 입력]) --> Retrieve[retrieve: 로컬 문서 검색]
    Retrieve --> GradeDocs[grade_documents: 문서 적합성 판정]
    
    GradeDocs --> DecideGenerate{Decide to Generate?<br>유효 문서 존재 여부}
    DecideGenerate -- Yes (문서 충분) --> Generate[generate: 답변 생성]
    DecideGenerate -- No (문서 부족) --> TransformQuery[transform_query: 검색어 재구성]
    
    TransformQuery --> WebSearch[web_search: Tavily 웹 검색 실행]
    WebSearch --> GradeDocs
    
    Generate --> GradeHallucination{Grade Generation?<br>환각 및 유용성 평가}
    
    GradeHallucination -- 1. 환각 없음 & 질문 해결 --> Useful([종료: 최종 답변 반환])
    GradeHallucination -- 2. 환각 검출 --> Hallucination[재생성 루프] --> Generate
    GradeHallucination -- 3. 질문 미해결 --> NotUseful[재검색 루프] --> TransformQuery
    GradeHallucination -- 4. 최대 시도 횟수 초과 --> MaxLoops([종료: 안전 답변 반환])
```

---

## 🛠️ 기술 스택 (Tech Stack)

*   **Framework**: LangGraph, LangChain Core
*   **LLM**: Google Gemini 2.5 Flash (`gemini-2.5-flash`)
*   **Search API**: Tavily Search API
*   **Backend**: FastAPI, Uvicorn
*   **Deployment**: Docker, Docker Compose, Nginx
*   **Frontend**: Vanilla HTML5, CSS3, Javascript

---

## 🚀 시작 가이드 (Quick Start)

### 1. 환경 변수 설정
프로젝트 루트 디렉토리에 `.env` 파일을 생성하고 필요한 API 키를 입력합니다.
```env
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 2. Docker Compose로 실행 (권장)
프로젝트는 Docker 컨테이너 환경에서 실행하는 것을 권장합니다.
```bash
# 컨테이너 빌드 및 백그라운드 구동
docker compose up -d --build
```
*   **실시간 서비스 웹 UI 접속**: [https://injun-cloud.duckdns.org/rag/](https://injun-cloud.duckdns.org/rag/)
*   **실시간 서비스 API 문서**: [https://injun-cloud.duckdns.org/rag/docs](https://injun-cloud.duckdns.org/rag/docs)
*   **로컬 웹 UI 접속**: `http://localhost:8000/`
*   **로컬 API 문서 (Swagger)**: `http://localhost:8000/docs`

### 3. 로컬 가상환경에서 실행
```bash
# 가상환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

# FastAPI 서버 실행
python app.py
```

---

## 📂 파일 구조 (Directory Structure)

*   `app.py`: FastAPI 웹 서버 엔트리포인트 및 static UI 렌더링 라우터
*   `main.py`: LangGraph 워크플로우 구성 및 로컬 CLI 테스터
*   `agent/`: 에이전트 핵심 로직 패키지
    *   `__init__.py`: 패키지 초기화 파일
    *   `state.py`: 그래프 노드 간 상태를 저장하고 넘겨주는 `AgentState` 정의
    *   `nodes.py`: 에이전트 그래프를 구성하는 각 노드 함수 구현체
    *   `chains.py`: 문서 평가, 질문 재구성, 답변 생성, 검증에 사용되는 LLM 체인 정의
*   `templates/index.html`: 에이전트 상호작용 및 생각 흐름 시각화를 위한 Web UI 템플릿
*   `Dockerfile` / `docker-compose.yml`: 컨테이너 빌드 및 서비스 오케스트레이션 구성 파일
*   `report.md`: 프로젝트 최종 보고서 (한양대학교)
