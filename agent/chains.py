import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

# .env 파일 로드
load_dotenv()

# Gemini LLM 초기화 (gemini-2.5-flash)
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0,
    max_retries=1,  # API 할당량 초과 시 무한 재시도로 인한 Nginx 504 대기 현상 방지
    timeout=15      # 15초 타임아웃 제한
)

# ----------------------------------------------------
# 1. 문서 평가기 (Document Grader)
# ----------------------------------------------------
class GradeDocuments(BaseModel):
    """검색된 문서의 관련성 점수 판정"""
    binary_score: str = Field(
        description="문서가 질문과 관련이 있으면 'yes', 관련이 없으면 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)

grader_system_prompt = (
    "당신은 검색된 문서와 사용자 질문 사이의 관련성을 평가하는 엄격한 평가자입니다.\n"
    "제공된 문서에 사용자 질문과 관련된 정보나 키워드가 포함되어 있다면 관련이 있는 것('yes')으로 판정하십시오.\n"
    "질문과 관련이 없는 문서라면 'no'로 판정하십시오."
)

grader_prompt = ChatPromptTemplate.from_messages([
    ("system", grader_system_prompt),
    ("human", "검색된 문서: \n\n {document} \n\n 사용자 질문: {question}")
])

doc_grader_chain = grader_prompt | structured_llm_grader


# ----------------------------------------------------
# 2. 답변 생성기 (Generator)
# ----------------------------------------------------
generator_system_prompt = (
    "당신은 고신뢰성 Q&A 비서입니다. 오직 제공된 문서(Context)에만 기반하여 질문에 답하십시오.\n"
    "답변을 모르는 경우나 제공된 문서에 관련 내용이 없다면 억지로 만들지 말고 솔직하게 모른다고 답하십시오.\n"
    "답변은 한국어로 친절하고 정중하게 작성해 주세요."
)

generator_prompt = ChatPromptTemplate.from_messages([
    ("system", generator_system_prompt),
    ("human", "제공된 문서:\n\n{context}\n\n사용자 질문: {question}")
])

generator_chain = generator_prompt | llm


# ----------------------------------------------------
# 3. 환각 검증기 (Hallucination Grader)
# ----------------------------------------------------
class GradeHallucination(BaseModel):
    """답변의 환각 여부 판정"""
    binary_score: str = Field(
        description="답변이 문서의 내용에 완벽하게 근거하면 'yes', 조금이라도 근거 없는 주장이 섞여 있다면 'no'"
    )

structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucination)

hallucination_system_prompt = (
    "당신은 생성된 답변이 제공된 문서(Context)에 완벽하게 기반하고 있는지(Grounding) 평가하는 검증자입니다.\n"
    "생성된 답변에 제공된 문서에서 직접 추론하거나 확인할 수 없는 새로운 사실이나 거짓 정보가 포함되어 있다면 'no'로 판정하십시오.\n"
    "답변의 모든 내용이 문서의 팩트와 일치한다면 'yes'로 판정하십시오."
)

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", hallucination_system_prompt),
    ("human", "제공된 문서:\n\n{documents}\n\n생성된 답변: {generation}")
])

hallucination_grader_chain = hallucination_prompt | structured_llm_hallucination_grader


# ----------------------------------------------------
# 4. 답변 유용성 검증기 (Answer Grader)
# ----------------------------------------------------
class GradeAnswer(BaseModel):
    """답변이 질문을 실제로 해결했는지 여부 판정"""
    binary_score: str = Field(
        description="답변이 사용자의 질문을 정확히 해결하면 'yes', 질문의 의도와 어긋나거나 핵심 정보가 빠졌으면 'no'"
    )

structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)

answer_grader_system_prompt = (
    "당신은 생성된 답변이 원래의 사용자 질문을 해결했는지(Address) 평가하는 검증자입니다.\n"
    "답변이 질문의 의도에 부합하고, 질문에 대한 실제 대답이나 유용한 정보를 담고 있다면 'yes'로 판정하십시오.\n"
    "질문의 핵심을 비껴갔거나, 질문에 대해 알 수 없다고만 답해 유용하지 않은 경우 'no'로 판정하십시오."
)

answer_grader_prompt = ChatPromptTemplate.from_messages([
    ("system", answer_grader_system_prompt),
    ("human", "사용자 질문: {question}\n\n생성된 답변: {generation}")
])

answer_grader_chain = answer_grader_prompt | structured_llm_answer_grader


# ----------------------------------------------------
# 5. 질문 재구성기 (Query Rewriter)
# ----------------------------------------------------
rewriter_system_prompt = (
    "당신은 웹 검색에 최적화된 쿼리로 질문을 개선하는 전문가입니다.\n"
    "사용자의 원래 질문을 받아 더 정확하고 풍부한 정보를 찾을 수 있는 간결한 웹 검색 쿼리로 재구성해 주세요.\n"
    "부가적인 설명 없이 오직 재구성된 검색 쿼리 텍스트만 반환하십시오."
)

rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", rewriter_system_prompt),
    ("human", "원래 질문: {question}")
])

query_rewriter_chain = rewriter_prompt | llm
