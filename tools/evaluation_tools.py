from lib.llm import LLM
from lib.messages import SystemMessage, UserMessage
from lib.parsers import JsonOutputParser, PydanticOutputParser
from lib.tooling import tool
from models.EvaluationReport import EvaluationReport

@tool
def evaluate_retrieval(question: str, retrieved_docs: list):
    """
    Based on the user's question and the retrieved documents from games vector db,
    evaluates whether the documents are useful to answer the question.

    Args:
        question: Original user question
        retrieved_docs: List of retrieved documents from vector DB

    Returns:
        EvaluationReport:
        - useful: Whether documents are sufficient to answer the question
        - description: Explanation of the evaluation
    """

    if not retrieved_docs:
        return EvaluationReport(
            useful=False,
            score=0.0,
            description="No retrieved documents; insufficient information to answer the question."
        )


    # LLM evaluation prompt
    system_message = """
    You are a strict evaluation agent.
    Only judge based on provided documents.
    If documents do not clearly answer the question, mark useful=false.
    """

    # Prepare readable docs summary
    docs_text = "\n".join(
        [f"{i+1}. {doc}" for i, doc in enumerate(retrieved_docs)]
    )

    prompt = f"""
    USER QUESTION:
    {question}
    
    RETRIEVED DOCUMENTS:
    {docs_text}
    
    TASK:
        Evaluate whether the retrieved documents are sufficient to answer the user's question.
    
    Instructions:
    - Determine if the documents are useful to answer the question
    - If useful, explain why
    - If not useful, explain what is missing
    - Be clear and actionable
    
    OUTPUT RULES:
    - Must return ONLY valid JSON
    - No markdown
    - No commentary outside JSON
    - Follow this exact schema
       
   CORING GUIDE:
    0.0 = unusable
    0.5 = partially helpful
    1.0 = fully sufficient
    
   Return a structured evaluation.
    """

    messages = [
        SystemMessage(content=system_message),
        UserMessage(content=prompt)
    ]

    service = LLM()
    # Call LLM
    ai_message = service.invoke(input = messages, response_format = EvaluationReport)

    parser = PydanticOutputParser(model_class=EvaluationReport)
    report: EvaluationReport = parser.parse(ai_message)

    return report
