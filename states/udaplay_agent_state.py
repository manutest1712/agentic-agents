from typing import Optional

from lib.agents import AgentState
from models.EvaluationReport import EvaluationReport


from typing import Optional, List

class UdaPlayAgentState(AgentState):
    """
    Extended agent state schema for the UdaPlay Agent.

    This state builds on the base `AgentState` and adds fields required
    for retrieval-augmented reasoning, evaluation, and final answer generation.

    It supports a multi-step agent flow where the agent:
    1. Attempts to answer using internal knowledge
    2. Retrieves external documents if needed
    3. Evaluates retrieval usefulness
    4. Falls back to web search when required
    5. Produces a final answer

    Inherited fields from AgentState:
        - user_query: The original user question
        - instructions: System-level agent instructions
        - messages: Conversation history passed to the LLM
        - current_tool_calls: Any pending tool function calls
        - total_tokens: Running total of token usage
    """

    retrieved_docs: Optional[list]
    """Documents retrieved from the vector database or internal knowledge store."""

    evaluation_report: Optional[EvaluationReport]
    """
    Result of evaluating retrieved documents.
    Indicates whether the retrieved content is sufficient and relevant
    to answer the user query.
    """

    final_answer: Optional[str]
    """The final resolved answer produced by the agent after reasoning and tool usage."""

    reasoning_log: Optional[list]
    """Holds reasoning logs ."""

    tool_trace: Optional[list]
    """Holds tool trace logs ."""

    messages: Optional[List[dict]]  # NEW
    session_id: str  # NEW