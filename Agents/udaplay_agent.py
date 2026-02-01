import json

from config.settings import DEFAULT_MODEL
from lib.agents import AgentState
from lib.llm import LLM
from lib.memory import ShortTermMemory
from lib.messages import AIMessage, SystemMessage, UserMessage, ToolMessage
from lib.state_machine import StateMachine, EntryPoint, Step, Termination, Run
from states.udaplay_agent_state import UdaPlayAgentState

AGENT_INSTRUCTIONS = """
You are UdaPlay — a gaming research assistant.
Use internal knowledge first.
Evaluate answer quality.
If insufficient, search the web with the tools provided.
Never hallucinate.
Respond clearly, with reasoning and citations.
"""

class UdaPlayAgent:
    def __init__(self, tools: list):

        # Store tools in a dict for quick lookup
        self.tools = {tool.name: tool for tool in tools}
        print(self.tools)


        self.workflow = self.create_state_machine()
        self.memory = ShortTermMemory()

    def create_state_machine(self) -> StateMachine[UdaPlayAgentState]:
        workflow = StateMachine[UdaPlayAgentState](UdaPlayAgentState)

        entry = EntryPoint[UdaPlayAgentState]()
        retrieve = Step("retrieve_internal", self.retrieve_step)
        evaluate = Step("evaluate", self.evaluate_step)
        web_search = Step("web_search", self.web_search_step)
        answer = Step("answer", self.answer_step)
        termination = Termination[UdaPlayAgentState]()

        workflow.add_steps([
            entry, retrieve, evaluate,
            web_search, answer, termination
        ])

        workflow.connect(entry, retrieve)
        workflow.connect(retrieve, evaluate)

        workflow.connect(evaluate, [answer, web_search], self.route_decision)
        workflow.connect(web_search, answer)
        workflow.connect(answer, termination)
        return workflow

    def run(self, question: str, session_id: str = "default"):

        # Create session if missing
        self.memory.create_session(session_id)

        # Load previous messages
        previous_messages = []
        last_run: Run = self.memory.get_last_object(session_id)

        if last_run:
            last_state = last_run.get_final_state()
            if last_state:
                previous_messages = last_state.get("messages", [])


        initial_state: UdaPlayAgentState = {
            "user_query": question,
            "instructions": AGENT_INSTRUCTIONS,
            "retrieved_docs": None,
            "evaluation_report": None,
            "reasoning_log": [],
            "tool_trace": [],
            "final_answer": None,
            "messages": previous_messages,
            "session_id": session_id
        }



        result = self.workflow.run(initial_state)
        self.memory.add(result, session_id)
        # Safe final answer return
        final_state = result.get_final_state() or {}
        return final_state.get("final_answer", "No answer generated.")

    def retrieve_step(self, state: UdaPlayAgentState):
        tool = self.tools["retrieve_game"]
        docs = tool(state["user_query"])
        print(docs)
        state["reasoning_log"].append(
            "Retrieved internal knowledge using retrieve_game tool."
        )
        state["tool_trace"].append({
            "tool": "retrieve_game",
            "input": state["user_query"],
            "output_preview": str(docs)[:300]
        })
        return {"retrieved_docs": docs}

    def evaluate_step(self, state: UdaPlayAgentState):
        evaluation = self.tools["evaluate_retrieval"](
            question=state["user_query"],
            retrieved_docs=state["retrieved_docs"]
        )
        state["reasoning_log"].append(
            f"""Evaluation result: useful={evaluation.useful}, 
            Score: {evaluation.score},
            Message: {evaluation.description}"""
        )

        state["tool_trace"].append({
            "tool": "evaluate_retrieval",
            "output": str(evaluation)
        })
        return {"evaluation_report": evaluation}

    def route_decision(self, state: UdaPlayAgentState):
        if state["evaluation_report"].useful:
            state["reasoning_log"].append(
                "Internal knowledge sufficient. Proceeding to answer."
            )
            return self.workflow.steps["answer"]

        state["reasoning_log"].append(
            "Internal knowledge insufficient. Switching to web search."
        )
        return self.workflow.steps["web_search"]

    def web_search_step(self, state: UdaPlayAgentState):
        results = self.tools["game_web_search"](state["user_query"])

        state["tool_trace"].append({
            "tool": "game_web_search",
            "input": state["user_query"],
            "output_preview": str(results)[:300]
        })

        state["reasoning_log"].append(
            "Retrieved external web sources."
        )

        return {"retrieved_docs": results}

    def answer_step(self, state: UdaPlayAgentState):

        docs = state["retrieved_docs"]
        reasoning_log = "\n".join(state["reasoning_log"])
        tool_trace = json.dumps(state["tool_trace"], indent=2)

        # Prepare citations
        citations = []
        if isinstance(docs, list):
            for i, d in enumerate(docs):
                citations.append(f"[{i + 1}] {str(d)}")

        citation_text = "\n".join(citations) if citations else "No explicit citations available."

        system_message = """
        You are UdaPlay — a gaming research assistant who can summarise gaming research details
        """

        # Build conversation context
        conversation_context = ""
        if state.get("messages"):
            conversation_context = "\n".join(
                [
                    f"User: {m.get('query')}\nAssistant: {m.get('answer')}"
                    for m in state["messages"]
                    if isinstance(m, dict)
                ]
            )

        print(f"########  --- Conversation context : {conversation_context}")

        # LLM prompt — grounded & restricted
        synthesis_prompt = f"""      
            STRICT RULES:
            - Use ONLY the provided sources.
            - Do NOT hallucinate.
            - If info is missing, say so clearly.
            - If conversation context helps, use it carefully.
            - Write a clear, structured final answer.
        
            CONVERSATION CONTEXT:
            {conversation_context}

            USER QUESTION:
            {state["user_query"]}
        
            RETRIEVED SOURCES:
            {docs}
        
            TASK:
            Write a concise final answer grounded only in the sources.
            """

        llm = LLM(
            chat_model=DEFAULT_MODEL,
            temperature=0.0
        )

        llm_response = llm.invoke([
            SystemMessage(content=system_message),
            UserMessage(content=synthesis_prompt)
        ])

        final_answer_text = llm_response.content

        structured_report = f"""
        ===============================
        UdaPlay Research Report
        ===============================
    
        USER QUERY:
        {state["user_query"]}
    
        -------------------------------
        REASONING TRACE:
        {reasoning_log}
    
        -------------------------------
        TOOLS USED:
        {tool_trace}
        
        -------------------------------
        FINAL ANSWER (LLM-SYNTHESIZED):
        {final_answer_text}
    
        -------------------------------
        CITATIONS:
        {citation_text}
    
        ===============================
        """

        state["messages"].append({
            "query": state["user_query"],
            "answer": final_answer_text
        })
        return {"final_answer": structured_report}
