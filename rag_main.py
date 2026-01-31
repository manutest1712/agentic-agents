
from Agents.udaplay_agent import UdaPlayAgent

from tools.game_tools import retrieve_game
from vector_store.game_vector_store import game_vector_store
from tools import tool_registry, evaluate_retrieval, game_web_search

game_vector_store.index_games("data/games")

print("retrieving games")
list_games = retrieve_game("Pokémon")

result = evaluate_retrieval("Pokemon games", list_games)

tools_metadata = tool_registry.list_tools()
print(tools_metadata)



tools = [retrieve_game, evaluate_retrieval, game_web_search]

agent = UdaPlayAgent(tools)

while True:
    q = input("\nAsk UdaPlay: ")
    print("\nAnswer:", agent.run(q))
# @tool
# def get_games_xx(num_games: int = 1, top: bool = True) -> str:
#     """
#     Returns the top or bottom N games with highest or lowest scores.
#     args:
#         num_games (int): Number of games to return (default is 1)
#         top (bool): If True, return top games, otherwise return bottom (default is True)
#     """
#     data = [
#         {"Game": "The Legend of Zelda: Breath of the Wild", "Platform": "Switch", "Score": 98},
#         {"Game": "Super Mario Odyssey", "Platform": "Switch", "Score": 97},
#         {"Game": "Metroid Prime", "Platform": "GameCube", "Score": 97},
#         {"Game": "Super Smash Bros. Brawl", "Platform": "Wii", "Score": 93},
#         {"Game": "Mario Kart 8 Deluxe", "Platform": "Switch", "Score": 92},
#         {"Game": "Fire Emblem: Awakening", "Platform": "3DS", "Score": 92},
#         {"Game": "Donkey Kong Country Returns", "Platform": "Wii", "Score": 87},
#         {"Game": "Luigi's Mansion 3", "Platform": "Switch", "Score": 86},
#         {"Game": "Pikmin 3", "Platform": "Wii U", "Score": 85},
#         {"Game": "Animal Crossing: New Leaf", "Platform": "3DS", "Score": 88}
#     ]
#     # Sort the games list by Score
#     # If top is True, descending order
#     sorted_games = sorted(data, key=lambda x: x['Score'], reverse=top)
#
#     # Return the N games
#     return sorted_games[:num_games]
#
#
# tools = [get_games_xx]
# def prepare_messages_step(state: AgentState) -> AgentState:
#     """Step logic: Prepare messages for LLM consumption"""
#
#     messages = [
#         SystemMessage(content=state["instructions"]),
#         UserMessage(content=state["user_query"])
#     ]
#
#     return {
#         "messages": messages
#     }
#
#
# def llm_step(state: AgentState) -> AgentState:
#     """Step logic: Process the current state through the LLM"""
#
#     # Initialize LLM
#     llm = LLM(
#         model="gpt-4o-mini",
#         temperature=0.3,
#         tools=tools,
#     )
#
#     response = llm.invoke(state["messages"])
#     tool_calls = response.tool_calls if response.tool_calls else None
#
#     # Create AI message with content and tool calls
#     ai_message = AIMessage(content=response.content, tool_calls=tool_calls)
#
#     return {
#         "messages": state["messages"] + [ai_message],
#         "current_tool_calls": tool_calls
#     }
#
#
# def tool_step(state: AgentState) -> AgentState:
#     """Step logic: Execute any pending tool calls"""
#     tool_calls = state["current_tool_calls"] or []
#     tool_messages = []
#
#     for call in tool_calls:
#         # Access tool call data correctly
#         function_name = call.function.name
#         function_args = json.loads(call.function.arguments)
#         tool_call_id = call.id
#         # Find the matching tool
#         tool = next((t for t in tools if t.name == function_name), None)
#         if tool:
#             result = tool(**function_args)
#             tool_messages.append(
#                 ToolMessage(
#                     content=json.dumps(result),
#                     tool_call_id=tool_call_id,
#                     name=function_name,
#                 )
#             )
#
#     # Clear tool calls and add results to messages
#     return {
#         "messages": state["messages"] + tool_messages,
#         "current_tool_calls": None
#     }
#
#
# # Create steps
# entry = EntryPoint[AgentState]()
# message_prep = Step[AgentState]("message_prep", prepare_messages_step)
# llm_processor = Step[AgentState]("llm_processor", llm_step)
# tool_executor = Step[AgentState]("tool_executor", tool_step)
# termination = Termination[AgentState]()
#
# workflow.add_steps(
#     [
#         entry,
#         message_prep,
#         llm_processor,
#         tool_executor,
#         termination
#     ]
# )
#
# entry = EntryPoint[AgentState]()
# message_prep = Step[AgentState]("message_prep", prepare_messages_step)
# llm_processor = Step[AgentState]("llm_processor", llm_step)
# tool_executor = Step[AgentState]("tool_executor", tool_step)
# termination = Termination[AgentState]()
#
# workflow = StateMachine[AgentState](AgentState)
#
# workflow.add_steps(
#     [
#         entry,
#         message_prep,
#         llm_processor,
#         tool_executor,
#         termination
#     ]
# )
#
# # Add transitions
# workflow.connect(entry, message_prep)
# workflow.connect(message_prep, llm_processor)
#
# # Transition based on whether there are tool calls
# def check_tool_calls(state: AgentState) -> Union[Step[AgentState], str]:
#     """Transition logic: Check if there are tool calls"""
#     if state.get("current_tool_calls"):
#         return tool_executor
#     return termination
#
# # Routing: If tool calls -> tool_executor
# workflow.connect(
#     source=llm_processor,
#     targets=[tool_executor, termination],
#     condition=check_tool_calls
# )
#
# # Looping: Go back to llm after tool execution
# workflow.connect(
#     source=tool_executor,
#     targets=llm_processor
# )
#
# initial_state: AgentState = {
#     "user_query": "What's the best game in the dataset?",
#     "instructions": "You can bring insights about a game dataset based on users questions",
#     "messages": [],
# }
#
# run_object = workflow.run(initial_state)