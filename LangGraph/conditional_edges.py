from typing import Annotated
from langchain_core.messages import AIMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END,add_messages
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage,RemoveMessage
from langchain_core.runnables import Runnable
from collections.abc import Sequence


chat = ChatOllama(model = 'deepseek-r1:1.5b', 
                  temperature = 0)

my_list=add_messages([HumanMessage("Hi I'm Ajay"),
                      AIMessage("Hey Ajay .How are you ?")],
                    [ HumanMessage("I'm fine ")])
# Define State

class State(TypedDict, total=False):
    """Graph state."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    ask_again: str


state = State(messages=[HumanMessage(content="What's the weather like?")])


def ask_question(state: State) -> dict:
    print("whats your question ?")
    return {"messages": [HumanMessage(content=input())]}

def chatbot(state: State) -> dict:
    response = chat.invoke(state['messages'])
    response.pretty_print()
    return {"messages": [response]}

def ask_question_more(state: State) -> dict:
    print("Do you want to ask one more question ? (yes/no)")
    return {"ask_again": input().strip().lower()}

# trimming messages
def trim_messages(state: State) -> dict:
    remove_messages=[RemoveMessage(id=i.id)for i in state['messages'][:-5]]
    return {"messages": remove_messages}

# Defining Routing Function
def routing_function(state: State) -> Literal["ask_question", "__end__"]:
    if state.get("ask_again") == "yes":
        return "ask_question"
    else:
        return "__end__"

# Build Graph
graph = StateGraph(State)
graph.add_node('ask_question', ask_question)
graph.add_node('chatbot', chatbot)
graph.add_node('ask_question_more', ask_question_more)
graph.add_node('trim_messages', trim_messages)

graph.add_edge(START, 'ask_question')
graph.add_edge('ask_question', 'chatbot')
graph.add_edge('chatbot', 'ask_question_more')
graph.add_conditional_edges(source="ask_question_more", path=routing_function, path_map={"ask_question": "trim_messages", "__end__": END})
graph.add_edge('trim_messages','ask_question')
graph_comiled = graph.compile()

print(graph_comiled.get_graph().draw_ascii())

graph_comiled.invoke(State(messages=[]))