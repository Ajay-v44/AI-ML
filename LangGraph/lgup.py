from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph, START, END

from langchain_ollama import ChatOllama

from langchain_core.runnables import Runnable
from collections.abc import Sequence

from langgraph.prebuilt import create_react_agent

from langchain.tools import tool


class State(TypedDict):
    messages: Annotated[list,add_messages]

graph_builder = StateGraph(State)

chat = ChatOllama(model = 'llama3.2:latest', 
                  temperature = 0)

def chatbot(state:State)->State:
    return {"messages":[chat.invoke(state['messages'])]}

graph_builder.add_node("chatbot",chatbot)
graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot",END)

workflow = graph_builder.compile()

# graph_builder.print_ascii()

response=workflow.invoke({"messages": [("user", "Hi, how are you?")]})

print(response['messages'][-1].content)

for chunk in workflow.stream({"messages": [("user", "Hi, how are you?")]}):
    for val in chunk.values():
        print(val['messages'][-1].content, end=" ")
