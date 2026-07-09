from langchain_core.runnables import config
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph, START, END

from langchain_ollama import ChatOllama

from langchain_core.runnables import Runnable
from collections.abc import Sequence

from langgraph.prebuilt import create_react_agent

from langchain.tools import tool

memory=InMemorySaver()
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
config={"configurable":{"thread_id":"1"}}
workflow = graph_builder.compile(checkpointer=memory)

# graph_builder.print_ascii()

response=workflow.invoke({"messages": [("user", "Hi, how are you?")]}, config)

print(response['messages'][-1].content)

for chunk in workflow.stream({"messages": [("user", "Hi, how are you?")]}, config):
    for val in chunk.values():
        print(val['messages'][-1].content, end=" ")


response=workflow.invoke({"messages": [("user", "Hi, how are you ?,i AM ajay.")]},config)
print('\n\n\n\n\n')
print(response['messages'][-1].content)


response=workflow.invoke({"messages": [("user", "What is my name ?")]},config)
print('\n\n\n\n\n')
print(response['messages'][-1].content)
