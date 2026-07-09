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


class State(TypedDict):
    messages: Annotated[list,add_messages]
    
graph=StateGraph(State)


llm=ChatOllama(model='llama3.2:latest',temperature=0)

def bot(stream:State)->State:
    return {'messages': [llm.invoke(stream['messages'])]}

graph.add_node('bot',bot)
graph.add_edge(START,'bot')
graph.add_edge('bot',END)

state={
    'messages': [("user", "Hi, how are you?")]
}
graph=graph.compile()
config={'configurable':{'thread_id':'1'}}
for chunk in graph.stream(state,config,stream_mode='updates'):
    print(chunk)
    # for val in chunk.values():
    #     print(val['messages'][-1].content, end=" ")
