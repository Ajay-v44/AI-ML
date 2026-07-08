from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
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

@tool
def add(a:int , b:int )->int:
    """Add two integers."""
    return a + b
@tool
def sub(a:int , b:int )->int:
    """sub two integers."""
    return a - b
@tool
def mul(a:int , b:int )->int:
    """mul two integers."""
    return a * b
@tool
def div(a:int , b:int )->int:
    """div two integers."""
    return a / b    
@tool
def get_online_data(query:str):
    """Get online data from web"""
    return "no data ,its a mock data."

chat = ChatOllama(model = 'llama3.2:latest', 
                  temperature = 0)

def tool_caler_llm(state:State) :
    return {"messages":[chat.invoke(state['messages'])]}

tools=[add,sub,mul,div,get_online_data]

llm = create_react_agent(
        chat, tools=tools
    )

graph_builder.add_node("tool_caler_llm",llm)
graph_builder.add_node("tools",ToolNode(tools=tools))

graph_builder.add_edge(START,"tool_caler_llm")
graph_builder.add_conditional_edges(
    "tool_caler_llm",
    tools_condition
)
graph_builder.add_edge("tools","tool_caler_llm")

workflow = graph_builder.compile()
# response=workflow.invoke({"messages": [("user", "what is 23 * 45 ")]})
# for m in response['messages']:
#     print(m.pretty_print())
# print("*"*10)
# for chunk in workflow.stream({"messages": [("user", "what is 23 * 45 ")]}):
#     for val in chunk.values():
#         print(val['messages'][-1].content, end=" ")

# print("\n"+"="*10)

response=workflow.invoke({"messages": [("user", "what is latest online news from web, add 5+10 ")]})
for m in response['messages']:
    print(m.pretty_print())

