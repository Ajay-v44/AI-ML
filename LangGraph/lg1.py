from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import Runnable
from collections.abc import Sequence


chat = ChatOllama(model = 'deepseek-r1:1.5b', 
                  temperature = 0)


# Define State

class State(TypedDict):
    """Graph state.
    
    Attributes:
        messages: messages to pass to the llm (typed as a sequence of messages)
    """
    messages: Sequence[BaseMessage]


state=State(messages = [HumanMessage(content = "What's the weather like?")])

def chatbot(state:State)->State:
 response=chat.invoke(state['messages'])
 response.pretty_print()
 return State(messages=[response.content])


# Define Graph

graph=StateGraph(State)
graph.add_node('chatbot',chatbot)

graph.add_edge(START,'chatbot')
graph.add_edge('chatbot',END)

graph=graph.compile()

print(graph)
print(graph.invoke(state))


