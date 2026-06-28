import re
from langgraph.graph import MessagesState
from typing import Annotated
from langchain_core.messages import AIMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END, add_messages
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import Runnable
from collections.abc import Sequence


def clean_text(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


class State(MessagesState, total=False):
    summary: str
    ask_again: str


chat = ChatOllama(model='deepseek-r1:1.5b', temperature=0)


def ask_question(state: State) -> dict:
    question = "whats your question ?"
    print(question)
    return {"messages": [AIMessage(content=question), HumanMessage(content=input())]}


def chatbot(state: State) -> dict:
    system_message = f'''
        here's a quick summary of what's discussed so far :
        {state.get("summary","")}
        keep this in mind as you answer the next question.
    '''
    response = chat.invoke([SystemMessage(system_message)] + state["messages"])
    response.pretty_print()
    return {"messages": [response]}


def ask_question_more(state: State) -> dict:
    print("Do you want to ask one more question ? (yes/no)")
    return {"ask_again": input().strip().lower()}


# trimming messages
def summarize_and_delete_messages(state: State) -> dict:
    new_conversation = ""
    for msg in state['messages']:
        new_conversation += f"{msg.type}: {clean_text(msg.content)} \n\n"
        
    summary_instructions = f'''
    Update the ongoing summary by incorporating the new lines of conversation below.  
    Build upon the previous summary rather than repeating it so that the result  
    reflects the most recent context and developments.


    Previous Summary:
    {state.get("summary", "")}

    New Conversation:
    {new_conversation}
    '''
    summary = chat.invoke([HumanMessage(summary_instructions)])
    remove_messages = [RemoveMessage(id=i.id) for i in state['messages'][:]]
    return {"messages": remove_messages, "summary": clean_text(summary.content)}


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
graph.add_node('summarize_and_delete_messages', summarize_and_delete_messages)

graph.add_edge(START, 'ask_question')
graph.add_edge('ask_question', 'chatbot')
graph.add_edge('chatbot', 'ask_question_more')
graph.add_conditional_edges(source="ask_question_more", path=routing_function, path_map={"ask_question": "summarize_and_delete_messages", "__end__": END})
graph.add_edge('summarize_and_delete_messages', 'ask_question')
graph_comiled = graph.compile()

print(graph_comiled.get_graph().draw_ascii())

graph_comiled.invoke(State(messages=[]))
