from langchain_ollama import ChatOllama
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

model = ChatOllama(model="deepseek-r1:1.5b", temperature=0.7,tools=[add])



for chunk in model.stream("what is 1+1"):
    print(chunk.content, end="", flush=True)
 