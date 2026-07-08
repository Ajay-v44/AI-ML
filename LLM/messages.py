from langchain.messages import HumanMessage, AIMessage,SystemMessage
from langchain_ollama import ChatOllama

model = ChatOllama(model="deepseek-r1:1.5b", temperature=0.7)

messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Hi, how are you?"),
    AIMessage(content="I am fine, thank you. How can I help you?"),
    HumanMessage(content="What is the capital of France?")
]

reponse=model.invoke(messages)
print(reponse.content)