from langchain_ollama import ChatOllama



model = ChatOllama(model="deepseek-r1:1.5b", temperature=0.7)


for chunk in model.stream("Hi, how are you?"):
    print(chunk.content, end="", flush=True)

responses=model.batch(["Hi, how are you?", "How are you?","why parrot fly?"])
for response in responses:
    print(response.content)