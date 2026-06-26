from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser


chat_template = ChatPromptTemplate.from_messages([
    ('human', 
     "I've recently adopted a {pet} which is a {breed} Could you suggest several tarinig  steps for it? ")])



chat = ChatOllama(model = 'deepseek-r1:1.5b', 
                  temperature = 0)

chain=chat_template | chat

chat_template_result = chat_template.invoke({'pet':'dog','breed':'German Sheperd'})

chat_result = chat.invoke(chat_template_result)

print(chat_result.content)
res=chain.batch([{'pet':'dog','breed':'German Sheperd'},
            {'pet':'cat','breed':'tabby'}, {'pet':'parakeet','breed':'ringneck'}])

for i in res:
    print(i.content)

print('#######################\n')

res=chain.stream({'pet':'dog','breed':'German Sheperd'})

for i in res:
    print(i.content,end='',flush=True)