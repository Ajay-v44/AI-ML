from langchain_core.prompts.chat import AIMessagePromptTemplate
from langchain_core.prompts.chat import HumanMessagePromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage
from langchain_core.prompts import FewShotChatMessagePromptTemplate

model=ChatOllama(model="deepseek-r1:1.5b",temperature=0,top_p=0)
messages = [
    # SystemMessage(content="You are a helpful bmw seller."),
    HumanMessage(content="Hello, tell me about bmw 730li vs bmw m3"),
    AIMessage(content="this is summary for the above query"),
    HumanMessage(content="What is the price of each car?"),
    AIMessage(content="this is summary for the above query"),
    HumanMessage(content="Are there any other cars you would recommend?")
]
response=model.invoke(messages)

print(response.content)

TEMPLATE = """

System :
{description}

Human :
I've bought {product} ,how should i use it ?


"""
prompt_template=PromptTemplate.from_template(TEMPLATE)
prompt=prompt_template.invoke({"description":"You are a helpful car seller.","product":"bmw 730li"})
print(prompt)
response=model.invoke(prompt)
print(response.content)

chat_template=ChatPromptTemplate.from_template(TEMPLATE)
prompt=chat_template.invoke({"description":"You are a helpful car seller.","product":"bmw 730li"})
response=model.invoke(prompt)
print("\n",response.content)


print("############################################")

TEMPLATE_AI='''{response}'''

message_template_h=HumanMessagePromptTemplate.from_template(TEMPLATE)
message_template_ai=AIMessagePromptTemplate.from_template(TEMPLATE_AI)

example_templates=ChatPromptTemplate.from_messages([message_template_h,message_template_ai])

examples=[
    {"description":"You are a helpful car seller.","product":"bmw 730li","response":"Hello, how can i help you ?I am a car seller"},
    {"description":"You are a helpful car seller.","product":"bmw m3","response":"hey here is my response"}
]
few_shot_prompt=FewShotChatMessagePromptTemplate(example_prompt=example_templates, examples=examples)


chat_template=ChatPromptTemplate.from_messages([few_shot_prompt,message_template_h])
prompt=chat_template.invoke({"description":"You are a helpful car seller.","product":"bmw 730li"})
response=model.invoke(prompt)
print("\n",response.content)

for i in prompt.messages:
    print(i.content)