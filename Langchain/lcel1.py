from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser

list_instructions = CommaSeparatedListOutputParser().get_format_instructions()

chat_template = ChatPromptTemplate.from_messages([
    ('human', 
     "I've recently adopted a {pet}. Could you suggest three {pet} names? \n" + list_instructions)])


print(chat_template.messages[0].prompt.template)

chat = ChatOllama(model = 'deepseek-r1:1.5b', 
                  temperature = 0)

list_output_parser = CommaSeparatedListOutputParser()

chat_template_result = chat_template.invoke({'pet':'dog'})

chat_result = chat.invoke(chat_template_result)


list_output_parser.invoke(chat_result)



chain = chat_template | chat | list_output_parser

result=chain.invoke({'pet':'dog'})

print(result)