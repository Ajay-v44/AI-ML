from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.runnables import RunnablePassthrough


chat_template_tools = ChatPromptTemplate.from_template('''
   What are the 5 most importanat  tools  a  {job_title} needs ?
   Answer only  by listing the tools.
   ''')


chat_template_strategy = ChatPromptTemplate.from_template('''
Considering the tools provided, develop a strategy for effectively learning and mastering them:
{tools}
''')


chat = ChatOllama(model = 'deepseek-r1:1.5b', 
                  temperature = 0)

string_paser=StrOutputParser()

chain_tools=chat_template_tools | chat  | string_paser |{'tools':RunnablePassthrough()}
chain_strategy=chat_template_strategy | chat | string_paser

tools=chain_tools.invoke({'job_title':'machine learning engineer'})
print('tools',tools)
print('\n##########################\n')
strategy=chain_strategy.invoke({'tools':tools})
print('strategy',strategy)

print('*'*50,'\n')
chain_combined=chain_tools|chain_strategy

print(chain_combined.invoke({'job_title':'Artificial Intelligence engineer'}))



