from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


chat_template_topics = ChatPromptTemplate.from_template('''
   What are the top 5 trending topics of {category}  in {timeframe} ? 
   Answer only  by listing the topics.
   ''')


chat_template_projects = ChatPromptTemplate.from_template('''
    suggest three intresting {category} based projects.
   ''')


chat = ChatOllama(model = 'deepseek-r1:1.5b', 
                  temperature = 0)

string_paser=StrOutputParser()


chain_topics=(chat_template_topics|chat|string_paser)

chain_projects=(chat_template_projects|chat|string_paser)


chain_parallel = RunnableParallel({'topics': chain_topics, 'projects': chain_projects})

print(chain_parallel.invoke({'category':'Artificial intelligence', 'timeframe':'last 6 months'}))

chain_parallel.get_graph().print_ascii()