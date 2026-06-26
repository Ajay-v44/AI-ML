from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel


chat_template_books = ChatPromptTemplate.from_template(
    '''
    Suggest three of the best intermediate-level {programming language} books. 
    Answer only by listing the books.
    '''
)

chat_template_projects = ChatPromptTemplate.from_template(
    '''
    Suggest three interesting {programming language} projects suitable for intermediate-level programmers. 
    Answer only by listing the projects.
    '''
)

chat_template_time = ChatPromptTemplate.from_template(
     '''
     I'm an intermediate level programmer.
     
     Consider the following literature:
     {books}
     
     Also, consider the following projects:
     {projects}
     
     Roughly how much time would it take me to complete the literature and the projects?
     
     '''
)


chat = ChatOllama(model = 'deepseek-r1:1.5b', 
                  temperature = 0)

string_paser=StrOutputParser()

chain_books=chat_template_books|chat|string_paser

chain_projects=chat_template_projects|chat|string_paser

chain_parallel=RunnableParallel({'books':chain_books,'projects':chain_projects})

print(chain_parallel.invoke({'programming language':'Python'}))

chain_time = (RunnableParallel({'books':chain_books, 
                                'projects':chain_projects}) 
              | chat_template_time 
              | chat 
              | string_paser
             )
             
print(chain_time.invoke({'programming language':'Python'}))

print(chain_time.get_graph().print_ascii())