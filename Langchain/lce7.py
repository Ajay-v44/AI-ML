from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


chat = ChatOllama(model = 'deepseek-r1:1.5b', 
                  temperature = 0)

runnable_sum=RunnableLambda(lambda x:sum(x))

my_list=[1,2,3,4,5]

print(runnable_sum.invoke(my_list))

runnable_square=RunnableLambda(lambda x: x*x)

print(runnable_square.invoke(10))

chain1=runnable_sum|runnable_square

print(chain1.invoke(my_list))

chain1.get_graph().print_ascii()
