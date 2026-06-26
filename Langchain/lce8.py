from langchain_core.runnables import RunnableLambda,chain


def find_sum(x):
    return sum(x)


def find_square(x):
    return x*x


@chain
def runnable_sum(x):
    return sum(x)

@chain
def runnable_square(x):
    return x*x

chain1=RunnableLambda(find_sum)|RunnableLambda(find_square)

my_list=[1,2,3,4,5]

print(chain1.invoke(my_list))

chain2=runnable_sum | runnable_square

print("\n")

print(chain2.invoke(my_list))

chain1.get_graph().print_ascii()
