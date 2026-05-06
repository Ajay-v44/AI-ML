from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from embeding import retriever

model=OllamaLLM(model="llama3.2")

template="""
    You are an exeprt in answering questions about a pizza restaurant

    Here are some relevant reviews: {reviews}

    Here is the question to answer: {question}
    """


prompt=ChatPromptTemplate.from_template(template)

chain=prompt|model


while True:
    question = input("Enter your question: ")
    if question=="q":
        break
    reviews=retriever.invoke(question)
    result=chain.invoke(
        {
            "reviews": reviews,
            "question": question
            }
        )
    print(result)


