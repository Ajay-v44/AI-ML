from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# Initialize embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

chat = ChatOllama(model = 'deepseek-r1:1.5b', 
                  temperature = 0)


vector_store_from_directory = Chroma(
    persist_directory="./index",
    embedding_function=embeddings,
    collection_name="data_science_docs"
)


retriver=vector_store_from_directory.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 2, 'fetch_k': 10, 'lambda_mult': 0.5}
)

TEMPLATE = '''
Answer the following question:
{question}

To answer the question, use only the following context:
{context}

At the end of the response, specify the name of the lecture this context is taken from in the format:
Resources: *Lecture Title*
where *Lecture Title* should be substituted with the title of all resource lectures.
'''

prompt_template = PromptTemplate.from_template(TEMPLATE)

question="what software does data scientists use ?"


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


chain = ({'context': retriver | format_docs, 
         'question': RunnablePassthrough()} 
         | prompt_template 
         | chat 
         | StrOutputParser())


print(chain.invoke(question))

