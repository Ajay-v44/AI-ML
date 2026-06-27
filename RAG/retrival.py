from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# Initialize embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")


vector_store_from_directory = Chroma(
    persist_directory="./index",
    embedding_function=embeddings,
    collection_name="data_science_docs"
)

question="what is python ?"

retriver=vector_store_from_directory.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 2, 'fetch_k': 10, 'lambda_mult': 0.5}
)

retrived_docs=retriver.invoke(question)

for i in retrived_docs:
    print(i.page_content)


print("*"*100)

question2="which library is used for data analysis ?"

retrived_docs2=retriver.invoke(question2)

for i in retrived_docs2:
    print(i.page_content)


