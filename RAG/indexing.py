from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader
import copy
from langchain_ollama import OllamaEmbeddings
import numpy as np
from langchain_community.vectorstores import Chroma

loader=Docx2txtLoader("Introduction_to_Data_and_Data_Science.docx")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

pages=loader.load()

for i in range(len(pages)):
    pages[i].page_content=" ".join(pages[i].page_content.split())

character_splitter=CharacterTextSplitter(separator=".",chunk_size=1000,chunk_overlap=200)

pages_char_split=character_splitter.split_documents(pages)

print(pages_char_split)

for i in pages_char_split:
    i.metadata["embeddings"]=embeddings.embed_query(i.page_content)

print(pages_char_split[0])


print(pages_char_split[0].metadata["embeddings"])
print(pages_char_split[1].metadata["embeddings"])
print(pages_char_split[2].metadata["embeddings"])

print(np.dot(pages_char_split[0].metadata["embeddings"],pages_char_split[1].metadata["embeddings"]))
print(np.dot(pages_char_split[0].metadata["embeddings"],pages_char_split[2].metadata["embeddings"]))
print(np.dot(pages_char_split[1].metadata["embeddings"],pages_char_split[2].metadata["embeddings"]))

print(np.linalg.norm(pages_char_split[0].metadata["embeddings"]))
print(np.linalg.norm(pages_char_split[1].metadata["embeddings"]))
print(np.linalg.norm(pages_char_split[2].metadata["embeddings"]))


vector_store=Chroma.from_documents(documents=pages_char_split,
                                    embedding=embeddings,
                                    persist_directory="./index"
                                    )

vector_store_from_dircetory=Chroma(
    persist_directory="./index",
    embedding_function=embeddings
)

added_document=Document(page_content='Analysis vs Analytics Alright! So… Let’s discuss the not-so-obvious differences between the terms analysis and analytics. Due to the similarity of the words, some people believe they share the same meaning, and thus use them interchangeably. Technically, this isn’t correct. There is, in fact, a distinct difference between the two. And the reason for one often being used instead of the other is the lack of a transparent understanding of both. So, let’s clear this up, shall we? First, we will start with analysis. Consider the following… You have a huge dataset containing data of various types. Instead of tackling the entire dataset and running the risk of becoming overwhelmed, you separate it into easier to digest chunks and study them individually and examine how they relate to other parts. And that’s analysis in a nutshell. One important thing to remember, however, is that you perform analyses on things that have already happened in the past', metadata={'source': 'Introduction_to_Data_and_Data_Science.docx'})
print(vector_store.add_documents([added_document]))

print(vector_store.similarity_search("What is analytics"))

print(vector_store.delete('1bc478dd-559f-4c13-b9a0-ba4194b85e3b'))
