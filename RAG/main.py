from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader
import copy

loader=PyPDFLoader("Introduction_to_Data_and_Data_Science.pdf")
loader1=Docx2txtLoader("Introduction_to_Data_and_Data_Science.docx")


doc=loader.load()
doc_1=loader1.load()


print(doc)

print("doc_1",doc_1)

pages_pdf_cut=copy.deepcopy(doc)

' '.join(pages_pdf_cut[0].page_content.split())


for i in pages_pdf_cut:
    i.page_content=' '.join(i.page_content.split())

print(pages_pdf_cut)