import pandas as pd
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv, find_dotenv
import pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

files=pd.read_csv("course_descriptions.csv",encoding='ANSI')
print(files.head())



def create_course_description(row):
    return f'''The course name is {row["course_name"]}, the slug is {row["course_slug"]},
            the technology is {row["course_technology"]} and the course topic is {row["course_topic"]}'''


pd.set_option('display.max_rows', 106)
files['course_description_new'] = files.apply(create_course_description, axis = 1)
print(files["course_description_new"])


pc = Pinecone(api_key = os.environ.get("PINECONE_API_KEY"))


index_name = "my-index"
dimension = 768
metric = "cosine"


if index_name in [index.name for index in pc.list_indexes()]:
    pc.delete_index(index_name)
    print(f"{index_name} succesfully deleted.")
else:
     print(f"{index_name} not in index list.")


pc.create_index(
    name = index_name, 
    dimension = dimension, 
    metric = metric, 
    spec = ServerlessSpec(
        cloud = "aws", 
        region = "us-east-1")
    )

index = pc.Index(index_name)

model = SentenceTransformer('multi-qa-distilbert-cos-v1')


def create_embeddings(row):
    combined_text = ' '.join([str(row[field]) for field in ['course_description', 'course_description_new', 'course_description_short']])
    embedding = model.encode(combined_text, show_progress_bar = False)
    return embedding

files["embedding"] = files.apply(create_embeddings, axis = 1)

vectors_to_upsert = [(str(row["course_name"]), row["embedding"].tolist()) for _, row in files.iterrows()]
index.upsert(vectors = vectors_to_upsert)

print("Data upserted to Pinecone index")

query="clustering"
query_embedding=model.encode(query,show_progress_bar = False).tolist()

result=index.query(vector=query_embedding,top_k = 12,include_values = True)

print(result["matches"])