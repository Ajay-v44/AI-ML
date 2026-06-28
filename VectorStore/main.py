from pinecone import ServerlessSpec
from pinecone import Pinecone
import os
from  dotenv import load_dotenv

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("myindex")
print(index.describe_index_stats())

print(pc.list_indexes())

# pc.create_index("myindex2" , dimension=3, metric="cosine",spec=ServerlessSpec(
#     cloud='aws',
#     region='us-east-1'
# ))

print(pc.list_indexes())

index=pc.Index("myindex2")
print(index.describe_index_stats())

index.upsert(
    vectors=[
        {
            "id": "vec1",
            "values": [0.1, 0.2, 0.3],
            "metadata": {"name": "vec1"}
        },
        {
            "id": "vec2",
            "values": [0.4, 0.5, 0.6],
            "metadata": {"name": "vec2"}
        },
        {
            "id": "vec3",
            "values": [0.7, 0.8, 0.9],
            "metadata": {"name": "vec3"}
        }
    ]
)

index.query(
    vector=[0.1, 0.2, 0.3],
    top_k=5,
    include_values=True,
    include_metadata=True
)
