import os

from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
)

idx = pc.Index("rb3-li-naive-partition0")
print(idx.describe_index_stats())
idx = pc.Index("rb3-li-naive-partition1")
print(idx.describe_index_stats())
idx = pc.Index("rb3-li-naive-partition2")
print(idx.describe_index_stats())
idx = pc.Index("rb3-li-naive-partition3")
print(idx.describe_index_stats())