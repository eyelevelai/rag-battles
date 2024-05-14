import os

from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone


lcpc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
)
lipc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY_LI"),
)


lcpcidx = [
    "rb3-lcpc-partition0",
    "rb3-lcpc-partition1",
    "rb3-lcpc-partition2",
    "rb3-lcpc-partition3",
]
lipcidx = [
    "rb3-li-naive-partition0",
    "rb3-li-naive-partition1",
    "rb3-li-naive-partition2",
    "rb3-li-naive-partition3",
]


for id in lcpcidx:
    idx = lcpc.Index(id)

    print(id)
    print(idx.describe_index_stats())
    print()

for id in lipcidx:
    idx = lipc.Index(id)

    print(id)
    print(idx.describe_index_stats())
    print()