import os

from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone

import lcpc.upload as lcupload
import li.upload as liupload


dry_run = False
partition = 1


folder = "Pa/partition1"
addfiles = [
    "usc42_ch40to81_Secs3271to6892@118-44not42.pdf",
    "usc45@118-44not42.pdf",
]


lcpcidx = [
    "rb3-lcpc-partition1",
    "rb3-lcpc-partition2",
    "rb3-lcpc-partition3",
]
lipcidx = [
    "rb3-li-naive-partition1",
    "rb3-li-naive-partition2",
    "rb3-li-naive-partition3",
]


if __name__ == "__main__":
    lcpc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
    )
    lipc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY_LI"),
    )

    for f in addfiles:
        lcupload.process_file(dry_run, partition, f, folder, lcpcidx)

    liupload.process_file_naive(dry_run, partition, folder, lipcidx)