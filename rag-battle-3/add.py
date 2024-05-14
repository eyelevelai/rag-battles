import os

from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone

import lcpc.upload as lcupload
import li.upload as liupload


dry_run = True
partition = 1

folder = "Partitions/partition1-addendum"
addfiles = ["ct_audit_manual.pdf"]


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

lcpc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
)
lipc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY_LI"),
)

for f in addfiles:
    lcupload.process_file(dry_run, partition, f, folder, lcpcidx)


liupload.process_file_naive(dry_run, partition, folder, lipcidx)