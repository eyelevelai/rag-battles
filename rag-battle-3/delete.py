import os

from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone


dry_run = False


lcpcidx = [
    "rb3-lcpc-partition1",
    #"rb3-lcpc-partition2",
    #"rb3-lcpc-partition3",
]
lipcidx = [
    "rb3-li-naive-partition1",
    #"rb3-li-naive-partition2",
    #"rb3-li-naive-partition3",
]


delfiles = [
    "pwc-worldwide-tax-summaries-corporate-2013-14-unlocked.pdf",
]


lcpc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
)
lipc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY_LI"),
)


def deleteRecords(index, records):
    deleted = 0
    for i in range(0, len(records), 1000):
        group = records[i:i + 1000]
        index.delete(ids=group)
        deleted += len(group)

    print(f"deleted [{deleted}] of [{len(records)}] records")


print()
print(delfiles)

for lidx in lcpcidx:
    index = lcpc.Index(lidx)

    print(f"\nchecking [{lidx}]\n")
    deletes = []
    for ids in index.list():
        records = index.fetch(ids)
        for n in records['vectors']:
            id = records['vectors'][n]['id']
            source = records['vectors'][n]['metadata']['source']
            for d in delfiles:
                if d.lower() in source.lower():
                    deletes.append(id)
                if dry_run:
                    break
            if dry_run:
                break 
        if dry_run:
            break

    print(f"found [{len(deletes)}] records that match delete files")
    if dry_run:
        break
    if len(deletes) > 0:
        print("deleting")
        deleteRecords(index, deletes)

for lidx in lipcidx:
    index = lipc.Index(lidx)

    print(f"\nchecking [{lidx}]\n")
    deletes = []
    for ids in index.list():
        records = index.fetch(ids)
        for n in records['vectors']:
            id = records['vectors'][n]['id']
            source = records['vectors'][n]['metadata']['file_name']
            for d in delfiles:
                if d.lower() in source.lower():
                    deletes.append(id)
                if dry_run:
                    break
            if dry_run:
                break 
        if dry_run:
            break

    print(f"found [{len(deletes)}] records that match delete files")
    if dry_run:
        break
    if len(deletes) > 0:
        print("deleting")
        deleteRecords(index, deletes)