import os, time


from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone

from unstructured.partition.utils.constants import PartitionStrategy


dry_run = False


def process_file(dry, partition, file_name, folder, index_names):
    global dry_run

    dry_run = dry

    addendum = ""
    # Load documents
    loader_args = None
    if partition > 0:
        loader_args = {"strategy": PartitionStrategy.FAST}
        addendum = " FAST"

    print(
        f"\n\n[LCPC] [partition{partition}]\tProcessing file [{folder}/{file_name}]{addendum}"
    )

    if dry_run:
        for index_name in index_names:
            if index_name != "rb3-lcpc-partition0":
                print(f"\tPinecone [{index_name}] updated")

        return

    start = time.time()

    docs = DirectoryLoader(
        folder, glob=file_name, use_multithreading=True, loader_kwargs=loader_args
    ).load()

    check1 = time.time()
    print(f"\tUnstructured [{len(docs)}] doc processed [{check1 - start:.4f}]")

    # Split documents into texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    check2 = time.time()
    print(f"\tChunk [{file_name}]  [{len(texts)}] chunks [{check2 - check1:.4f}]")

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_KEY"))

    # Update multiple indices
    for index_name in index_names:
        if index_name != "rb3-lcpc-partition0":
            check1 = time.time()
            Pinecone.from_documents(texts, embeddings, index_name=index_name)

            check2 = time.time()
            print(f"\tPinecone [{index_name}] updated [{check2 - check1:.4f}]")

    print(
        f"[partition{partition}]\tProcessing file complete [{folder}/{file_name}] [{time.time() - start:.4f}]\n"
    )


def process(dry, startp, endp, content_dir, partitions):
    global dry_run

    dry_run = dry

    for j, folder in enumerate(partitions):
        if (startp < 0 or j >= startp) and (endp < 0 or j <= endp):
            nd = f"{content_dir}{folder}"
            index_names = [f"rb3-lcpc-partition{k}" for k in range(j, len(partitions))]
            for file_name in os.listdir(nd):
                process_file(dry, j, file_name, nd, index_names)
    print()
