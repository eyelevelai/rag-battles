import time, os

from pinecone import Pinecone
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore


pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY_LI"),
)

dry_run = False


def upload_vectors(index_names, documents):
    # Update multiple indices
    for index_name in index_names:
        check1 = time.time()
        storage_context = StorageContext.from_defaults(
            vector_store=PineconeVectorStore(
                pinecone_index=pc.Index(
                    index_name,
                ),
            ),
        )
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )

        check2 = time.time()
        print(f"\tPinecone [{index_name}] updated [{check2 - check1:.4f}]")


def process_file_advanced(ty, dry, partition, folder, index_names):
    global dry_run, pc

    dry_run = dry

    print(f"\n\n[LI] [partition{partition}]\tProcessing files [{folder}]")

    if dry_run:
        for index_name in index_names:
            if index_name != 'rb3-li-naive-partition0':
                print(f"\tPinecone [{index_name}] updated")

        return

    start = time.time()

    if ty == 2:
        print('new advanced processing strategy')

        # this is naive code for loading documents

        documents = SimpleDirectoryReader(
            folder,
        ).load_data()

        check1 = time.time()
        print(f"\tLlamaIndex [{len(documents)}] documents processed [{check1 - start:.4f}]")

        upload_vectors(index_names, documents)

        # end naive code

    else:
        print(f"not set yet strategy [{ty}]")

    print(
        f"[partition{partition}]\tProcessing files complete [{folder}] [{time.time() - start:.4f}]\n"
    )


def process_file_naive(dry, partition, folder, index_names):
    global dry_run, pc

    dry_run = dry

    print(f"\n\n[LI] [partition{partition}]\tProcessing files [{folder}] FAST")

    if dry_run:
        for index_name in index_names:
            if index_name != 'rb3-li-naive-partition0':
                print(f"\tPinecone [{index_name}] updated")

        return

    start = time.time()

    documents = SimpleDirectoryReader(
        folder,
    ).load_data()

    check1 = time.time()
    print(f"\tLlamaIndex [{len(documents)}] documents processed [{check1 - start:.4f}]")

    # Update multiple indices
    upload_vectors(index_names, documents)

    print(
        f"[partition{partition}]\tProcessing files complete [{folder}] [{time.time() - start:.4f}]\n"
    )


def process(dry, ty, startp, endp, content_dir, partitions):
    global dry_run

    dry_run = dry

    for j, folder in enumerate(partitions):
        if (
            (startp < 0 or j >= startp) and
            (endp < 0 or j <= endp)
        ):
            nd = f"{content_dir}{folder}"
            if ty > 1:
                index_names = [
                    f"rb3-li-adv-{ty}-partition{k}" for k in range(j, len(partitions))
                ]
                process_file_advanced(ty, dry, j, nd, index_names)
            elif ty == 1:
                index_names = [
                    f"rb3-li-naive-partition{k}" for k in range(j, len(partitions))
                ]
                process_file_naive(dry, j, nd, index_names)

    print()
