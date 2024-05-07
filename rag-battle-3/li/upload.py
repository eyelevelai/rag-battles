import time, os, zipfile

from dotenv import load_dotenv

load_dotenv()


if os.getenv("OPENAI_API_KEY") is None or os.getenv("PINECONE_API_KEY") is None:
    raise Exception(
        """


    You have not set a required environment variable (OPENAI_KEY, PINECONE_API_KEY, or PINECONE_HOST)
    Copy .env.sample and rename it to .env then fill in the missing values

"""
    )


#content_dir = '../Pa/'
#partitions = ['partition0']
content_dir = '../Partitions/'
partitions = ['partition0', 'partition1', 'partition2', 'partition3']


from pinecone import Pinecone
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore


pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
)


def process_file_advanced(partition, folder, index_names):
    global pc

    print(f'\n\n[partition{partition}]\tProcessing files [{folder}]')
    print(index_names)

    start = time.time()


def process_file_naive(partition, folder, index_names):
    global pc

    print(f'\n\n[partition{partition}]\tProcessing files [{folder}] FAST')

    start = time.time()

    documents = SimpleDirectoryReader(
        folder,
    ).load_data()

    check1 = time.time()
    print(f'\tLlamaIndex [{len(documents)}] documents processed [{check1 - start:.4f}]')

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
        print(f'\tPinecone [{index_name}] updated [{check2 - check1:.4f}]')

    print(f'[partition{partition}]\tProcessing files complete [{folder}] [{time.time() - start:.4f}]\n')


def process_ben(ty):
    for j, folder in enumerate(partitions):
        j = j
        nd = f"{content_dir}{folder}"
        if False:
            index_names = ['rb3-li-naive-partitionb']
            process_file_naive(j, nd, index_names)
        elif ty == 2:
            index_names = [f'rb3-li-advanced-partition{k}' for k in range(j, len(partitions))]
            process_file_advanced(j, nd, index_names)
        else:
            index_names = [f'rb3-li-naive-partition{k}' for k in range(j, len(partitions))]
            process_file_naive(j, nd, index_names)

    print()


files = os.listdir(content_dir)
for _, file in enumerate(files):
    dir = f"{content_dir}{file.split('.')[0]}"

    if os.path.exists(dir) is False:
        print(f"[{dir}] unzipping")

        os.mkdir(dir)
        with zipfile.ZipFile(content_dir+file, 'r') as zip_ref:
            zip_ref.extractall(dir)


process_ben(1)