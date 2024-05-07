import multiprocessing, os, time, zipfile

from dotenv import load_dotenv

load_dotenv()


if os.getenv("OPENAI_API_KEY") is None or os.getenv("PINECONE_API_KEY") is None or os.getenv("TESSDATA_PREFIX") is None:
    raise Exception(
        """


    You have not set a required environment variable (OPENAI_KEY, PINECONE_API_KEY, or TESSDATA_PREFIX)
    Copy .env.sample and rename it to .env then fill in the missing values

"""
    )

from tqdm import tqdm

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone

from unstructured.partition.utils.constants import PartitionStrategy

content_dir = '../Partitions/'
partitions = ['partition0', 'partition1', 'partition2', 'partition3']


def process_file(partition, file_name, folder, index_names):
    addendum = ''
    # Load documents
    loader_args = None
    if partition > 0:
        loader_args = {"strategy": PartitionStrategy.FAST}
        addendum = ' FAST'

    print(f'\n\n[partition{partition}]\tProcessing file [{folder}/{file_name}]{addendum}')

    start = time.time()

    docs = DirectoryLoader(folder, glob=file_name, use_multithreading=True, loader_kwargs=loader_args).load()

    check1 = time.time()
    print(f'\tUnstructured [{len(docs)}] doc processed [{check1 - start:.4f}]')

    # Split documents into texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    check2 = time.time()
    print(f'\tChunk [{file_name}]  [{len(texts)}] chunks [{check2 - check1:.4f}]')

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_KEY"))

    # Update multiple indices
    for index_name in index_names:
        check1 = time.time()
        Pinecone.from_documents(texts, embeddings, index_name=index_name)

        check2 = time.time()
        print(f'\tPinecone [{index_name}] updated [{check2 - check1:.4f}]')

    print(f'[partition{partition}]\tProcessing file complete [{folder}/{file_name}] [{time.time() - start:.4f}]\n')


def worker_task(task_queue):
    while True:
        task = task_queue.get()
        if task is None:  # Sentinel value to signal end of tasks
            break

        file_name, folder, index_names = task
        process_file(file_name, folder, index_names)

        task_queue.task_done()


def process_folders(folders, num_processes, api_key):
    task_queue = multiprocessing.JoinableQueue()

    # Start worker processes
    processes = [multiprocessing.Process(target=worker_task, args=(task_queue, api_key))
                 for _ in range(num_processes)]
    for p in processes:
        p.start()

    # Enqueue tasks with corresponding index names
    num_partitions = len(folders)
    for i, folder in enumerate(folders):
        index_names = [f'rb3-lcpc-partition{j}' for j in range(i, num_partitions)]
        for file_name in os.listdir(folder):
            task_queue.put((file_name, folder, index_names))

    # Signal workers to finish
    for _ in range(num_processes):
        task_queue.put(None)

    task_queue.join()

    # End worker processes
    for p in processes:
        p.join()


def process_ben():
    for j, folder in enumerate(partitions):
        nd = f"{content_dir}{folder}"
        index_names = [f'rb3-lcpc-partition{k}' for k in range(j, len(partitions))]
        for file_name in os.listdir(nd):
            process_file(j, file_name, nd, index_names)
    print()


files = os.listdir(content_dir)
for _, file in enumerate(files):
    dir = f"{content_dir}{file.split('.')[0]}"

    if os.path.exists(dir) is False:
        print(f"[{dir}] unzipping")

        os.mkdir(dir)
        with zipfile.ZipFile(content_dir+file, 'r') as zip_ref:
            zip_ref.extractall(dir)


process_ben()