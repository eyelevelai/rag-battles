import multiprocessing, os, time, zipfile

from dotenv import load_dotenv
from tqdm import tqdm

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone

content_dir = 'Partitions/'

load_dotenv()


if os.getenv("OPENAI_API_KEY") is None or os.getenv("PINECONE_API_KEY") is None:
    raise Exception(
        """


    You have not set a required environment variable (OPENAI_KEY)
    Copy .env.sample and rename it to .env then fill in the missing values

"""
    )


def process_file(file_name, folder, index_names):
    print(f'\n\nProcessing file [{folder}{file_name}]\n')

    start = time.time()

    # Load documents
    docs = DirectoryLoader(folder, glob=file_name, use_multithreading=True).load()

    check1 = time.time()
    print(f'\t[{check1 - start:.4f}] Loaded [{len(docs)}] documents for [{file_name}]\n')

    # Split documents into texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    check2 = time.time()
    print(f'\t[{check2 - check1:.4f}] Split [{file_name}]\n')

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_KEY"))

    # Update multiple indices
    for index_name in index_names:
        check1 = time.time()
        docsearch = Pinecone.from_documents(texts, embeddings, index_name=index_name)

        check2 = time.time()
        print(f'\t[{check2 - check1:.4f}] Updated {index_name} with {len(texts)} texts')

    print(f'\n\t[{time.time() - start:.4f}] Processing file complete [{folder}{file_name}]\n\n')


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


def process_ben(folders):
    for _, folder in enumerate(folders):
        nd = f"{content_dir}{folder}"
        for file_name in os.listdir(nd):
            process_file(file_name, nd, ['rb-2-pb'])


files = os.listdir(content_dir)
for _, file in enumerate(files):
    dir = f"{content_dir}{file.split('.')[0]}"

    if os.path.exists(dir) is False:
        print(f"[{dir}] unzipping")

        os.mkdir(dir)
        with zipfile.ZipFile(content_dir+file, 'r') as zip_ref:
            zip_ref.extractall(dir)


process_ben(['partitionb'])