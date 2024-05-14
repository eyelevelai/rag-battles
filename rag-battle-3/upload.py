import time, os, zipfile

from dotenv import load_dotenv

load_dotenv()


if (
    os.getenv("OPENAI_API_KEY") is None
    or os.getenv("PINECONE_API_KEY") is None
    or os.getenv("TESSDATA_PREFIX") is None
):
    raise Exception(
        """


    You have not set a required environment variable (OPENAI_KEY, PINECONE_API_KEY, or PINECONE_HOST)
    Copy .env.sample and rename it to .env then fill in the missing values

"""
    )


import lcpc.upload as lcpc
import li.upload as li


content_dir = "Partitions/"
partitions = ["partition0", "partition1", "partition2", "partition3"]
#content_dir = "Pa/"
#partitions = ["partition0", "partition1"]

dry_run = False
start_partition = 1


files = os.listdir(content_dir)
for _, file in enumerate(files):
    dir = f"{content_dir}{file.split('.')[0]}"

    if os.path.exists(dir) is False:
        print(f"[{dir}] unzipping")

        os.mkdir(dir)
        with zipfile.ZipFile(content_dir + file, "r") as zip_ref:
            zip_ref.extractall(dir)


lcpc.process_ben(dry_run, start_partition, content_dir, partitions)
li.process(dry_run, 1, start_partition, content_dir, partitions)