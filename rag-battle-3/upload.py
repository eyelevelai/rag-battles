import os, zipfile

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


runGroundX = False
runLCPC = False
runLIPC = False


content_dir = "Partitions/"
partitions = ["partition0", "partition1", "partition2", "partition3"]

dry_run = False
start_partition = 0
end_partition = 0


if __name__ == "__main__":
    files = os.listdir(content_dir)
    for _, file in enumerate(files):
        dir = f"{content_dir}{file.split('.')[0]}"

        if os.path.exists(dir) is False:
            print(f"[{dir}] unzipping")

            os.mkdir(dir)
            with zipfile.ZipFile(content_dir + file, "r") as zip_ref:
                zip_ref.extractall(dir)

    if runLCPC:
        lcpc.process(dry_run, start_partition, end_partition, content_dir, partitions)

    if runLIPC:
        ragStrategy = 1

        li.process(
            dry_run,
            ragStrategy,
            start_partition,
            end_partition,
            content_dir,
            partitions,
        )
