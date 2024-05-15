import os

from dotenv import load_dotenv

load_dotenv()


if (
    os.getenv("OPENAI_API_KEY") is None
    or os.getenv("PINECONE_API_KEY") is None
    or os.getenv("PINECONE_API_KEY_LI") is None
):
    raise Exception(
        """


    You have not set a required environment variable (OPENAI_KEY, PINECONE_API_KEY, or PINECONE_HOST)
    Copy .env.sample and rename it to .env then fill in the missing values

"""
    )


import pandas as pd

from lcpc.rag import run as lcpcrun


runLCPC = True

question_file = "tests/RAGBattle3/tests.csv"
results_path = f'results'

model_name = 'gpt-4-turbo-2024-04-09'
tabulated_approach_name = 'LCPC_RAG'
partition_blame = 'Ben'

lcpc_experiment = "13_2_LCPC_naive"
li_experiment = "13_2_LI_naive"


lcpcidx = [
    "rb3-lcpc-partition0",
    "rb3-lcpc-partition1",
    "rb3-lcpc-partition2",
    "rb3-lcpc-partition3",
]
lipcidx = [
    "rb3-li-naive-partition0",
    "rb3-li-naive-partition1",
    "rb3-li-naive-partition2",
    "rb3-li-naive-partition3",
]


questions = pd.read_csv(question_file)
questions.head()

if runLCPC:
    results = lcpcrun(model_name, lcpcidx, questions)
    results.to_csv(f"{results_path}/{lcpc_experiment}_{model_name}.tsv", index=False, sep="\t")