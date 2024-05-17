import multiprocessing, os

from dotenv import load_dotenv
import pandas as pd


load_dotenv()


if (
    os.getenv("OPENAI_API_KEY") is None
    or os.getenv("PINECONE_API_KEY") is None
    or os.getenv("PINECONE_API_KEY_LI") is None
    or os.getenv("GROUNDX_API_KEY") is None
):
    raise Exception(
        """


    You have not set a required environment variable (OPENAI_KEY, PINECONE_API_KEY, or PINECONE_HOST)
    Copy .env.sample and rename it to .env then fill in the missing values

"""
    )


runGroundX = False
runLCPC = False
runLI = True

model_name = 'gpt-4-turbo-2024-04-09'
results_path = 'results'

questions = pd.read_csv("tests/RAGBattle3/tests.csv")
questions.head()


def doLCPC():
    global model_name, questions, results_path

    from lcpc.rag import run as lcpcrun

    lcpc_experiment = "13_2_LCPC_naive"

    lcpcidx = [
        "rb3-lcpc-partition0",
        "rb3-lcpc-partition1",
        "rb3-lcpc-partition2",
        "rb3-lcpc-partition3",
    ]

    results = lcpcrun(model_name, lcpcidx, questions)
    results.to_csv(f"{results_path}/{lcpc_experiment}_{model_name}.csv", index=False)


def doLI():
    global model_name, questions, results_path

    from li.rag import run as lipcrun

    li_experiment = "13_3_LI_naive_naive"

    lipcidx = [
        "rb3-li-naive-partition0",
        "rb3-li-naive-partition1",
        "rb3-li-naive-partition2",
        "rb3-li-naive-partition3",
    ]

    results = lipcrun(model_name, lipcidx, questions)
    results.to_csv(f"{results_path}/{li_experiment}_{model_name}.csv", index=False)


def doGroundX():
    global model_name, questions, results_path

    from gx.rag import run as gxrun

    gx_experiment = "13_1_GroundX_naive"

    gxidx = [
        7802,
        7980,
        7981,
        7982,
    ]

    results = gxrun(model_name, gxidx, questions)
    results.to_csv(f"{results_path}/{gx_experiment}_{model_name}.csv", index=False)


if __name__ == "__main__":
    functions = []

    if runGroundX:
        functions.append(doGroundX)
    if runLCPC:
        functions.append(doLCPC)
    if runLI:
        functions.append(doLI)

    if len(functions) > 1:
        processes = []

        for fn in functions:
            p = multiprocessing.Process(target=fn)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    elif len(functions) == 1:
        functions[0]()