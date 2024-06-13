import os, time

from groundx import Groundx
from openai import OpenAI
import pandas as pd

from util import move_columns


gx = None
oai = None
system_prompt = "you are a helpful AI agent tasked with answering questions with the content provided to you"


def init():
    global gx, oai

    gx = Groundx(
        api_key=os.getenv("GROUNDX_API_KEY"),
    )

    oai = OpenAI()


def rag(query, model_name, indexes):
    global gx, oai, system_prompt

    res = []

    print(f"\n\n{query}\n")

    for i, pid in enumerate(indexes):
        print(f"querying partition {i}, {pid}")

        result = None
        retrievals = None
        source = None
        sourceCount = None

        tasktime = time.time()
        for _ in range(3):
            reqtime = time.time()
            try:
                retrievals = gx.search.content(id=pid, query=query).body
                break
            except Exception as e:
                print(f"gx error, trying again [{time.time() - reqtime:.4f}]")
                print(e)
                print(type(e))
                time.sleep(3)
        print(f"gx done [{time.time() - tasktime:.4f}]")

        if "search" not in retrievals or "text" not in retrievals["search"]:
            print("empty search result")
            print(retrievals)

            source = ""
            sourceCount = 0

            raise Exception("empty GX search result")

        source = str(retrievals["search"]["text"])
        sourceCount = len(retrievals["search"]["results"])

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"content:\n{source}\n\nanswer the following question using the content above: {query}",
            },
        ]

        tasktime = time.time()
        for _ in range(3):
            reqtime = time.time()
            try:
                result = (
                    oai.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=1.0,
                        top_p=0.7,
                    )
                    .choices[0]
                    .message.content
                )
                if result is not None:
                    print(str(result))
                    break
                else:
                    print(
                        f"openAI result is none, trying again [{time.time() - reqtime:.4f}]"
                    )
                    time.sleep(3)
            except Exception as e:
                print("error, trying again")
                print(f"error, trying again [{time.time() - reqtime:.4f}]")
                print(e)
                print(type(e))
                time.sleep(3)
                raise e
        print(f"openAI done [{time.time() - tasktime:.4f}]")

        # composing response fields
        d = {}
        d["approach"] = "RAG"
        d["response"] = str(result)
        d["source"] = source
        d["retrieval_count"] = sourceCount
        d["partition_name"] = f"partition_{i}"
        d["partition_id"] = pid

        res.append(d)

    return res


def run(model_name, indexes, questions):
    init()

    completed = []
    for i, row in questions.iterrows():
        row = dict(row)
        res = rag(row["query"], model_name, indexes)

        for res_inst in res:
            this_row = {**row, **res_inst}
            completed.append(this_row)

    df_completed = pd.DataFrame(completed)

    return move_columns(df_completed)
