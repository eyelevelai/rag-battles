import os, time

from groundx import Groundx
from openai import OpenAI
import pandas as pd

from util import move_columns


gx = None
oai = None
system_prompt = 'you are a helpful AI agent tasked with answering questions with the content provided to you'


def init():
    global gx, oai

    gx = Groundx(
        api_key=os.getenv("GROUNDX_API_KEY"),
    )

    oai = OpenAI()


def rag(query, model_name, indexes):
    global gx, oai, system_prompt

    res = []

    print(f'\n\n{query}\n')

    for i, pid in enumerate(indexes):
        print(f'querying partition {i}, {pid}')

        result = None
        retrievals = None
        source = None
        sourceCount = None

        for _ in range(3):
            try:        
                retrievals = gx.search.content(
                    id=pid,
                    query=query
                ).body

                if 'search' not in retrievals or 'text' not in retrievals['search']:
                    print('empty search result')
                    print(retrievals)

                    source = ""
                    sourceCount = 0
                else:
                    source = str(retrievals['search']['text'])
                    sourceCount = len(retrievals['search']['results'])

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

                    result = oai.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=1.0,
                        top_p=0.7,
                    ).choices[0].message.content

                break
            except Exception as e:
                print('error, trying again')
                print(e)
                print(retrievals['search'])
                time.sleep(5)

        #composing response fields
        d = {}
        d['approach'] = "RAG"
        d['response'] = str(result)
        d['source'] = source
        d['retrieval_count'] = sourceCount
        d['partition_name'] = f'partition_{i}'
        d['partition_id'] = pid

        res.append(d)

    return res


def run(model_name, indexes, questions):
    init()

    completed = []
    for i, row in questions.iterrows():
        row = dict(row)
        res = rag(row['query'], model_name, indexes)

        for res_inst in res:
            this_row = {**row, **res_inst}
            completed.append(this_row)

    df_completed = pd.DataFrame(completed)

    return move_columns(df_completed)