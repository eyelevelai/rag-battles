import os, time

from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.pinecone import PineconeVectorStore
import pandas as pd
from pinecone import Pinecone

from util import move_columns


query_engines = []


def init(model_name, index_names):
    global query_engines

    llm = OpenAI(temperature=0.0, model=model_name)

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY_LI"])

    vector_stores = [PineconeVectorStore(pinecone_index=pc.Index(index_name)) for index_name in index_names]

    indexes = [VectorStoreIndex.from_vector_store(vector_store=vector_store) for vector_store in vector_stores]

    query_engines = [index.as_query_engine(llm=llm) for index in indexes]


def rag(query, index_names):
    res = []

    print(f'RAG actoss all of {index_names}')

    for i, index_name in enumerate(index_names):
        print(f'querying partition {i}, {index_name}')

        result = None
        retrievals = None

        for _ in range(3):
            try:
                cres = query_engines[i].query(query)
                result = str(cres)
                retrievals = cres.source_nodes
                break
            except:
                print('error running chain, retrying')
                time.sleep(5)
        else:
            raise ValueError('persistent issue running chain')

        #composing response fields
        d = {}
        d['approach'] = "RAG"
        d['response'] = str(result)
        d['source'] = str(retrievals)
        d['retrieval_count'] = len(retrievals)
        d['partition_name'] = f'partition_{i}'
        d['partition_id'] = index_name

        res.append(d)

    return res


def run(model_name, index_names, questions):
    init(model_name, index_names)

    completed = []
    for i, row in questions.iterrows():
        row = dict(row)
        res = rag(row['query'], index_names)

        for res_inst in res:
            this_row = {**row, **res_inst}
            completed.append(this_row)

    df_completed = pd.DataFrame(completed)

    return move_columns(df_completed)