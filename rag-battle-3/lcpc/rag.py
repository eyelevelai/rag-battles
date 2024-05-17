import os, time

from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import pandas as pd

from util import move_columns


chains = []


def init(model_name, index_names):
    global chains

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    retrievers = [
        Pinecone.from_existing_index(
            index_name,
            embeddings,
        ).as_retriever(
            include_metadata=True,
            metadata_key = 'source',
        ) for index_name in index_names
    ]
    chains = [make_chain(model_name, retriever) for retriever in retrievers]


def make_chain(model_name, retriever):
    llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain


def rag(query, index_names):
    res = []

    print(f'RAG actoss all of {index_names}')

    for i, index_name in enumerate(index_names):
        print(f'querying partition {i}, {index_name}')

        result = None
        retrievals = None

        for _ in range(3):
            try:
                cres = chains[i](query)
                result = cres['result']
                retrievals = cres['source_documents']
                break
            except:
                print('error running chain, retrying')
                time.sleep(5)
        else:
            raise ValueError('persistent issue running chain')

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
    for _, row in questions.iterrows():
        row = dict(row)
        res = rag(row['query'], index_names)

        for res_inst in res:
            this_row = {**row, **res_inst}
            completed.append(this_row)

    df_completed = pd.DataFrame(completed)
    return move_columns(df_completed)
