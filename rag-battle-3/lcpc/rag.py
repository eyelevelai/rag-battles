import os, time

from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Pinecone
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
import pandas as pd
from tqdm import tqdm


chains = []
embeddings = None
retrievers = []


def init(model_name, index_names):
    global chains, embeddings, retrievers

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


def move_columns(df):
    column_to_move = df.pop("query")
    df.insert(0, "query", column_to_move)

    column_to_move = df.pop("expected_response")
    df.insert(1, "expected_response", column_to_move)

    column_to_move = df.pop("response")
    df.insert(2, "response", column_to_move)

    column_to_move = df.pop("approach")
    df.insert(3, "approach", column_to_move)

    column_to_move = df.pop("partition_id")
    df.insert(4, "partition_id", column_to_move)

    column_to_move = df.pop("partition_name")
    df.insert(5, "partition_name", column_to_move)

    column_to_move = df.pop("problem_type")
    df.insert(6, "problem_type", column_to_move)

    column_to_move = df.pop("context_modality")
    df.insert(7, "context_modality", column_to_move)

    column_to_move = df.pop("retrieval_count")
    df.insert(8, "retrieval_count", column_to_move)

    column_to_move = df.pop("source")
    df.insert(8, "source", column_to_move)

    df.pop("context_file")

    check = ["query", "expected_response", "response", "approach", "partition_id", "partition_name", "problem_type", "context_modality", "retrieval_count", "source"]
    if set(df.columns) != set(check):
        print(df.columns)
        raise ValueError('incompatible columns')

    return df


def rag(query, index_names):
    res = []

    print(f'RAG actoss all of {index_names}')

    for i, index_name in tqdm(enumerate(index_names)):
        print(f'querying partition {i}, {index_name}')

        result = None
        retrieval = None

        for _ in range(3):
            try:
                cres = chains[i](query)
                result = cres['result']
                retrieval = cres['source_documents']
                break
            except:
                print('error running chain, retrying')
                time.sleep(5)
        else:
            raise ValueError('persistent issue running chain')

        d = {}
        d['approach'] = "RAG"
        d['response'] = str(result)
        d['source'] = str(retrieval)
        d['retrieval_count'] = len(retrieval)
        d['partition_name'] = f'partition_{i}'
        d['partition_id'] = index_name

        res.append(d)

    return res


def run(model_name, index_names, questions):
    init(model_name, index_names)

    completed = []
    for _, row in tqdm(questions.iterrows()):
        row = dict(row)
        res = rag(row['query'], index_names)

        for res_inst in res:
            this_row = {**row, **res_inst}
            completed.append(this_row)

    df_completed = pd.DataFrame(completed)
    return move_columns(df_completed)
