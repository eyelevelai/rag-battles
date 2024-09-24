# rag-battles

## RAG Battle 3

### Pre-Requisites

1. Create 2 new projects in PineCone: 1 for LangChain and 1 for LlamaIndex
2. In each project, create 4 indexes for the 4 partitions:
    - For LangChain, name them **rb3-lcpc-partition0**, **rb3-lcpc-partition1**, **rb3-lcpc-partition2**, and **rb3-lcpc-partition3**
    - For LlamaIndex, name them **rb3-li-naive-partition0**, **rb3-li-naive-partition1**, **rb3-li-naive-partition2**, and **rb3-li-naive-partition3**
3. Create an API key for each project and make note of them
4. Create 1 new project in GroundX
5. Create 4 buckets in the project for **partition0**, **partition1**, **partition2**, and **partition3**
6. Make note of the bucket IDs and project ID
    - The bucket IDs will be needed for uploading files
    - The project IDs will be needed for performing RAG queries

### Uploading Test Files

1. Run `pip install -r requirements.txt` to install dependencies
2. Copy **.env.sample** to **.env** and replace placeholder values with your own
```
GROUNDX_API_KEY=your GroundX API key
OPENAI_API_KEY=your OpenAI API key
PINECONE_API_KEY=an API key for a PineCone instance, used with LangChain
PINECONE_API_KEY_LI=an API key for a PineCone instance, used with LlamaIndex
TESSDATA_PREFIX=the location of your tessdata directory, needed for LangChain/Unstructured to work, ask ChatGPT for help finding this or consult LangChain documentation
```
3. Create a folder called **Partitions** in the **rag-battle-3** directory
4. Save **partition0.zip**, **partition1.zip**, **partition2.zip**, and **partition3.zip** in **rag-battle-3/Partitions**
3. Open **rag-battle-3/upload.py** and change the run flag for the corresponding RAG system from False to True
   - For GroundX, set `runGroundX = True`
   - For LangChain, set `runLCPC = True`
   - For LlamaIndex, set `runLIPC = True`
4. Change directory to **rag-battle-3** and run `python upload.py`

### Doing RAG

1. Change the **gxidx** array values to the corresponding bucket IDs for the partitions
2. Open **rag-battle-3/rag.py** and change the run flag for the corresponding RAG system from False to True
   - For GroundX, set `runGroundX = True`
   - For LangChain, set `runLCPC = True`
   - For LlamaIndex, set `runLIPC = True`
3. Change directory to **rag-battle-3** and run `python rag.py`