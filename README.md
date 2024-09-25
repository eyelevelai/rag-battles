# rag-battles
EyeLevel.ai conducts regular tests which explore the performance of various RAG solutions. We call these "RAG Battles" as they often feature direct comparisons between RAG approaches. Within this repo you will find several examples of these "RAG Battles", which compare the relative performance of common RAG approaches.

## ⭐ RAG Battle 1
An early internal smoke test not available for public review, listed here because starting at 2 seems silly.

## ⭐ RAG Battle 2
This test features a toe-to-toe comparison of three "out of the box" RAG solutions:
- LlamaIndex
- LangChain with PineCone
- GroundX
While both LangChain and LlamaIndex support various RAG approaches, we wanted to see how well the default approaches compared. If you're interested in advanced RAG approaches, we have an upcoming RAG battle which tests the practical impact of common advanced RAG approaches.

The code for RB2 can be found [here](https://drive.google.com/drive/u/0/folders/1l45ljrGfOKsiNFh8QPji2eBAd2hOB51c), and a corresponding article describing results can be found [here](https://www.eyelevel.ai/post/most-accurate-rag). This was the percentage of correctly answered questions for each RAG approach, when given the same documents and the same questions:
GroundX: 97.83%
LangChain / Pinecone: 64.13%
LlamaIndex: 44.57%

## ⭐ RAG Battle 3
This RAG battle features the same approaches as RAG Battle 2, but explores if LlamaIndex, LangChain, and GroundX experience a degredation in performance when exposed to more documents. If a RAG system has to answer the same question in small and large store of doccuments, in theory the larger store of doccuments would be harder to search through as there would be more oportunities for relevent and irrelevent doccuments to overlap within the search space. In RAG Battle 3 we saw this theory play out when RAG approaches were applied to real world documents.

The same questions and documents which were featured in RAG Battle 2 were also featured in RAG Battle 3, with additional documents added which were irrelevent to the questions being asked. We created sets of 1,000, 10,000, 50,000, and 100,000 pages which were queried against, and saw that GroundX degrades by 2%/100,000 pages, LangChain degrades by 10%/100,000 pages, and LlamaIndex degrades by 12%/100,000 pages.

The code for this test can be found in the [RAG Battle 3 folder in this repo](https://github.com/eyelevelai/rag-battles/tree/main/rag-battle-3/tests/RAGBattle3), the article describing the test can be found [here](https://www.eyelevel.ai/post/do-vector-databases-lose-accuracy-at-scale), and the steps to re-create the test yourself can be found below.

### RB3: Pre-Requisites
The following is required to setup various document stores for conducting RAG Battle 3.

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

### RB3: Uploading Test Files
The following steps are required to upload documents to each system, creating the document stores with different numbers of doccuments which will be queried against to test performance at scale.

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

### RB3: Doing RAG
The following steps are required to run RAG Battle 3.

1. Change the **gxidx** array values to the corresponding bucket IDs for the partitions
2. Open **rag-battle-3/rag.py** and change the run flag for the corresponding RAG system from False to True
   - For GroundX, set `runGroundX = True`
   - For LangChain, set `runLCPC = True`
   - For LlamaIndex, set `runLIPC = True`
3. Change directory to **rag-battle-3** and run `python rag.py`
