import langchain
from langchain_openai import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.memory import ChatMessageHistory
langchain.verbose = True


def create_index() -> VectorStoreIndexWrapper:
    loader = DirectoryLoader("./src/", glob="**/*.py")
    return VectorstoreIndexCreator().from_loaders([loader])


def chat(
    message: str, history: ChatMessageHistory, index: VectorStoreIndexWrapper
) -> str:
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    return index.query(question=message, llm=llm)