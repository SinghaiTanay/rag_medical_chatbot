from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()

# Load PDF files from the specified directory
def load_pdf(data_path):

    loader = DirectoryLoader(
                path = data_path,
                glob='**/*.pdf', 
                loader_cls=PyPDFLoader
            )
    return loader.load()

def chunking(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    return splitter.split_documents(documents)

def embedding_vectorstore(chunks,path):
    embedding_model = OpenAIEmbeddings(
        base_url="https://openrouter.ai/api/v1",
        model="openai/text-embedding-3-small"
    )
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=path,
        collection_metadata={"hsnw:space":"cosine"}
    )

documents = load_pdf('data')
print(f"Length of PDF pages: {len(documents)}")
chunks = chunking(documents)
print(f"Length of Chunks: {len(chunks)}")
embedding_vectorstore(chunks=chunks,path="db/chroma")
print(f"DB created successfully")