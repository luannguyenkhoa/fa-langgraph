import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter


DOCUMENTS_PATH = "documents"
VECTOR_STORE_PERSIST_PATH = "vector_data"

class CustomOpenAIEmbeddings(OpenAIEmbeddings):

    def __init__(self, openai_api_key, *args, **kwargs):
        super().__init__(openai_api_key=openai_api_key, *args, **kwargs)
        
    def _embed_documents(self, texts):
        return super().embed_documents(texts)

    def __call__(self, input):
        return self._embed_documents(input) 

def load_chunk_persist_pdf() -> Chroma:
    documents = []
    for file in os.listdir(DOCUMENTS_PATH):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(DOCUMENTS_PATH, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunked_documents = text_splitter.split_documents(documents)

    vector_db = Chroma.from_documents(
        documents=chunked_documents,
        embedding=CustomOpenAIEmbeddings(os.environ["OPENAI_API_KEY"]),
        persist_directory=VECTOR_STORE_PERSIST_PATH
    )
    vector_db.persist()

    return vector_db
