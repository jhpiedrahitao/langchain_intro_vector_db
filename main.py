import os
from lib2to3.pgen2.token import OP
from unittest import result
from xml.dom.minidom import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone 
from langchain import VectorDBQA, OpenAI
import pinecone

pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENVIROMENT"))

if __name__=='__main__':
    print("hello vectorstore")
    loader= TextLoader("mediumblogs/mediumblog1.txt")
    document= loader.load()

    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts= text_splitter.split_documents(document)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"), show_progress_bar=True)
    #docsearch = Pinecone.from_documents(texts, embeddings, index_name="medium-blogs-embbeding-index")
    docsearch = Pinecone.from_existing_index(embedding=embeddings,index_name="medium-blogs-embbeding-index")
    qa= VectorDBQA.from_chain_type(llm=OpenAI(),chain_type="stuff", vectorstore=docsearch, return_source_documents=True)
    query="que es un vectordatabase dame una respuesta sencilla en menos de 50 palabras para un novato en el tema"
    result=qa({"query":query})
    print(result)