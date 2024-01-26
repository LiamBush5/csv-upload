
import pandas as pd
import os
from typing import Callable, List, Dict
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import pinecone
import os
import openai
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import streamlit as st


def read_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = df.fillna(' ')  # Replace NaNs with an empty space
    return df


def text_to_docs(df: pd.DataFrame) -> List[Document]:
    docs = []
    total_rows = len(df)
    for index, row in df.iterrows():
        unique_id = str(row['Unique ID'])
        content = str(row['Content'])
        text = "Unique ID: " + unique_id + " Content: " + content

        metadata = {column: row[column]
                    for column in df.columns if column != 'Content'}
        doc = Document(page_content=text, metadata=metadata)
        docs.append(doc)

        # Print progress
        print(
            f"Processing document {index + 1}/{total_rows} with Unique ID: {unique_id}")

    return docs


def get_similiar_docs(query, k=2, score=False):
    if score:
        similar_docs = docsearch.similarity_search_with_score(query, k=k)
    else:
        similar_docs = docsearch.similarity_search(query, k=k)
    return similar_docs


if __name__ == "__main__":
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file is not None:
        df = read_csv(uploaded_file)
        if st.button('Upload to Pinecone'):
            load_dotenv()
            file_path = "src/data/data/rclup.csv"
            df = read_csv(file_path)
            document_chunks = text_to_docs(df)

            pinecone.init(api_key='8843c877-4c11-45c7-abd4-4403b88aa5c5',
                          environment='asia-southeast1-gcp-free')
            index_name = 'rcl'

            embeddings = OpenAIEmbeddings()

            document_chunks = text_to_docs(df)

            docsearch = Pinecone.from_documents(
                document_chunks, embeddings, index_name=index_name)
            print("finished")
            st.success('Documents uploaded successfully to Pinecone!')
