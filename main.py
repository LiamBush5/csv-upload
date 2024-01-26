import streamlit as st
import pandas as pd
import os
from langchain.docstore.document import Document
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from dotenv import load_dotenv

# Function to read and process CSV file


def read_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df = df.fillna(' ')  # Replace NaNs with an empty space
    return df

# Function to convert DataFrame rows into Document objects


def text_to_docs(df: pd.DataFrame) -> list:
    docs = []
    for _, row in df.iterrows():
        unique_id = str(row['Unique ID'])
        content = str(row['Content'])
        text = "Unique ID: " + unique_id + " Content: " + content
        metadata = {column: row[column]
                    for column in df.columns if column != 'Content'}
        doc = Document(page_content=text, metadata=metadata)
        docs.append(doc)
    return docs


# Initialize Streamlit app
st.title('CSV to Pinecone Uploader')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
if uploaded_file is not None:
    df = read_csv(uploaded_file)
    if st.button('Upload to Pinecone'):
        # Load environment variables
        load_dotenv()
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pinecone_environment = os.getenv(
            'PINECONE_ENVIRONMENT', 'us-west1-gcp')
        index_name = os.getenv('PINECONE_INDEX_NAME', 'your_index_name')

        # Initialize Pinecone
        pinecone.init(api_key=pinecone_api_key,
                      environment=pinecone_environment)
        # Check if index exists, if not create one
        if index_name not in pinecone.list_indexes():
            # Adjust dimension based on your embeddings
            pinecone.create_index(index_name, dimension=768)
        index = pinecone.Index(index_name)

        # Process DataFrame to documents
        document_chunks = text_to_docs(df)

        # Embeddings and uploading
        embeddings = OpenAIEmbeddings()  # Make sure to adjust according to your setup
        vectors = [(doc.metadata['Unique ID'], embeddings.embed(
            [doc.page_content])[0]) for doc in document_chunks]

        # Upsert documents to Pinecone
        index.upsert(vectors=vectors)

        st.success('Documents uploaded successfully to Pinecone!')
