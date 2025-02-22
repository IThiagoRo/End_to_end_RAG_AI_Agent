import json
import os
import sys
import boto3

## We will be using Titan Embeddings Model To generate Embeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

# Vector Embedding And Vector Store
from langchain.vectorstores import FAISS

# LLm Models
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Bedrock Clients
session = boto3.Session(profile_name="personal")
s3 = session.client(service_name="s3")
bedrock = session.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)

folder_path = 'data_vectors/'  

def load_index(BUCKET_NAME, folder_path):
    """
    Downloads all files from an S3 bucket within a specified folder path.
    Args:
        BUCKET_NAME (str): The name of the S3 bucket.
        folder_path (str): The local folder path where the files will be stored.

    Returns:
        None
    """
    os.makedirs(folder_path, exist_ok=True)  

    # List files in the bucket within the specified folder_path
    response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=folder_path)

    if 'Contents' in response:
        for obj in response['Contents']:
            key = obj['Key']
            filename = os.path.join(folder_path, os.path.basename(key))  # Save only the filename
            s3.download_file(Bucket=BUCKET_NAME, Key=key, Filename=filename)
            print(f"Downloaded: {filename}")
    else:
        print(f"No files found in {folder_path} within bucket {BUCKET_NAME}")

def get_faiss_index_name(folder_path):
    """
    Searches for a FAISS index file in the given folder and returns its name without extension.
    
    Args:
        folder_path (str): The local folder path to search for FAISS index files.

    Returns:
        str or None: The name of the FAISS index file without extension if found, otherwise None.
    """
    for file in os.listdir(folder_path):
        if file.endswith(".faiss"):
            return os.path.splitext(file)[0]  # Return the filename without extension
    return None  # Return None if no .faiss file is found

def get_llama3_llm():
    """
    Initializes and returns a Llama 3 model using Bedrock API.
    
    Returns:
        Bedrock: An instance of the Llama 3 language model.
    """
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock,
                  model_kwargs={'max_gen_len': 512})
    return llm

def get_response(llm, vectorstore, query):
    """
    Generates a response using a RetrievalQA chain based on a vector store and a query.
    
    Args:
        llm (Bedrock): The LLM model used for generating responses.
        vectorstore (FAISS): The FAISS vector store for retrieving relevant documents.
        query (str): The input question for which an answer is needed.

    Returns:
        str: The generated response from the RetrievalQA system.
    """
    prompt_template = """
    Human: Use the following pieces of context to provide a concise answer to the question at
    the end but use at summarize with 100 words with detailed explanations. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    answer = qa({"query": query})  
    return answer['result']

