import os
import boto3
import requests
import feedparser

# Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Pdf Loader
from langchain_community.document_loaders import PyPDFLoader
# Embeddings
from langchain_community.embeddings import BedrockEmbeddings
# import FAISS
from langchain_community.vectorstores import FAISS

# Session AWS
session = boto3.Session(profile_name="personal")
s3 = session.client(service_name="s3")
bedrock=session.client(service_name="bedrock-runtime")
bedrock_embeddings=BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",client=bedrock)


def search_arxiv(query, max_results=1):
    """
    Searches for articles on arXiv using a given query.

    :param query: str, search query (e.g., 'Cancer')
    :param max_results: int, maximum number of results
    :return: list of arXiv identifiers
    """
    base_url = "http://export.arxiv.org/api/query?"
    search_url = f"{base_url}search_query={query.replace(' ', '+')}&start=0&max_results={max_results}"
    feed = feedparser.parse(search_url)
    arxiv_ids = []
    for entry in feed.entries:
        arxiv_id = entry.id.split('/abs/')[-1]
        arxiv_ids.append(arxiv_id)
    return arxiv_ids

def download_arxiv_pdf(arxiv_id):
    """
    Downloads a PDF file from arXiv and uploads it to an S3 bucket.
    :param arxiv_id: str, arXiv article identifier
    """
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        os.makedirs('data/', exist_ok=True)  
        file_name = f"data/arxiv_paper_{arxiv_id}.pdf"
        with open(file_name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Download completed: {file_name}")
        upload_to_s3(file_name)
        return file_name
    else:
        print(f"Error downloading the file. Status code: {response.status_code}")
        return None

def upload_to_s3(file_name, bucket_name="datalakecancer", s3_key=None):
    """
    :param file_name: str, local file name
    :param bucket_name: str, S3 bucket name
    :param s3_key: str, S3 key 
    """
    if s3_key is None:
        s3_key = file_name

    s3.upload_file(file_name, bucket_name, s3_key)
    print(f"File uploaded to S3: s3://{bucket_name}/{s3_key}")

def split_text(pages, chunk_size, chunk_overlap):
    # Split the pages, text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def create_vector_store(file_name, documents, bucket_name="datalakecancer"):
    """
    Creates and saves a FAISS vector store from the documents and uploads it to an S3 bucket.

    :param documents: list, processed documents in chunks
    :param bucket_name: str, name of the S3 bucket
    """
    # Create the FAISS vector store
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embeddings)

    folder_path = "data_vectors"  # Local folder 
    os.makedirs(folder_path, exist_ok=True)

    faiss_path = f"{folder_path}/{file_name}.faiss"
    pkl_path = f"{folder_path}/{file_name}.pkl"

    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)
    print(f"Vector store saved locally in {folder_path}")

    # Upload to S3
    upload_to_s3(faiss_path, bucket_name, f"data_vectors/{file_name}.faiss")
    upload_to_s3(pkl_path, bucket_name, f"data_vectors/{file_name}.pkl")
    print(f"Vector store uploaded to S3 at s3://{bucket_name}/data_vectors/")

#if __name__ == '__main__':
#    query = "Cancer"
#    arxiv_ids = search_arxiv(query)
#
#    for arxiv_id in arxiv_ids:
#        file_name = download_arxiv_pdf(arxiv_id)
#        loader = PyPDFLoader(file_name)
#        pages = loader.load_and_split()
#
#        splitted_docs = split_text(pages, 1000, 200)
#        create_vector_store(file_name[5:], splitted_docs)
