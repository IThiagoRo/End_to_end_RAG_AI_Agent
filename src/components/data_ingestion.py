import requests
import feedparser
import boto3

def search_arxiv(query, max_results=5):
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
        file_name = f"data/arxiv_paper_{arxiv_id}.pdf"
        with open(file_name, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Download completed: {file_name}")
        upload_to_s3(file_name)
    else:
        print(f"Error downloading the file. Status code: {response.status_code}")

def upload_to_s3(file_name, bucket_name="datalakecancer", s3_key=None):
    """
    :param file_name: str, local file name
    :param bucket_name: str, S3 bucket name
    :param s3_key: str, S3 key 
    """
    session = boto3.Session(profile_name="personal")

    s3 = session.client("s3")
    if s3_key is None:
        s3_key = file_name

    s3.upload_file(file_name, bucket_name, s3_key)
    print(f"File uploaded to S3: s3://{bucket_name}/{s3_key}")


if __name__ == '__main__':
    query = "Cancer"
    arxiv_ids = search_arxiv(query)

    for arxiv_id in arxiv_ids:
        download_arxiv_pdf(arxiv_id)

