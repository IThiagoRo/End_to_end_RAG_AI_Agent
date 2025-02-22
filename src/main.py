import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from components.data_ingestion import search_arxiv, download_arxiv_pdf, PyPDFLoader, split_text, create_vector_store
from components.rag_ai_agent import load_index, get_faiss_index_name, FAISS, bedrock_embeddings, get_llama3_llm, get_response

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.get("/process/")
def process(query: str = "Cancer"):
    arxiv_ids = search_arxiv(query)
    if not arxiv_ids:
        return {"message": "No articles found"}
    
    for arxiv_id in arxiv_ids:
        file_name = download_arxiv_pdf(arxiv_id)
        if not file_name:
            return {"message": "Download failed"}
        
        loader = PyPDFLoader(file_name)
        pages = loader.load_and_split()
        splitted_docs = split_text(pages, 1000, 200)
        create_vector_store(file_name[5:], splitted_docs)
    
    return {"message": "Processing complete", "arxiv_ids": arxiv_ids}

@app.post("/query/")
def query_rag(request: QueryRequest):
    load_index("datalakecancer", "data_vectors/")
    index_name = get_faiss_index_name("data_vectors/")
    faiss_index = FAISS.load_local(
        index_name=index_name,
        folder_path="data_vectors/",
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )
    llm = get_llama3_llm()
    answer = get_response(llm, faiss_index, request.question)
    return {"question": request.question, "answer": answer}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
