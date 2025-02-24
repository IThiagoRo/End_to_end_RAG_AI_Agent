# Search Cancer Investigations Agent RAG AI 
This project aims to develop an AI-powered RAG agent that helps oncologists and healthcare experts stay up to date with new scientific publications on cancer.

To achieve this, the agent has been built using AWS cloud services and automatically downloads articles published on arXiv related to cancer. It then analyzes these documents to extract relevant information, making it easier to access the latest advancements in cancer research.

## Data:
The information is obtained from arxiv api 
(link: https://arxiv.org/)

## Architecture Diagram:
![Architecture](https://github.com/user-attachments/assets/03f9e34f-8c4a-405b-ae79-57ad987b50d9)

## Models Used:
```bash
Amazon Titan Embedding G1 - Text
Meta Llama 3
```

## Requirements:
python3, pip3 

## Installation:

1.Clone the project
```bash
git clone https://github.com/IThiagoRo/End_to_end_RAG_AI_Agent.git
```
2.Enter to the project dir
```bash
cd End_to_end_RAG_AI_Agent
```

3.Create a virtual environment 
```bash
python3 -m venv venv
```
4.Enter to the virtual environment
```bash
. venv/bin/activate
```
5.With pip3 install
```bash
pip3 install setuptools
```
6.Execute the command
```bash
python3 setup.py install
```
7.Configure your AWS CLI credentials 

(Read: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html)

8.Run the api server
 
```bash
python3 src/main.py 
```

## API Reference

The api has two endpoints: 

- Process
- Query

(API Link: ec2-18-207-208-61.compute-1.amazonaws.com:8000/docs)

### 1. Process arXiv Articles

**URL:** `/process/`

**Method:** `GET`

**Parameters:**
- `query` (str, optional, default "Cancer"): Query to search for articles on arXiv.

**Response:**
```json
{
  "message": "Processing complete",
  "arxiv_ids": ["2203.00502v1"]
}
```

### 2. Query the RAG System

**URL:** `/query/`

**Method:** `POST`

**Request Body:**
```json
{
  "question": "Tell me about breast cancer"
}
```

**Response:**
```json
{
  "question": "Tell me about breast cancer",
  "answer": "Breast cancer is a serious health threat to women worldwide. The breast cancer dataset..."
}
```
