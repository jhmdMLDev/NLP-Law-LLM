import os
import sys
sys.path.append(os.getcwd())
from Backend.Dataset.Dataset import Law_RAG

from fastapi import FastAPI, requests
from pydantic import BaseModel


app = FastAPI()


class TextInput(BaseModel):
    textRequest: str


class LLM_query_service():
    
    def __init__(self):
        
        embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        llm_name = "mistralai/Mistral-7B-Instruct-v0.1"
            
        path_save_vs = r"C:\Users\javad\Projects\Personal Projects\NLP-Law-LLM\canlii_cases\vectorStores"
        path_save_vsFile = path_save_vs+'/faiss_index'

        self.law_RAG = Law_RAG(model_id=llm_name, vectorstore=None, path_vectorestore=path_save_vsFile, model_name_embedding=embedding_model_name)


    @app.post("/count_words")
    def get_inference_for_query(self, data):
        
        answer = self.law_RAG.fetch_qury(data.textRequest)
        
        return answer
    


# query = "What is the Supreme Court's stance on reasonable doubt in criminal cases?"
# query = "what happens in the case of a second degree murder where there is a video available to prove?"
# answer = law_RAG.fetch_query(query=query)

# print(answer)