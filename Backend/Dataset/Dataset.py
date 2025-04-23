import torch
from torch.utils.data import Dataset
from glob import glob
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA


class dataset_canlii_embedder(Dataset):
    
    def __init__(self, data_root, seq_length, min_length, gap, embedding_model):
        super().__init__()
        
        self.path_all_files = glob(data_root + '/*.json')
        
        self.seq_length = seq_length
        self.min_length = min_length
        self.gap = gap
        self.embedding_model = embedding_model

        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = SentenceTransformer(self.embedding_model)
        
        self.fetch_data_info()
    
    
    def fetch_data_info(self):
        self.all_word_intervals = [0]
        self.batch_word_intervals = [0]
        for path in self.path_all_files:
            with open(path, 'r', encoding='utf-8') as f:
                file = json.load(f)
            
            text = file['content']
            
            N_words = len(text.split())
            self.all_word_intervals.append(N_words + self.all_word_intervals[-1])
            
            N_inputs = int(np.ceil((N_words - self.min_length) / self.gap))
            self.batch_word_intervals.append(N_inputs + self.batch_word_intervals[-1])
            
            
    def decode_index(self, index):
        def binary_search(seq, number, start=0, end=-1):
            if end==-1:
                end = len(seq) - 1
            
            if end - start < 2:
                return start
            
            mid = (end + start) // 2
            if number == seq[mid]:
                return mid
            elif number < seq[mid]:
                end = mid
            elif number > seq[mid]:
                start = mid
            
            return binary_search(seq, number, start, end)
        
        file_index = binary_search(self.batch_word_intervals, index)
        index_data_batch = index - self.batch_word_intervals[file_index]
        
        return file_index, index_data_batch
    
    def create_input_text(self, file_index, batch_index):
        
        with open(self.path_all_files[file_index], 'r', encoding='utf-8') as f:
            text = json.load(f)['content']
        
        words = text.split()
        
        start = self.gap * batch_index
        end = min(start + self.seq_length, self.all_word_intervals[file_index+1] - self.all_word_intervals[file_index] - start)
        input_words = (' ').join(words[start : end])
        
        # input_words2 = self.pad_to_fixed_length(input_words)
        # if end - start < self.seq_length:
        #     fill_the_rest_of_input_words
        
        return input_words
    
    
    def embed_chunks(self, chunks):
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        return embeddings


    def __getitem__(self, index):
        
        file_index, index_data_batch = self.decode_index(index)
        input_words = self.create_input_text(file_index, index_data_batch)
        
        input_embeddings = self.embed_chunks(input_words.split())
        
        return input_embeddings, input_words
        
        
    def __len__(self):
        
        return self.batch_word_intervals[-1]
    

class embedding_to_faiss():
    
    def __init__(self, model_name):
        
        # Use the same embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
        
        
    def fecth_data_to_vectorstore(self, texts):
        
        # Prepare LangChain documents
        docs = [Document(page_content=text) for text in texts]

        # Build FAISS index
        self.vectorstore = FAISS.from_documents(docs, self.embedding_model)
        
    def save_vectorstore(self, path_save):
        # Save locally
        self.vectorstore.save_local(path_save)
    
        

class Law_RAG():
    
    def __init__(self, model_id, vectorstore):
        
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        self.build_llm(model_id)
        
        self.build_RAG_pipeline()
    
    def build_llm(self, model_id):
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",  # Auto-place on GPU
            torch_dtype=torch.float16,
            load_in_4bit=True  # optional for 4060 to save memory
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.3,
            top_p=0.95,
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)
    
    def build_RAG_pipeline(self,):
        # Build RAG chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True
        )
    
    def fetch_query(self, query):
        
        return self.qa_chain(query)
        
    

if __name__=="__main__":
    
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    llm_name = "mistralai/Mistral-7B-Instruct-v0.1"
    data_root = r"C:\Users\javad\Projects\Personal Projects\NLP-Law-LLM\canlii_cases\texts"
    path_save_vs = r"C:\Users\javad\Projects\Personal Projects\NLP-Law-LLM\canlii_cases\vectorStores"
    
    # query = "What is the Supreme Court's stance on reasonable doubt in criminal cases?"
    query = "what happens in the case of a second degree murder where there is a video available to prove?"
    
    dataset_embedder = dataset_canlii_embedder(data_root=data_root, seq_length=1000, min_length=300, gap=200, embedding_model=embedding_model_name)
    embed_saver = embedding_to_faiss(model_name=embedding_model_name)
    
    words_chunks = []
    for i in range(len(dataset_embedder)):
        input_embeddings, words_chunk = dataset_embedder.__getitem__(i)
        words_chunks.append(words_chunk)
    
    embed_saver.fecth_data_to_vectorstore(words_chunks)
    embed_saver.save_vectorstore(path_save_vs+'/faiss_index')
    
    law_RAG = Law_RAG(model_id=llm_name, vectorstore=embed_saver.vectorstore)
    
    answer = law_RAG.fetch_query(query=query)
    
    a = 1
        