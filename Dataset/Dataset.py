import torch
from torch.utils.data import Dataset
from glob import glob
import json
import numpy as np
from sentence_transformers import SentenceTransformer


class dataset_canlii(Dataset):
    
    def __init__(self, data_root, seq_length, min_length, gap, embedding_model):
        super().__init__()
        
        self.path_all_files = glob(data_root + '/*.json')
        
        self.seq_length = seq_length
        self.min_length = min_length
        self.gap = gap
        self.embedding_model = embedding_model
        
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
        
        with open(self.path_all_files[file_index], 'r') as f:
            text = json.load(f)['content']
        
        words = text.split()
        
        start = self.gap * batch_index
        end = min(start + self.seq_length, self.all_word_intervals[file_index+1] - self.all_word_intervals[file_index] - start)
        input_words = (' ').join(words[start : end])
        
        # if end - start < self.seq_length:
        #     fill_the_rest_of_input_words
        
        return input_words
    
    
    def embed_chunks(self, chunks):
        model = SentenceTransformer(self.embedding_model)
        embeddings = model.encode(chunks, show_progress_bar=True)
        return embeddings


    def __get_item__(self, index):
        
        file_index, index_data_batch = self.decode_index(index)
        input_words = self.create_input_text(file_index, index_data_batch)
        
        input_embeddings = self.embed_chunks(input_words)
        
        return input_embeddings
        
        
    def __len__(self):
        
        return self.batch_word_intervals[-1]
    

if __name__=="__main__":
    
    embedding_model = "all-MiniLM-L6-v2"
    data_root = r'C:\Users\javad\Projects\Personal Projects\LawLanguageModel\canlii_cases'
    
    dataset = dataset_canlii(data_root=data_root, seq_length=1000, min_length=300, gap=200, embedding_model=embedding_model)
    
    words = dataset.__get_item__(208)
    
    a = 1
        