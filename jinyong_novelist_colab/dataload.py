# some imports
import multiprocessing
import datetime as dt
import torch
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
import numpy as np
from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")

def word_to_idx(lst):
    vocabulary_length = 0
    Dict = {}
    for word in lst:
        if word not in Dict.keys():
            Dict[word] = vocabulary_length
            vocabulary_length += 1
            
    vocabulary_length += 1
    return Dict, vocabulary_length


class MyDataset(Dataset):
    def __init__(self, data_path, input_word_count):
        file = open(data_path, 'r', encoding='utf-8')
        text = file.read()
        file.close()

        self.wv, self.vocabulary_length = word_to_idx(text)
        
        self.input_word_count = input_word_count

        self.data = []

        for i in text:
            self.data.append(self.wv[i])
            
        self.data = np.array(self.data)

        
    def __len__(self):
        return len(self.data)

        
    def __getitem__(self, idx):
        if idx <= len(self.data)-self.input_word_count:
            return self.data[idx:idx+self.input_word_count]
        else:
            return self.__getitem__(idx-(len(self.data)-self.input_word_count))


# testing and generating word model
if __name__ == '__main__':
    my_dataset = MyDataset('texts/test_dataload.txt', 10, save_word_model=True)
    my_dataloader = DataLoader(my_dataset, batch_size=3, shuffle=True)
    print(next(iter(my_dataloader)).shape)



class TokenDataset(Dataset):
    def __init__(self, path, tokenizer, input_word_count):
        file = open(path, 'r', encoding='utf-8')
        text = file.read()
        file.close()

        self.data = []
        self.input_word_count = input_word_count
        self.data.append(101)
        for i in text:
            self.data.append(tokenizer.encode(i)[1])
        self.data.append(102)

        self.data = torch.tensor(self.data) ###

        self.vocabulary_length = len(tokenizer.vocab)


    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        if idx <= len(self.data)-self.input_word_count:
            return self.data[idx:idx+self.input_word_count]
        else:
            return self.__getitem__(idx-(len(self.data)-self.input_word_count))