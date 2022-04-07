"""Use json files to generate data"""
import torch
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
import numpy as np
import json
from transformers import AutoTokenizer
PRETRAINED = "raynardj/wenyanwen-ancient-translate-to-modern"
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)

wm_paths = "word_model_paths/"
data_path = "dataset/lunyu.json"

zero = torch.zeros(1,100)

class MyDataset_embed(Dataset):
    def __init__(self, wm_paths, data_path):
        super().__init__()
        self.wm_classical = Word2Vec.load(wm_paths+"classical_wm_lunyu")
        self.wm_modern = Word2Vec.load(wm_paths+"modern_wm_lunyu")

        self.wv_classical = self.wm_classical.wv
        self.wv_modern = self.wm_modern.wv

        self.feature = []
        self.label = []

        with open(data_path, "r", encoding="utf-8") as file:
            datas = json.load(file)
            for data in datas:
                contents = data['contents']
                for content in contents:
                    source = []
                    for word in content['source']:
                        source.append(self.wv_classical[word])
                    target = []
                    for word in content['target']:
                        target.append(self.wv_modern[word])
                    self.feature.append(np.array(source))
                    self.label.append(np.array(target))


        max_length = 0
        for i in range(len(self.feature)):
            if self.feature[i].shape[0] > max_length:
                max_length = len(self.feature[i])
        for i in range(len(self.feature)):
            if self.feature[i].shape[0] < max_length:
                for k in range(max_length - self.feature[i].shape[0]):
                    self.feature[i] = np.concatenate((self.feature[i], zero), 0)

        max_length = 0
        for i in range(len(self.label)):
            if self.label[i].shape[0] > max_length:
                max_length = len(self.label[i])
        for i in range(len(self.label)):
            if self.label[i].shape[0] < max_length:
                for k in range(max_length - self.label[i].shape[0]):
                    self.label[i] = np.concatenate((self.label[i], zero), 0)

        self.feature = np.stack(self.feature)
        self.label = np.stack(self.label)


    def __len__(self):
        return len(self.feature)


    def __getitem__(self, index):
        return self.feature[index], self.label[index]
                
####################################################################################

class MyDataset_unembed(Dataset):
    def __init__(self, data_path):
        super().__init__()

        self.feature = []
        self.label = []

        with open(data_path, "r", encoding="utf-8") as file:
            datas = json.load(file)
            for data in datas:
                contents = data['contents']
                for content in contents:
                    source = []
                    for word in content['source']:
                        source.append(tokenizer(word, padding=True, truncation=True, max_length=128))
                    target = []
                    for word in content['target']:
                        target.append(tokenizer(word, padding=True, truncation=True, max_length=128))
                    self.feature.append(np.array(source))
                    self.label.append(np.array(target))

        self.feature = np.array(self.feature)
        self.label = np.array(self.label)

        
    def __len__(self):
        return len(self.feature)


    def __getitem__(self, index):
        return self.feature[index], self.label[index]