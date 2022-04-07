#coding:utf-8
from __future__ import print_function
"""This is a simple python script reconstruction of the notebook train.ipynb."""
"""It may be used in the future for UI"""
# necessary imports
import torch
from torch import nn
import numpy as np
from lgg_model import *
from gensim.models import Word2Vec


punc = ['，','。','？','！','；','【','】', '）', '（', ' ', '\n', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
del_lst = []


def generate(word_model_path, lgg_model_path, head):
    # load the word2vec model and language model
    word_model_path = 'word_model_paths/' + word_model_path
    word_model = Word2Vec.load(word_model_path)
    wv = word_model.wv
    lgg_model_path = 'lgg_model_paths/' + lgg_model_path
    lgg_model = torch.load(lgg_model_path)
    lgg_model.eval()

    head = head + '。'
    lst = list(head) # default to be 4 Chinese characters

    for i in lst:
        if i not in wv.key_to_index:
            del_lst.append(i)
    for i in del_lst:
        lst.remove(i)

    data = np.array([])
    count = (len(head)-1) * 5

    print("输入：", head[:-1])
    print("输出：")

    idx = wv.key_to_index[lst[0]]

    i = 0

    while i < count:
        i += 1
        new_word = wv.index_to_key[idx]
        print(new_word, end='')

        # lst.append(new_word)
        data = np.append(data, idx)
        data = np.stack((data,))

        x = torch.Tensor(data)
        x = x.to(torch.long)
        y = lgg_model(x)[0][-1]
        p = y.detach().numpy()
        p = softmax(p)

        idx = np.random.choice(np.arange(len(wv)), p=p)

        if (i % 5 != 0):
            while wv.index_to_key[idx] in punc:
                idx = np.random.choice(np.arange(len(wv)), p=p)

        if i % 5 == 0 and i != 0: # default for 5 characters in a sentence
            if i % 10 == 0:
                idx = wv.key_to_index['。']

            else:
                idx = wv.key_to_index['，']

            data = np.append(data, idx)
            data = np.stack((data,))

            new_word = wv.index_to_key[idx]
            print(new_word, end='')

            if i % 10 == 0:
                print()

            head_idx = i // 5
            head_word = head[head_idx]
            idx = wv.key_to_index[head_word]


if __name__ == '__main__':

    word_model_path = '唐诗2'
    lgg_model_path = '唐诗2_2022-03-20_13_57_25'

    head = input("请输入藏头诗的”头“：") # 如：秋风萧瑟
    generate(word_model_path, lgg_model_path, head)