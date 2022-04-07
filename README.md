- [Intro2AI_NLP](#intro2ai_nlp)
  - [Contents](#contents)
  - [Logs](#logs)
    - [2022/3/18](#2022318)
    - [2022/3/19](#2022319)
    - [2022/3/20](#2022320)
    - [2022/3/21](#2022321)
    - [2022/3/22](#2022322)
    - [2022/3/24](#2022324)
    - [2022/3/26](#2022326)
    - [2022/3/27](#2022327)
    - [2022/4/7](#202247)
  - [Experiment records](#experiment-records)

# Intro2AI_NLP
This is the course project for the NLP class of ***Introduction to AI*** in Peking University.
## Contents
- [Head-hidden poem generator](head_hidden_poem_generator)
- [Passage generator](passage_generator)
- [Jin Yong style novel clip generator](jinyong_novelist_colab)
- [PKU hole generator](pkuhole)
## Logs
### 2022/3/18
Topic: [Head-hidden poem generator](head_hidden_poem_generator)
- Add [dataload.py](head_hidden_poem_generator/dataload.py) and [lgg_model.py](head_hidden_poem_generator/lgg_model.py) to make some basic preparations.
- Add some toy datasets.
- Add a naive version of [train.ipynb](head_hidden_poem_generator/train.ipynb).
- First develop a head-hidden poem *(藏头诗)* generator system, which can generate a poem with a given head-word. But its performance is highly unstable, which means it needs more debugging and tuning.

### 2022/3/19
Topic: [Head-hidden poem generator](head_hidden_poem_generator)
- Retrain a new model using a much bigger dataset (around 110kb).
- Add a markdown file to record experiment details and results.
- Stabilize the generation module to generate poems in a fixed pattern.

### 2022/3/20
Topic: [Head-hidden poem generator](head_hidden_poem_generator)
- Use 4-layer LSTM, which is much worse than the 2-layer one.
- Use 1-layer LSTM, which is much better than the 2-layer one. Less is More! However, without dropout, the model is overfitting.
- Use 2-layer GRU, which is slightly better than 2-layer LSTM on the same condition.
- Reconstruct the script for training of HHP. See [trainer.py](head_hidden_poem_generator/trainer.py) for details.
 
### 2022/3/21
Topic: [Passage generator](passage_generator)
- Reconstruct the passage generator module.

### 2022/3/22
Topic: [Head-hidden poem generator](head_hidden_poem_generator)
- Add [a generator python script](head_hidden_poem_generator/generator.py) to quickly generate poems. ***Warning: you shall only run the script in the path ```head_hidden_poem_generator```!!!***

### 2022/3/24
Topic: [Classical Chinese](classical_chinese)
- Create a new project folder about the classical Chinese part of the NCEE.

### 2022/3/26
Topic: [Classical Chinese](classical_chinese)
- Create a seq2seq model for NMT, with poor performance and countless bugs though.

### 2022/3/27
Topic: [Classical Chinese](classical_chinese)
- Add pretrained models for translation and punctuation.

### 2022/4/7
Upload 2 new subprojects.

- - -

## Experiment records
- Head-hidden poems
  
  Check out [experiment.md](head_hidden_poem_generator/experiment.md) for details.

- Passage generator

  Check out [this repository](https://github.com/yxKryptonite/language_model_from_scratch_pytorch) for details.
