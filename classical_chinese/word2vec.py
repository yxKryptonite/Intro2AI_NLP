"""convert json file into txt file"""
"""Then use these txt file to create word models"""
from distutils import text_file
import json
from gensim.models import Word2Vec
import multiprocessing

path = "dataset/lunyu.json"

json_file = open(path, "r", encoding="utf-8")
datas = json.load(json_file)
json_file.close()

txt_file_classical = open("dataset/lunyu_classical.txt", "w", encoding="utf-8")
txt_file_modern = open("dataset/lunyu_modern.txt", "w", encoding="utf-8")

classical = []
modern = []

for data in datas:
    contents = data['contents']
    for content in contents:
        classical.append(content['source'])
        modern.append(content['target'])

txt_file_classical.write("\n".join(classical))
txt_file_modern.write("\n".join(modern))

txt_file_classical.close()
txt_file_modern.close()


with open("dataset/lunyu_classical.txt", "r", encoding="utf-8") as f:
    classical = f.read().split("\n")
    classical_wm = Word2Vec(classical, vector_size=100, 
                            window=5, min_count=1, workers=multiprocessing.cpu_count(), epochs=100)
    classical_wm.save("word_model_paths/classical_wm_lunyu")


with open("dataset/lunyu_modern.txt", "r", encoding="utf-8") as f:
    modern = f.read().split("\n")
    modern_wm = Word2Vec(modern, vector_size=100, 
                            window=5, min_count=1, workers=multiprocessing.cpu_count(), epochs=100)
    modern_wm.save("word_model_paths/modern_wm_lunyu")
