import pickle
import json

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from transformers import pipeline, AutoTokenizer, AutoModel
import json
import pickle

from main import ask_llm, get_embedding, make_description, hdfs_prompt, bgl_prompt, tbird_prompt

with open("./preprocessed_dataset/hdfs/base_train_normal_embds.pkl", "rb") as f:
    train = pickle.load(f)
train_normal_data_embds = [i["embedding"] for i in train][:]
embed_model = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(embed_model, trust_remote_code=True)
model = AutoModel.from_pretrained(embed_model, torch_dtype=torch.float16, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")

with open("./preprocessed_dataset/hdfs/base_test_normal_100.json", "r") as f:
    data = json.load(f)
normal_data = [i["text"] for i in data[1:11]]

with open("./preprocessed_dataset/hdfs/base_test_abnormal_100.json", "r") as f:
    data = json.load(f)
abnormal_data = [i["text"] for i in data[1:11]]


print("normal")
for i in normal_data: 
    emb = get_embedding(i, tokenizer, model)
    sims = cosine_similarity(np.vstack([emb]), np.vstack(train_normal_data_embds))[0]
    max_index = np.argmax(sims)
    
    prompt = f'''Compare the two data sets to see if they are similar.
D1 - Max: 1.00000, Mean: 0.99522, Std: 0.00499
D2 - Max: {round(sims[max_index], 5)}, Mean: {round(np.mean(sims), 5)}, Std: {round(np.std(sims), 5)}
Answer (no explanation, only yes or no)'''
    print(f"{round(sims[max_index], 5)}, {round(np.mean(sims), 5)}, {round(np.std(sims), 5)}")
    prediction = ask_llm(prompt)
    print(prediction)


print("abnormal")
for i in abnormal_data: 
    emb = get_embedding(i, tokenizer, model)
    sims = cosine_similarity(np.vstack([emb]), np.vstack(train_normal_data_embds))[0]
    max_index = np.argmax(sims)
    
    prompt = f'''Compare the two data sets to see if they are similar.
D1 - Max: 1.00000, Mean: 0.99522, Std: 0.00499
D2 - Max: {round(sims[max_index], 5)}, Mean: {round(np.mean(sims), 5)}, Std: {round(np.std(sims), 5)}
Answer (no explanation, only yes or no)'''
    print(f"{round(sims[max_index], 5)}, {round(np.mean(sims), 5)}, {round(np.std(sims), 5)}")
    prediction = ask_llm(prompt)
    print(prediction)