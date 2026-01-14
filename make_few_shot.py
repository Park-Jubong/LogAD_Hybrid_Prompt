import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from transformers import pipeline, AutoTokenizer, AutoModel
import json
import pickle

from main import ask_llm, get_embedding, make_description, hdfs_prompt, bgl_prompt, tbird_prompt


dataset = "tbird"

with open(f"./preprocessed_dataset/{dataset}/base_train_normal_embds.pkl", "rb") as f:
    sequence_data = pickle.load(f)
        
train_normal_data = [i["text"] for i in sequence_data]
train_normal_data_embds = [i["embedding"] for i in sequence_data][1:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


embed_model = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(embed_model, trust_remote_code=True)
model = AutoModel.from_pretrained(embed_model, torch_dtype=torch.float16, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
input_data = train_normal_data[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if dataset == "hdfs":
    hdfs_prompt({"text": input_data}, train_normal_data[1:], train_normal_data_embds, device, tokenizer, model, True, True, True, True)
elif dataset == "bgl":
    bgl_prompt({"text": input_data}, train_normal_data[1:], train_normal_data_embds, device, tokenizer, model, True, True, True, True)
elif dataset == "tbird":
    tbird_prompt({"text": input_data}, train_normal_data[1:], train_normal_data_embds, device, tokenizer, model, True, True, True, True)
