from openai import OpenAI
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import argparse
import re

from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle

from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity    
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split

import os

from prompts import hdfs_prompt, bgl_prompt, tbird_prompt

def ask_llm_batch(prompt_file_name):
    secret_key = 'Your OpenAI Key'
    organization_key = 'Your OpenAI ORG Key'
    client = OpenAI(api_key = secret_key, organization = organization_key)

    batch_input_file = client.files.create(
        file=open(prompt_file_name, "rb"),
        purpose="batch"
    )
    
    batch_input_file_id = batch_input_file.id
    batch_info = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    batch = client.batches.retrieve(batch_info.id)
    print(batch)


def ask_llm (prompt) :
    secret_key = 'Your OpenAI Key'
    organization_key = 'Your OpenAI ORG Key'
    client = OpenAI(api_key = secret_key, organization = organization_key)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature = 0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return completion.choices[0].message.content


def ask_llm_ (prompt, pipe, model = "meta-llama/Meta-Llama-3-8B-Instruct") :
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": prompt
        }
    ]

    result = pipe(messages)
    
    return result[0]['generated_text'][2]["content"]

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    # print(len(inputs['input_ids'][0]))
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.squeeze().cpu().numpy()

def preprocess_embedding (dataset, model):
    # train_data_ = f"./preprocessed_dataset/{dataset}/base_train_normal_wo_valid.json"
    train_data_ = f"./preprocessed_dataset/{dataset}/base_train_normal.json"

    with open(train_data_, "r") as f:
        train_data = json.load(f)
    train_normal_data_embds = []
    train_normal_data = [i["text"] for i in train_data]
    print("Similarity")
    # embed_model = "meta-llama/Meta-Llama-3-8B"
    # tokenizer = AutoTokenizer.from_pretrained(embed_model, trust_remote_code=True)
    # model = AutoModel.from_pretrained(embed_model, torch_dtype=torch.float16, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # train_normal_data_embds = model.encode(train_normal_data, batch_size=1, show_progress_bar=True, normalize_embeddings=True)
    train_normal_data_embds = []
    print(len(train_normal_data))
    for i in tqdm(train_normal_data[:]) :
        train_normal_data_embds.append(model.encode(i, normalize_embeddings=True))
    final_data = []
    for i, j in zip(train_normal_data, train_normal_data_embds) :
        final_data.append({"text": i, "embedding": j})
    # with open(f"./preprocessed_dataset/{dataset}/base_train_normal_wo_valid_embds.pkl", "wb") as f :
    #     pickle.dump(final_data, f) 
    with open(f"./preprocessed_dataset/{dataset}/base_train_normal_embds.pkl", "wb") as f :
        pickle.dump(final_data, f)  
        
def preprocess_desc_embedding (dataset, model):
    # train_data_ = f"./preprocessed_dataset/{dataset}/base_train_normal_wo_valid.json"
    train_desc = f"./preprocessed_dataset/{dataset}/train_desc_sequence.json"

    with open(train_desc, "r") as f:
        train_data = json.load(f)
    train_normal_data_embds = []
    train_normal_data = [i["text"] for i in train_data]
    print("Similarity")
    # embed_model = "meta-llama/Meta-Llama-3-8B"
    # tokenizer = AutoTokenizer.from_pretrained(embed_model, trust_remote_code=True)
    # model = AutoModel.from_pretrained(embed_model, torch_dtype=torch.float16, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # train_normal_data_embds = model.encode(train_normal_data, batch_size=1, show_progress_bar=True, normalize_embeddings=True)
    train_normal_data_embds = []
    for i in tqdm(train_normal_data[:]) :
        train_normal_data_embds.append(model.encode(i, normalize_embeddings=True))
    final_data = []
    for i, j in zip(train_normal_data, train_normal_data_embds) :
        final_data.append({"text": i, "embedding": j})
    # with open(f"./preprocessed_dataset/{dataset}/base_train_normal_wo_valid_embds.pkl", "wb") as f :
    #     pickle.dump(final_data, f) 
    with open(f"./preprocessed_dataset/{dataset}/train_desc_seq_emb.pkl", "wb") as f :
        pickle.dump(final_data, f)          
# def make_description (dataset, pipe, tokenizer, model, log_datas):
#     dataset_info = {
#         "hdfs": {
#             "prompt": "You are a bot that writes a description that best describes the log sequence. You will be given a log sequence from the HDFS log dataset. The HDFS dataset consists of log messages generated during a Hadoop-based MapReduce job running on an Amazon EC2 node.\nYou're role is to generate a description that best describes the log sequence. Rather than focusing on each log template in the sequence, explain the overall characteristics of the sequence or patterns that are difficult to observe directly in the sequence.",
#         },
#         "bgl": {
#             "prompt": "You are a bot that writes a description that best describes the log template. This log is from BGL log dataset. The BGL dataset consists of log messages collected from the BlueGene/L supercomputer system operating at Lawrence Livermore National Labs (LLNL).\nYou will be given a log template. You're role is to generate a description that best describes the log template.",            
#         },
#         "tbird": {
#             "prompt": "You are a bot that writes a description that best describes the log template. This log is from Thunderbird log dataset. Thunderbird is an open dataset of logs collected from a Thunderbird supercomputer system at Sandia National Labs (SNL) in Albuquerque, with 9,024 processors and 27,072GB memory. The log contains alert and non-alert messages identified by alert category tags.\nYou will be given a log template. You're role is to generate a description that best describes the log template.",
#         }
#     }

    
#     prompts = []
#     for i in log_datas[:] :
#         messages = [
#             {
#                 "role": "system",
#                 "content": dataset_info[dataset]["prompt"]
#             },
#             {
#                 "role": "user",
#                 "content": f"Log sequence: {i}\nDescription:"
#             }
#         ]
#         prompts.append(messages)
        
#     def batch_iter(data, size):
#         for i in range(0, len(data), size):
#             yield data[i:i+size]

#     all_results = []
#     batch_size = 32
#     for batch in tqdm(batch_iter(prompts, batch_size), total=(len(prompts) + batch_size - 1)//batch_size, desc="Generating descriptions"):
#         outputs = pipe(batch, max_new_tokens=512, do_sample=False, temperature=None, top_p=None, batch_size=batch_size)
#         all_results.extend([r[0]["generated_text"][2]["content"] for r in outputs])


#     model.eval()
        
#     generated_text = []
#     embedding = []
#     for i in all_results:
#         gen_text = i
#         embed = get_embedding(gen_text, tokenizer, model)
#         generated_text.append(gen_text)
#         embedding.append(embed.tolist())

#     return generated_text, embedding

def make_description (dataset, log_data):
    dataset_info = {
        "hdfs": {
            "prompt": "You are a bot that writes a description that best describes the log template. You will be given a log template from the HDFS log dataset. The HDFS dataset consists of log messages generated during a Hadoop-based MapReduce job running on an Amazon EC2 node.\nYou're role is to generate a description that best describes the log template. Write in about 10 words.",
        },
        "bgl": {
            "prompt": "You are a bot that writes a description that best describes the log template. This log is from BGL log dataset. The BGL dataset consists of log messages collected from the BlueGene/L supercomputer system operating at Lawrence Livermore National Labs (LLNL).\nYou will be given a log template. You're role is to generate a description that best describes the log template.",            
        },
        "tbird": {
            "prompt": "You are a bot that writes a description that best describes the log template. This log is from Thunderbird log dataset. Thunderbird is an open dataset of logs collected from a Thunderbird supercomputer system at Sandia National Labs (SNL) in Albuquerque, with 9,024 processors and 27,072GB memory. The log contains alert and non-alert messages identified by alert category tags.\nYou will be given a log template. You're role is to generate a description that best describes the log template.",
        }
    }

    
    secret_key = 'Your OpenAI Key'
    organization_key = 'Your OpenAI ORG Key'
    client = OpenAI(api_key = secret_key, organization = organization_key)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature = 0,
        messages = [
            {
                "role": "system",
                "content": dataset_info[dataset]["prompt"]
            },
            {
                "role": "user",
                "content": f"Log template: {log_data}\nDescription:"
            }
        ]
    )
    return "'" + completion.choices[0].message.content+"'"


def predict (prompts, data, dataset, data_type, compare_nv, seq_info, compare_sim, cot_prompt) :
    preds = []
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model_infer = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"  # ✅ 여러 GPU 사용
    )

    pipe = pipeline(
        "text-generation",
        model=model_infer,
        tokenizer=tokenizer,
        # device=0,            # ❌ 제거
        pad_token_id=tokenizer.eos_token_id,
        max_length=8192,
    )

    pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    pipe.tokenizer.padding_side = 'left'

    cnt = 0
    for i in tqdm(prompts) :
        try : 
            # prediction = ask_llm(i)
            prediction = ask_llm_(i, pipe)
            preds.append(prediction)
            
        except :
            preds.append("{\"classification\" : \"abnormal\"}")
            cnt = cnt + 1
    print(f"context_length_exceeded :{cnt}")

                
    results = []
    for d, prompt, pred in zip(data, prompts, preds) :
        result = {"sequence" : d, "prompt" : prompt, "prediction" : pred}
        results.append(result)
    if not os.path.exists(f"./output/{dataset}"):
        os.makedirs(f"./output/{dataset}")

    with open(f'./output/{dataset}/{data_type}_{len(data)}_{compare_nv}_{seq_info}_{compare_sim}_{cot_prompt}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent="\t")
    

def inference_result (llm_batch, data_path, data_type) :
    data_list = []
    if llm_batch == True : 
        with open(data_path, "r") as f:
            for line in f:
                data_list.append(json.loads(line))
        # print(data_list[0]["response"]["body"]["choices"][0]["message"]["content"])
        results = []
        for idx, i in enumerate(data_list) :
            prediction = i["response"]["body"]["choices"][0]["message"]["content"]
            # prediction_ = prediction.split('":')[1].split(",")[0].strip()
            if isinstance(prediction, str) :
                try : 
                    prediction = json.loads(prediction)
                    prediction = prediction["classification"]
                except :
                    prediction = prediction.split('\"classification\"')[1].split(",")[0].replace(":", "").replace('\"', "").strip()
            else :
                    prediction = prediction
                    prediction = prediction["classification"]

            results.append(prediction)
        for _ in range(len(results), 1000) :
            results.append("abnormal")
    else : 
        with open(data_path, "r") as f:
            results_ = json.load(f)
        results = []
        for i in results_ :
            if isinstance(i["prediction"], str) :
                try : 
                    prediction = json.loads(i["prediction"])
                    prediction = prediction["classification"]
                except :
                    prediction = i["prediction"].split('\"classification\"')[1].split(",")[0].replace(":", "").replace('\"', "").strip()
            else :
                    prediction = i["prediction"]
                    prediction = prediction["classification"]

            results.append(prediction)
    if data_type == "normal" :
        cnt = results.count("normal")
    elif data_type == "abnormal" :
        cnt = results.count("abnormal")
    print(results)
    return cnt, results


def make_prompt (dataset, data, data_type, model, compare_nv, seq_info, compare_sim, cot_prompt) :

    # case 1: Log Similarity
    with open(f"./preprocessed_dataset/{dataset}/base_train_normal_embds.pkl", "rb") as f:
        sequence_data = pickle.load(f)
    vocab_desc_file = f'./preprocessed_dataset/{dataset}/train_vocab_desc.json'
    # with open(f"./preprocessed_dataset/{dataset}/train_desc_seq_emb.pkl", "rb") as f:
    #     sequence_data = pickle.load(f)
    train_normal_data_embds = []
    if compare_sim == True or seq_info == True:
        train_normal_data = [i["text"] for i in sequence_data]
        train_normal_data_embds = [i["embedding"] for i in sequence_data]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("Similarity")
        
        # embed_model = "meta-llama/Meta-Llama-3-8B"
        # tokenizer = AutoTokenizer.from_pretrained(embed_model, trust_remote_code=True)
        # model = AutoModel.from_pretrained(embed_model, torch_dtype=torch.float16, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
        
        tokenizer = None

        wcss = []
        K_range = range(1, 20)
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(train_normal_data_embds)
            wcss.append(kmeans.inertia_)

        kneedle = KneeLocator(K_range, wcss, curve='convex', direction='decreasing')
        optimal_k = kneedle.elbow

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(train_normal_data_embds)

        cluster_info = {}
        for i in range(optimal_k):
            cluster_info[i] = []
            
        for a, c in zip(train_normal_data_embds, cluster_labels) :
            cluster_info[c].append(a)
            
        # for i in tqdm(train_normal_data[1:]) :
        #     emb = get_embedding(i, tokenizer, model)
        #     train_normal_data_embds.append(emb.tolist())
        # train_normal_data = train_normal_data[1:]
        
    # case 2: Log Description
    # with open(f"./preprocessed_dataset/{dataset}/base_train_normal_wo_valid_desc.json", "r") as f:
    #     train_normal_desc = json.load(f)
    # if compare_sim == True :
    #     train_normal_data = [i["text"] for i in train_normal_desc][1:]
    #     train_normal_data_descs = [i["description"] for i in train_normal_desc][1:]
    #     train_normal_data_embds = [i["embedding"] for i in train_normal_desc][1:]
    #     embed_model = "meta-llama/Meta-Llama-3-8B-Instruct"
    #     tokenizer = AutoTokenizer.from_pretrained(embed_model, trust_remote_code=True)
    #     model = AutoModel.from_pretrained(embed_model, torch_dtype=torch.float16, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
    #     pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", device=0, pad_token_id=tokenizer.eos_token_id, trust_remote_code=True, torch_dtype=torch.float16)
    #     pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
    #     pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
    #     pipe.tokenizer.padding_side='left'
    else :
        train_normal_data = []
        train_normal_data_embds = []
        cluster_info = {}
        kmeans = None
        device = None
        tokenizer = None
        model = None
        
    prompts = []
    batch_input = []
    for idx, input_data in tqdm(enumerate(data[:])) :
        if dataset == "hdfs" :
            # prompt_ = hdfs_prompt(input_data, train_normal_data, train_normal_data_embds, pipe, tokenizer, model, compare_nv, seq_info, compare_sim, cot_prompt)
            prompt_ = hdfs_prompt(input_data, train_normal_data, train_normal_data_embds, device, tokenizer, model, kmeans, cluster_info, compare_nv, seq_info, compare_sim, cot_prompt, vocab_desc_file)
        elif dataset == "bgl" :
            prompt_ = bgl_prompt(input_data, train_normal_data, train_normal_data_embds, device, tokenizer, model, kmeans, cluster_info, compare_nv, seq_info, compare_sim, cot_prompt, vocab_desc_file)
        elif dataset == "tbird" :
            prompt_ = tbird_prompt(input_data, train_normal_data, train_normal_data_embds, device, tokenizer, model, kmeans, cluster_info, compare_nv, seq_info, compare_sim, cot_prompt, vocab_desc_file)
        
        if compare_nv == False and seq_info == False and compare_sim == False and cot_prompt == False :
            prompt = prompt_["prompt_base"] + prompt_["prompt_response"]
        else : 
            prompt = prompt_["prompt_dataset"] + prompt_["prompt_base"] + prompt_["prompt_cot"] +  prompt_["prompt_vocab"] + prompt_["prompt_sim"] + prompt_["prompt_seq"] + prompt_["prompt_response"]
        temp = {"custom_id": f"req-{idx}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": 1000}}
        batch_input.append(temp)
        prompts.append(prompt)

    if not os.path.exists(f"./prompts/{dataset}"):
        os.makedirs(f"./prompts/{dataset}")

    with open(f"./prompts/{dataset}/{data_type}_{len(batch_input)}_{compare_nv}_{seq_info}_{compare_sim}_{cot_prompt}_prompt.jsonl", "w") as f:
        for data in batch_input:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    return prompts




def make_temp_seq_data (dataset, data_type): 
    if data_type == "test_normal" :
        data_path = f"./logbert/output/{dataset}/test_normal"
    elif data_type == "test_abnormal" :
        data_path = f"./logbert/output/{dataset}/test_abnormal"
    elif data_type == "train_normal" :
        data_path = f"./logbert/output/{dataset}/train"
        
    if dataset == "hdfs" :
        seq_temp_path = f"./logbert/output/hdfs/hdfs_log_templates.json"
        with open(seq_temp_path, 'r') as f :
            seq_temp_data = json.load(f)
        seq_temp_data = {v:k for k, v in seq_temp_data.items()}

        template_data = pd.read_csv("./logbert/output/hdfs/HDFS.log_templates.csv")
            

        template_dict = {}
        for template, e_id in zip(template_data["EventTemplate"], template_data["EventId"]):
            template_dict[e_id] = template

        sequence_data = []

        with open(data_path, 'r') as f:
            while True :
                temp = []
                d = f.readline()
                if not d: 
                    break
                for i in d.replace("\n", "").split(" ") :
                    temp.append(int(i))
                sequence_data.append(temp)

        str_sequence_data = []
        for sequence in tqdm(sequence_data) :
            temp = []
            for j in sequence : 
                temp.append("'"+template_dict[seq_temp_data[j]]+"'")
            temp_str = ", ".join(temp)

            str_sequence_data.append({"text":temp_str})

    elif dataset == "bgl" or dataset == "tbird" :
        if dataset == "bgl" :
            template_data_path = "./logbert/output/bgl/BGL.log_templates.csv"
        elif dataset == "tbird" :
            template_data_path = "./logbert/output/tbird/Thunderbird_20M.log_templates.csv"
            
        template_data = pd.read_csv(template_data_path)
        
        template_dict = {}
        for template, e_id in zip(template_data["EventTemplate"], template_data["EventId"]):
            if e_id not in template_dict.keys() :
                template_dict[e_id] = template
         
        sequence_data = []       
        with open(data_path, 'r') as f:
            while True :
                temp = []
                d = f.readline()
                if not d: 
                    break
                for i in d.strip().replace("\n", "").split(" ") :
                    temp.append(i)
                sequence_data.append(temp)
        str_sequence_data = []
        for sequence in tqdm(sequence_data) :
            temp = []
            for j in sequence : 
                temp.append("'"+template_dict[j]+"'")
            temp_str = ", ".join(temp)

            str_sequence_data.append({"text":temp_str})
    if not os.path.exists(f"./preprocessed_dataset/{dataset}"):
        os.makedirs(f"./preprocessed_dataset/{dataset}")
    with open(f"./preprocessed_dataset/{dataset}/base_{data_type}.json", "w") as f :
        json.dump(str_sequence_data, f, indent=4)

# def make_valid_samples (dataset):
#     # test_normal
#     seq_data = f"./preprocessed_dataset/{dataset}/base_test_normal.json"
#     with open(seq_data, "r") as f:
#         sequence_data = json.load(f)
#     final_data = []
#     for i in sequence_data :
#         if len(i["text"].split(", ")) >= 10 :
#             final_data.append(i)
#     test_normal = final_data
#     with open(f'./preprocessed_dataset/{dataset}/base_test_normal_100.json', 'w', encoding='utf-8') as f:
#         json.dump(test_normal[:100], f, indent="\t")
#     with open(f'./preprocessed_dataset/{dataset}/base_test_normal_250.json', 'w', encoding='utf-8') as f:
#         json.dump(test_normal[:250], f, indent="\t")
#     with open(f'./preprocessed_dataset/{dataset}/base_test_normal_1000.json', 'w', encoding='utf-8') as f:
#         json.dump(test_normal[:1000], f, indent="\t")
#     with open(f'./preprocessed_dataset/{dataset}/base_test_normal_2000.json', 'w', encoding='utf-8') as f:
#         json.dump(test_normal[:2000], f, indent="\t")
#     print("test_normal", len(test_normal))
    
#     # valid_normal
#     seq_data = f"./preprocessed_dataset/{dataset}/base_train_normal.json"
#     with open(seq_data, "r") as f:
#         sequence_data = json.load(f)
#     final_data = []
#     for i in sequence_data :
#         if len(i["text"].split(", ")) >= 10 :
#             final_data.append(i)
            
    
    
#     train_normal, valid_normal = train_test_split(final_data, test_size=0.1, random_state=42)
#     with open(f'./preprocessed_dataset/{dataset}/base_valid_normal.json', 'w', encoding='utf-8') as f:
#         json.dump(valid_normal, f, indent="\t")
#     with open(f'./preprocessed_dataset/{dataset}/base_train_normal_wo_valid.json', 'w', encoding='utf-8') as f:
#         json.dump(train_normal, f, indent="\t")    
#     print("valid_normal", len(valid_normal))
#     print("train_normal_wo_valid", len(train_normal))
    
#     # normal vocab
#     def update_keys(data, keys):
#         for row in data:
#             keys.update([entry.strip() for entry in row["text"].split(",") if entry.strip()])
#         keys = list(keys)
#         return keys
#     train_keys = set()
#     train_keys = update_keys(final_data, train_keys)
#     with open(f'./preprocessed_dataset/{dataset}/train_vocab.json', 'w', encoding='utf-8') as f:
#         json.dump(train_keys, f, indent="\t")
           
#     train_keys = set()
#     train_keys = update_keys(train_normal, train_keys)
#     with open(f'./preprocessed_dataset/{dataset}/train_vocab_wo_valid.json', 'w', encoding='utf-8') as f:
#         json.dump(train_keys, f, indent="\t")


#     f = open(f"./logbert/output/{dataset}/test_normal", "r") 
#     sequence_data = f.readlines()
#     final_data = []
#     for i in sequence_data :
#         i = i.replace("\n", "")
#         if len(i.split(" ")) >= 10 :
#             final_data.append(i)
#     test_normal = final_data
#     with open(f"./logbert/output/{dataset}/test_normal_100", "w") as f:
#         for i in test_normal[:100]:
#             f.write(i)
#             f.write("\n")
#     f.close()
#     with open(f"./logbert/output/{dataset}/test_normal_250", "w") as f:
#         for i in test_normal[:250]:
#             f.write(i)
#             f.write("\n")
#     f.close()
#     with open(f"./logbert/output/{dataset}/test_normal_1000", "w") as f:
#         for i in test_normal[:1000]:
#             f.write(i)
#             f.write("\n")
#     f.close()
#     with open(f"./logbert/output/{dataset}/test_normal_2000", "w") as f:
#         for i in test_normal[:2000]:
#             f.write(i)
#             f.write("\n")
#     f.close()
    
#     f = open(f"./logbert/output/{dataset}/train", "r") 
#     sequence_data = f.readlines()
#     final_data = []
#     for i in sequence_data :
#         i = i.replace("\n", "")
#         if len(i.split(" ")) >= 10 :
#             final_data.append(i)
#     train_normal, valid_normal = train_test_split(final_data, test_size=0.1, random_state=42)
#     with open(f"./logbert/output/{dataset}/valid_normal", "w") as f:
#         for i in valid_normal:
#             f.write(i)
#             f.write("\n")
#     f.close()
#     with open(f"./logbert/output/{dataset}/train_normal_wo_valid", "w") as f:
#         for i in train_normal:
#             f.write(i)
#             f.write("\n")
#     f.close()
    
#     # test_abnormal
#     seq_data = f"./preprocessed_dataset/{dataset}/base_test_abnormal.json"
#     with open(seq_data, "r") as f:
#         sequence_data = json.load(f)
#     final_data = []
#     for i in sequence_data :
#         if len(i["text"].split(", ")) >= 10 :
#             final_data.append(i)
#     test_abnormal, valid_abnormal = train_test_split(final_data, test_size=0.1, random_state=42)
#     with open(f'./preprocessed_dataset/{dataset}/base_test_abnormal_100.json', 'w', encoding='utf-8') as f:
#         json.dump(test_abnormal[:100], f, indent="\t")
#     with open(f'./preprocessed_dataset/{dataset}/base_test_abnormal_250.json', 'w', encoding='utf-8') as f:
#         json.dump(test_abnormal[:250], f, indent="\t")
#     with open(f'./preprocessed_dataset/{dataset}/base_test_abnormal_1000.json', 'w', encoding='utf-8') as f:
#         json.dump(test_abnormal[:1000], f, indent="\t")
#     with open(f'./preprocessed_dataset/{dataset}/base_test_abnormal_2000.json', 'w', encoding='utf-8') as f:
#         json.dump(test_abnormal[:2000], f, indent="\t")
#     with open(f'./preprocessed_dataset/{dataset}/base_test_abnormal_wo_valid.json', 'w', encoding='utf-8') as f:
#         json.dump(test_abnormal[:], f, indent="\t")
#     with open(f'./preprocessed_dataset/{dataset}/base_valid_abnormal.json', 'w', encoding='utf-8') as f:
#         json.dump(valid_abnormal, f, indent="\t")
        
#     print("valid_abnormal", len(valid_abnormal))
#     print("test_abnormal_wo_valid", len(test_abnormal))
#     f = open(f"./logbert/output/{dataset}/test_abnormal", "r") 
#     sequence_data = f.readlines()
#     f.close()
    
#     final_data = []
#     for i in sequence_data :
#         i = i.replace("\n", "")
#         if len(i.split(" ")) >= 10 :
#             final_data.append(i)
#     test_abnormal, valid_abnormal = train_test_split(final_data, test_size=0.1, random_state=20)
#     with open(f"./logbert/output/{dataset}/test_abnormal_100", "w") as f:
#         for i in test_abnormal[:100]:
#             f.write(i)
#             f.write("\n")
#     f.close()
#     with open(f"./logbert/output/{dataset}/test_abnormal_250", "w") as f:
#         for i in test_abnormal[:250]:
#             f.write(i)
#             f.write("\n")
#     f.close()
#     with open(f"./logbert/output/{dataset}/test_abnormal_1000", "w") as f:
#         for i in test_abnormal[:1000]:
#             f.write(i)
#             f.write("\n")
#     f.close()
#     with open(f"./logbert/output/{dataset}/test_abnormal_2000", "w") as f:
#         for i in test_abnormal[:2000]:
#             f.write(i)
#             f.write("\n")
#     f.close()
#     with open(f"./logbert/output/{dataset}/test_abnormal_wo_valid", "w") as f:
#         for i in test_abnormal[:]:
#             f.write(i)
#             f.write("\n")
#     f.close()
#     with open(f"./logbert/output/{dataset}/valid_abnormal", "w") as f:
#         for i in valid_abnormal: 
#             f.write(i)
#             f.write("\n")
#     f.close()
    
def make_valid_samples (dataset):
    # test_normal
    seq_data = f"./preprocessed_dataset/{dataset}/base_test_normal.json"
    with open(seq_data, "r") as f:
        sequence_data = json.load(f)
    final_data = []
    for i in sequence_data :
            final_data.append(i)
    test_normal = final_data
    with open(f'./preprocessed_dataset/{dataset}/base_test_normal_100.json', 'w', encoding='utf-8') as f:
        json.dump(test_normal[:100], f, indent="\t")
    with open(f'./preprocessed_dataset/{dataset}/base_test_normal_250.json', 'w', encoding='utf-8') as f:
        json.dump(test_normal[:250], f, indent="\t")
    with open(f'./preprocessed_dataset/{dataset}/base_test_normal_1000.json', 'w', encoding='utf-8') as f:
        json.dump(test_normal[:1000], f, indent="\t")
    with open(f'./preprocessed_dataset/{dataset}/base_test_normal_2000.json', 'w', encoding='utf-8') as f:
        json.dump(test_normal[:2000], f, indent="\t")
    print("test_normal", len(test_normal))
    
    # train_normal
    seq_data = f"./preprocessed_dataset/{dataset}/base_train_normal.json"
    with open(seq_data, "r") as f:
        sequence_data = json.load(f)
    final_data = []
    for i in sequence_data :
        final_data.append(i)
    
    # normal vocab
    def update_keys(data, keys):
        for row in data:
            keys.update([entry.strip().replace("'", "") for entry in row["text"].split("',") if entry.strip()])
        keys = list(keys)
        return keys
    
    train_keys = set()
    train_keys = update_keys(final_data, train_keys)
    with open(f'./preprocessed_dataset/{dataset}/train_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(train_keys, f, indent="\t")

    vocab_descs = {}
    for i in tqdm(train_keys) :
        vocab_desc = make_description(dataset, i)
        vocab_descs[i] = vocab_desc
    with open(f'./preprocessed_dataset/{dataset}/train_vocab_desc.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_descs, f, indent="\t")
    with open(f'./preprocessed_dataset/{dataset}/total_vocab_desc.json', 'w', encoding='utf-8') as f:
        json.dump(vocab_descs, f, indent="\t")
    
    ############### 여기에 만드는중
    desc_sequences = []
    for log_sequence in final_data :
        log_sequence = str(log_sequence["text"]).split("', ")
        desc_sequence = []
        for i in log_sequence :
            i = i.replace("'","").strip()
            vocab_desc = vocab_descs[i]
            desc_sequence.append(vocab_desc)
        # # 현재 방법    
        desc_sequence = ", ".join(desc_sequence)
        # 1. 2. 3. 
        # desc_sequence_ = ""
        # for idx, i in enumerate(desc_sequence) :
        #     desc_sequence_ = desc_sequence_ + str(idx) + ". " + i + " "
        # desc_sequence = desc_sequence_
        # Log 1 is.
        # desc_sequence_ = ""
        # for idx, i in enumerate(desc_sequence) :
        #     desc_sequence_ = desc_sequence_ + "Log " + str(idx) + " is " + i + " "
        # desc_sequence = desc_sequence_
        # # random shuffle
        # import random
        # random.shuffle(desc_sequence)
        # desc_sequence = ", ".join(desc_sequence)     
        desc_sequences.append({"text":desc_sequence})
        
    with open(f'./preprocessed_dataset/{dataset}/train_desc_sequence.json', 'w', encoding='utf-8') as f:
        json.dump(desc_sequences, f, indent="\t") 
    
    f = open(f"./logbert/output/{dataset}/test_normal", "r") 
    sequence_data = f.readlines()
    final_data = []
    for i in sequence_data :
        i = i.replace("\n", "")
        final_data.append(i)
    test_normal = final_data
    with open(f"./logbert/output/{dataset}/test_normal_100", "w") as f:
        for i in test_normal[:100]:
            f.write(i)
            f.write("\n")
    f.close()
    with open(f"./logbert/output/{dataset}/test_normal_250", "w") as f:
        for i in test_normal[:250]:
            f.write(i)
            f.write("\n")
    f.close()
    with open(f"./logbert/output/{dataset}/test_normal_1000", "w") as f:
        for i in test_normal[:1000]:
            f.write(i)
            f.write("\n")
    f.close()
    with open(f"./logbert/output/{dataset}/test_normal_2000", "w") as f:
        for i in test_normal[:2000]:
            f.write(i)
            f.write("\n")
    f.close()
    
    f = open(f"./logbert/output/{dataset}/train", "r") 
    sequence_data = f.readlines()
    final_data = []
    for i in sequence_data :
        i = i.replace("\n", "")
        final_data.append(i)
        
    with open(f"./logbert/output/{dataset}/train_normal", "w") as f:
        for i in final_data:
            f.write(i)
            f.write("\n")
    f.close()
    
    # test_abnormal
    seq_data = f"./preprocessed_dataset/{dataset}/base_test_abnormal.json"
    with open(seq_data, "r") as f:
        sequence_data = json.load(f)
    final_data = []
    for i in sequence_data :
        final_data.append(i)
    test_abnormal = final_data
    with open(f'./preprocessed_dataset/{dataset}/base_test_abnormal_100.json', 'w', encoding='utf-8') as f:
        json.dump(test_abnormal[:100], f, indent="\t")
    with open(f'./preprocessed_dataset/{dataset}/base_test_abnormal_250.json', 'w', encoding='utf-8') as f:
        json.dump(test_abnormal[:250], f, indent="\t")
    with open(f'./preprocessed_dataset/{dataset}/base_test_abnormal_1000.json', 'w', encoding='utf-8') as f:
        json.dump(test_abnormal[:1000], f, indent="\t")
    with open(f'./preprocessed_dataset/{dataset}/base_test_abnormal_2000.json', 'w', encoding='utf-8') as f:
        json.dump(test_abnormal[:2000], f, indent="\t")


        
    f = open(f"./logbert/output/{dataset}/test_abnormal", "r") 
    sequence_data = f.readlines()
    f.close()
    
    final_data = []
    for i in sequence_data :
        i = i.replace("\n", "")
        final_data.append(i)
    test_abnormal = final_data
    with open(f"./logbert/output/{dataset}/test_abnormal_100", "w") as f:
        for i in test_abnormal[:100]:
            f.write(i)
            f.write("\n")
    f.close()
    with open(f"./logbert/output/{dataset}/test_abnormal_250", "w") as f:
        for i in test_abnormal[:250]:
            f.write(i)
            f.write("\n")
    f.close()
    with open(f"./logbert/output/{dataset}/test_abnormal_1000", "w") as f:
        for i in test_abnormal[:1000]:
            f.write(i)
            f.write("\n")
    f.close()
    with open(f"./logbert/output/{dataset}/test_abnormal_2000", "w") as f:
        for i in test_abnormal[:2000]:
            f.write(i)
            f.write("\n")
    f.close()






def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="preprocess", help="choose 'preprocess' or 'inference' or 'confusion_matrix'"
    )
    parser.add_argument(
        "--dataset", type=str, default="hdfs", help="choose 'hdfs' or 'bgl' or 'tbird'"
    )
    parser.add_argument(
        "--num_samples", type=int, default=100, help="number of samples to use for inference"
    )
    
    parser.add_argument(
        "--llm_batch", action="store_true"
    )
    
    
    # args for prompt
    parser.add_argument(
        "--compare_nv", action="store_true"
    )
    parser.add_argument(
        "--seq_info", action="store_true"
    )
    parser.add_argument(
        "--compare_sim", action="store_true"
    )
    parser.add_argument(
        "--cot_prompt", action="store_true"
    )
    
    

    args = parser.parse_args()

    return args



        
        
def main():
    args = parse_args()
    print(args)
    
    if args.mode == "preprocess" :
        make_temp_seq_data(args.dataset, "test_normal")
        make_temp_seq_data(args.dataset, "test_abnormal")
        make_temp_seq_data(args.dataset, "train_normal")
        
        make_valid_samples(args.dataset)
        
        model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device = "cuda", model_kwargs = {"torch_dtype": torch.float16})
        model.max_seq_length = 8192
        model[0].tokenizer.model_max_length = 8192
        model[0].auto_model.config.use_cache = False
        preprocess_embedding (args.dataset, model)
        preprocess_desc_embedding (args.dataset, model)
        
        
        # # train data description and embedding
        # train_data_ = f"./preprocessed_dataset/{args.dataset}/base_train_normal_wo_valid.json"
        # with open(train_data_, "r") as f:
        #     train_data = json.load(f)
        
        # # embed_model = "meta-llama/Meta-Llama-3-8B-Instruct"
        # # tokenizer = AutoTokenizer.from_pretrained(embed_model, trust_remote_code=True)
        # # model = AutoModel.from_pretrained(embed_model, torch_dtype=torch.float16, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
        # # pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", device=0, pad_token_id=tokenizer.eos_token_id, trust_remote_code=True, torch_dtype=torch.float16)
        # # pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
        # # pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        # # pipe.tokenizer.padding_side='left'
        
        # # log_data = [i["text"] for i in train_data]
        # # description, embedding = make_description(args.dataset, pipe, tokenizer, model, log_data)

        # # final_data = []
        # # for i, j, k in zip(log_data, description, embedding) :
        # #     final_data.append({"text": i, "description": j, "embedding": k})
        # # with open(f"./preprocessed_dataset/{args.dataset}/base_train_normal_wo_valid_desc.json", "w") as f :
        # #     json.dump(final_data, f, indent=4)
            
            
            
        # train_normal_data_embds = []
        # train_normal_data = [i["text"] for i in train_data]
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print("Similarity")
        # embed_model = "meta-llama/Meta-Llama-3-8B"
        # tokenizer = AutoTokenizer.from_pretrained(embed_model, trust_remote_code=True)
        # model = AutoModel.from_pretrained(embed_model, torch_dtype=torch.float16, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
        # for i in tqdm(train_normal_data[:]) :
        #     train_normal_data_embds.append(get_embedding(i, tokenizer, model))
            
        # final_data = []
        # for i, j in zip(train_normal_data, train_normal_data_embds) :
        #     final_data.append({"text": i, "embedding": j})
        # with open(f"./preprocessed_dataset/{args.dataset}/base_train_normal_wo_valid_embds.pkl", "wb") as f :
        #     pickle.dump(final_data, f)  
             
    elif args.mode == "inference" : 
        normal_seq_data = f"./preprocessed_dataset/{args.dataset}/base_test_normal_{args.num_samples}.json"
        abnormal_seq_data = f"./preprocessed_dataset/{args.dataset}/base_test_abnormal_{args.num_samples}.json"
        with open(normal_seq_data, "r") as f:
            normal_data = json.load(f)
        with open(abnormal_seq_data, "r") as f:
            abnormal_data = json.load(f)
            
        if (args.compare_sim == True) or (args.seq_info == True):
            model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device = "cuda", model_kwargs = {"torch_dtype": torch.float16})
            model.max_seq_length = 8192
            model[0].tokenizer.model_max_length = 8192
            model[0].auto_model.config.use_cache = False
        else :
            model = None
            
        normal_prompts = make_prompt(args.dataset, normal_data, "normal", model, args.compare_nv, args.seq_info, args.compare_sim, args.cot_prompt)
        abnormal_prompts = make_prompt(args.dataset, abnormal_data, "abnormal", model, args.compare_nv, args.seq_info, args.compare_sim, args.cot_prompt)

        if args.llm_batch == False :
            predict(normal_prompts, normal_data, args.dataset, "test_normal", args.compare_nv, args.seq_info, args.compare_sim, args.cot_prompt)
            predict(abnormal_prompts, abnormal_data, args.dataset, "test_abnormal", args.compare_nv, args.seq_info, args.compare_sim, args.cot_prompt)
        else :
            ask_llm_batch(f"./prompts/{args.dataset}/normal_{args.num_samples}_{args.compare_nv}_{args.seq_info}_{args.compare_sim}_{args.cot_prompt}_prompt.jsonl")
            ask_llm_batch(f"./prompts/{args.dataset}/abnormal_{args.num_samples}_{args.compare_nv}_{args.seq_info}_{args.compare_sim}_{args.cot_prompt}_prompt.jsonl")

    elif args.mode == "confusion_matrix" :
        
        if args.llm_batch == True :
            TP, _ = inference_result(args.llm_batch, f"./output/{args.dataset}/batch_test_abnormal_{args.num_samples}_{args.compare_nv}_{args.seq_info}_{args.compare_sim}_{args.cot_prompt}.jsonl", "abnormal")
            TN, _ = inference_result(args.llm_batch, f"./output/{args.dataset}/batch_test_normal_{args.num_samples}_{args.compare_nv}_{args.seq_info}_{args.compare_sim}_{args.cot_prompt}.jsonl", "normal")
        if args.llm_batch == False :
            TP, _ = inference_result(args.llm_batch, f"./output/{args.dataset}/test_abnormal_{args.num_samples}_{args.compare_nv}_{args.seq_info}_{args.compare_sim}_{args.cot_prompt}.json", "abnormal")
            TN, _ = inference_result(args.llm_batch, f"./output/{args.dataset}/test_normal_{args.num_samples}_{args.compare_nv}_{args.seq_info}_{args.compare_sim}_{args.cot_prompt}.json", "normal")
           
        FN = args.num_samples - TP
        FP = args.num_samples - TN
        P = 100 * TP / (TP + FP)
        R = 100 * TP / (TP + FN)
        F1 = 2 * P * R / (P + R)
        print(f"@@@ Confusion Matrix @@@\nTP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}, P: {P}, R: {R}, F1: {F1}\n")

if __name__ == "__main__":
    main()
