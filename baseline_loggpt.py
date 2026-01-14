import torch
from tqdm import tqdm
import json
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator



dataset = "hdfs"

import pickle
with open(f'./preprocessed_dataset/{dataset}/base_train_normal_embds.pkl', 'rb') as file:
    data = pickle.load(file)


log_data = [i["text"] for i in data]


normal_seq_data = f"./preprocessed_dataset/{dataset}/base_test_normal_100.json"
abnormal_seq_data = f"./preprocessed_dataset/{dataset}/base_test_abnormal_100.json"
with open(normal_seq_data, "r") as f:
    normal_data = json.load(f)
with open(abnormal_seq_data, "r") as f:
    abnormal_data = json.load(f)

# normal_data = normal_data[:10]
# abnormal_data = abnormal_data[:10]

import random
def make_prompt(dataset, data, label, train_data) :
    prompts = []
    for i in data : 
        examples = random.sample(train_data, 1)
        prompt = f'''Your task is to determine whether a given set of logs contains an anomaly or not and provide a binary classification: 0 for normal and 1 for the anomaly. If an anomaly, you should generate reports and suggest relevant preventive measures. Note that do not a lot of text.
Output format: Return back in json format, including keys: is_anomaly, reports, preventive_measures.
Here are some normal logs:
{examples}
Input log data (a python list):
[{i["text"]}]
Output:
'''
        prompts.append(prompt)
    return prompts


normal_prompts = make_prompt(dataset, normal_data, "", log_data)
abnormal_prompts = make_prompt(dataset, abnormal_data, "", log_data)

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

infer_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"  # ✅ 여러 GPU 사용
)

pipe = pipeline(
    "text-generation",
    model=infer_model,
    tokenizer=tokenizer,
    # device=0,            # ❌ 제거
    pad_token_id=tokenizer.eos_token_id,
    max_length=8192,
)

pipe.tokenizer.pad_token = pipe.tokenizer.eos_token
pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
pipe.tokenizer.padding_side = 'left'
# def ask_llm (prompt) :
#     secret_key = 'Your OpenAI Key'
#     organization_key = 'Your OpenAI ORG Key'
#     client = OpenAI(api_key = secret_key, organization = organization_key)

#     completion = client.chat.completions.create(
#         model="gpt-4o-mini",
#         temperature = 0,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {
#                 "role": "user",
#                 "content": prompt
#             }
#         ]
#     )
#     return completion.choices[0].message.content
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

import os
def predict (prompts, data, dataset, data_type) :
    preds = []
    cnt = 0 
    for i in tqdm(prompts) :
        try : 
            prediction = ask_llm_(i, pipe)
            preds.append(prediction)
            
        except :
            preds.append('"is_anomaly": 1')
            cnt = cnt + 1
    print(f"context_length_exceeded :{cnt}")
    results = []
    for d, prompt, pred in zip(data, prompts, preds) :
        result = {"sequence" : d, "prompt" : prompt, "prediction" : pred}
        results.append(result)
    if not os.path.exists(f"./output/{dataset}"):
        os.makedirs(f"./output/{dataset}")

    with open(f'./output/{dataset}/LOGGPT_{data_type}_{len(data)}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent="\t")
        
def inference_result(result_path, label) :
    with open(result_path, "r") as f:
        results_ = json.load(f)
    results = []
    cnt = 0
    for i in results_ :
        if isinstance(i["prediction"], str) :
            try :
                prediction = i["prediction"].split('"is_anomaly":')[1].split(",")[0].strip()
            except : 
                cnt += 1
                # print(i)
                # print(data_type)
        else :
                prediction = i["prediction"]

        results.append(prediction)
    print(cnt)    
    if label == "normal" :
        cnt = results.count("0")
    elif label == "abnormal" :
        cnt = results.count("1")
    print(results)
    return cnt
predict(normal_prompts, normal_data, dataset, "test_normal")
predict(abnormal_prompts, abnormal_data, dataset, "test_abnormal")  
    
TP = inference_result(f"./output/{dataset}/LOGGPT_test_abnormal_100.json", "abnormal")
TN = inference_result(f"./output/{dataset}/LOGGPT_test_normal_100.json", "normal")

FN = 100 - TP
FP = 100 - TN
P = 100 * TP / (TP + FP)
R = 100 * TP / (TP + FN)
F1 = 2 * P * R / (P + R)


print(f"@@@ Confusion Matrix @@@\nTP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}, P: {P}, R: {R}, F1: {F1}\n")