import torch
from tqdm import tqdm
import json
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator

from sentence_transformers import SentenceTransformer


# def get_embedding(text, tokenizer, model):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
#     with torch.no_grad():
#         outputs = model(**inputs)
#         embedding = outputs.last_hidden_state.mean(dim=1)
#     return embedding.squeeze().cpu().numpy()

dataset = "hdfs"


model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device = "cuda", model_kwargs = {"torch_dtype": torch.float16})
model.max_seq_length = 8192
model[0].tokenizer.model_max_length = 8192
model[0].auto_model.config.use_cache = False



import pickle
with open(f'./preprocessed_dataset/{dataset}/base_train_normal_embds.pkl', 'rb') as file:
    data = pickle.load(file)

# from transformers import AutoModel, AutoTokenizer
# embed_model = "meta-llama/Meta-Llama-3-8B"
# tokenizer = AutoTokenizer.from_pretrained(embed_model, trust_remote_code=True)
# model = AutoModel.from_pretrained(embed_model, torch_dtype=torch.float16, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")


embeds = []
log_data = [i["text"] for i in data]
embeds = [i["embedding"] for i in data]

# tokenizer = model.tokenizer
# cnt = 0
# for i in log_data :
#     # 토크나이즈
#     tokens = tokenizer(i, return_tensors="pt")
#     # 토큰 길이 확인
#     token_length = tokens["input_ids"].shape[1]
#     if token_length > 12000:
#         cnt = cnt + 1
# print(cnt)


embeds = np.vstack(embeds)




wcss = []
K_range = range(1, 20)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(embeds)
    wcss.append(kmeans.inertia_)

# 3. Kneedle로 elbow point 자동 탐지
kneedle = KneeLocator(K_range, wcss, curve='convex', direction='decreasing')
optimal_k = kneedle.elbow

# 4. 결과 출력 및 시각화
print(f"optimal_k : {optimal_k}")


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
cluster_labels = kmeans.fit_predict(embeds)

cluster_info = {}
for i in range(optimal_k):
    cluster_info[i] = []
    
for a, b, c in zip(log_data, embeds, cluster_labels) :
    cluster_info[c].append([a, b])


for i in cluster_info :
    print(len(cluster_info[i]))
normal_seq_data = f"./preprocessed_dataset/{dataset}/base_test_normal_100.json"
abnormal_seq_data = f"./preprocessed_dataset/{dataset}/base_test_abnormal_100.json"
with open(normal_seq_data, "r") as f:
    normal_data = json.load(f)
with open(abnormal_seq_data, "r") as f:
    abnormal_data = json.load(f)

# normal_data = normal_data[:10]
# abnormal_data = abnormal_data[:10]
import random
def make_prompt(dataset, data, label, kmeans, cluster_info) :
    prompts = []
    for i in tqdm(data) : 
        emb = model.encode(i["text"], normalize_embeddings=True)
        test_class = kmeans.predict([emb])
        candidates = cluster_info[test_class[0]]
        examples = random.sample(candidates, 1)
        prompt = f'''Classify the log sequence as normal or abnormal.
Normal Data Example : {examples}
Test Data: {i["text"]}
Answer(No explanation, only normal or abnormal) : '''
        prompts.append(prompt)
    return prompts



normal_prompts = make_prompt(dataset, normal_data, "", kmeans, cluster_info)
abnormal_prompts = make_prompt(dataset, abnormal_data, "", kmeans, cluster_info)

from openai import OpenAI
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
            cnt = cnt + 1
            preds.append("abnormal")
            
    print(f"context_length_exceeded :{cnt}")           
         
    results = []
    for d, prompt, pred in zip(data, prompts, preds) :
        result = {"sequence" : d, "prompt" : prompt, "prediction" : pred}
        results.append(result)
    if not os.path.exists(f"./output/{dataset}"):
        os.makedirs(f"./output/{dataset}")

    with open(f'./output/{dataset}/RAGLOG_{data_type}_{len(data)}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent="\t")
        
def inference_result(result_path, label) :
    with open(result_path, "r") as f:
        results_ = json.load(f)
    results = []
    cnt = 0
    for i in results_ :
        if isinstance(i["prediction"], str) :
            try : 
                prediction = i["prediction"].lower() 
            except : 
                cnt += 1
                # print(i)
                # print(data_type)
        else :
                prediction = i["prediction"]

        results.append(prediction)
    print(cnt)    
    if label == "normal" :
        cnt = results.count("normal")
    elif label == "abnormal" :
        cnt = results.count("abnormal")
    print(results)
    return cnt
predict(normal_prompts, normal_data, dataset, "test_normal")
predict(abnormal_prompts, abnormal_data, dataset, "test_abnormal")      
TP = inference_result(f"./output/{dataset}/RAGLOG_test_abnormal_100.json", "abnormal")
TN = inference_result(f"./output/{dataset}/RAGLOG_test_normal_100.json", "normal")

FN = 100 - TP
FP = 100 - TN
P = 100 * TP / (TP + FP)
R = 100 * TP / (TP + FN)
F1 = 2 * P * R / (P + R)
print(f"@@@ Confusion Matrix @@@\nTP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}, P: {P}, R: {R}, F1: {F1}\n")








# print(kmeans.predict([embeds[0]]))
# print(kmeans.predict([embeds[1]]))
# print(kmeans.predict([embeds[2]]))
# print(cluster_labels)
# from collections import Counter
# print(Counter(cluster_labels))


# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# # 2D로 축소
# pca = PCA(n_components=1024)
# emb_2d = pca.fit_transform(embeds)

# # 시각화
# plt.figure(figsize=(6, 4))
# plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=cluster_labels, cmap='tab10', s=10)
# plt.title("K-Means Clustering of Embeddings")
# plt.xlabel("PCA 1")
# plt.ylabel("PCA 2")
# plt.show()
# plt.savefig(f'{dataset}_kmeans_clustering.png')