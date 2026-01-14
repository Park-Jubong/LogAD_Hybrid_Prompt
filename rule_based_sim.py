import json
import pickle
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from main import get_embedding
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from kneed import KneeLocator
from openai import OpenAI


dataset = "tbird"

with open(f"./preprocessed_dataset/{dataset}/base_test_normal_1000.json", "r") as f:
    test_normal_data = json.load(f)
with open(f"./preprocessed_dataset/{dataset}/base_test_abnormal_1000.json", "r") as f:
    test_abnormal_data = json.load(f)
    
# with open(f"./preprocessed_dataset/hdfs/train_desc_seq_emb.pkl", "rb") as f:
with open(f"./preprocessed_dataset/{dataset}/base_train_normal_embds.pkl", "rb") as f:  
    sequence_data = pickle.load(f)

        
train_normal_data = [i["text"] for i in sequence_data]
train_normal_data_embds = [i["embedding"] for i in sequence_data]

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
    

model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device = "cuda", model_kwargs = {"torch_dtype": torch.float16})
model.max_seq_length = 8192
model[0].tokenizer.model_max_length = 8192
model[0].auto_model.config.use_cache = False


from tqdm import tqdm

abnormal_d = []
for i in tqdm(test_abnormal_data) :
    # vocab_desc_file = f'./preprocessed_dataset/hdfs/train_vocab_desc.json'

    # with open(vocab_desc_file, "r") as f :
    #     vocab_descs = json.load(f)
    # with open(f'./preprocessed_dataset/hdfs/total_vocab_desc.json', "r") as f :
    #     total_vocab_descs = json.load(f)
        
    # log_sequence = str(i["text"]).split("', ")
    # desc_sequence = []
    # for i in log_sequence :
    #     i = i.replace("'","").strip()
    #     if i in vocab_descs.keys() :
    #         vocab_desc = vocab_descs[i]
    #         desc_sequence.append(vocab_desc)
    #     else : 
    #         if i in total_vocab_descs.keys():
    #             desc = total_vocab_descs[i]
    #         else:
    #             desc = make_description("hdfs", i)
    #             total_vocab_descs[i] = desc
    #         desc_sequence.append(desc)

    #         with open(f'./preprocessed_dataset/hdfs/total_vocab_desc.json', "w") as f :
    #             json.dump(total_vocab_descs, f, indent="\t")
    # desc_sequence = ", ".join(desc_sequence)
    emb = model.encode(i["text"], normalize_embeddings=True)
    
    # emb = model.encode(i["text"], normalize_embeddings=True)

    cluster_class = kmeans.predict([emb])
    centroid = np.vstack(cluster_info[cluster_class[0]]).mean(axis=0)
    
    sims = cosine_similarity(np.vstack([emb]), np.vstack(cluster_info[cluster_class[0]]))[0]
    d2_mean = round(np.mean(sims), 5)
    d2_std = round(np.std(sims), 5)
    
    max_sims = cosine_similarity(np.vstack([centroid]), np.vstack(cluster_info[cluster_class[0]]))[0]
    d1_mean = round(np.mean(max_sims), 5)
    d1_std = round(np.std(max_sims), 5)
    
    pooled_sd = np.sqrt((d1_std**2 + d2_std**2) / 2)
    d = abs(d1_mean - d2_mean) / pooled_sd

    abnormal_d.append(d)


normal_d = []
for i in tqdm(test_normal_data) :
    # vocab_desc_file = f'./preprocessed_dataset/hdfs/train_vocab_desc.json'

    # with open(vocab_desc_file, "r") as f :
    #     vocab_descs = json.load(f)
    # with open(f'./preprocessed_dataset/hdfs/total_vocab_desc.json', "r") as f :
    #     total_vocab_descs = json.load(f)
        
    # log_sequence = str(i["text"]).split("', ")
    # desc_sequence = []
    # for i in log_sequence :
    #     i = i.replace("'","").strip()
    #     if i in vocab_descs.keys() :
    #         vocab_desc = vocab_descs[i]
    #         desc_sequence.append(vocab_desc)
    #     else : 
    #         if i in total_vocab_descs.keys():
    #             desc = total_vocab_descs[i]
    #         else:
    #             desc = make_description("hdfs", i)
    #             total_vocab_descs[i] = desc
    #         desc_sequence.append(desc)

    #         with open(f'./preprocessed_dataset/hdfs/total_vocab_desc.json', "w") as f :
    #             json.dump(total_vocab_descs, f, indent="\t")
    # desc_sequence = ", ".join(desc_sequence)
    emb = model.encode(i["text"], normalize_embeddings=True)
    
    # emb = model.encode(i["text"], normalize_embeddings=True)
    cluster_class = kmeans.predict([emb])
    centroid = np.vstack(cluster_info[cluster_class[0]]).mean(axis=0)
    
    sims = cosine_similarity(np.vstack([emb]), np.vstack(cluster_info[cluster_class[0]]))[0]
    d2_mean = round(np.mean(sims), 5)
    d2_std = round(np.std(sims), 5)
    
    max_sims = cosine_similarity(np.vstack([centroid]), np.vstack(cluster_info[cluster_class[0]]))[0]
    d1_mean = round(np.mean(max_sims), 5)
    d1_std = round(np.std(max_sims), 5)
    
    pooled_sd = np.sqrt((d1_std**2 + d2_std**2) / 2)
    d = abs(d1_mean - d2_mean) / pooled_sd

    normal_d.append(d)

for thresh in tqdm(np.arange(0, 5, 0.1)): 

    results = []
    for d in abnormal_d :
        if d > thresh :
            results.append("abnormal")
        else :
            results.append("normal")
    TP = results.count("abnormal")
    
    results = []
    for d in normal_d :
        if d > thresh :
            results.append("abnormal")
        else :
            results.append("normal")
    TN = results.count("normal")
    
    
    FN = 1000 - TP
    FP = 1000 - TN
    P = 100 * TP / (TP + FP)
    R = 100 * TP / (TP + FN)
    F1 = 2 * P * R / (P + R)
    
    print(f"{thresh}")
    print(f"P: {P}, R: {R}, F1: {F1}\n")

print("==============ensemble===============")
thresh = 0.8
abnormal_results = []
for d in abnormal_d :
    if d > thresh :
        abnormal_results.append("abnormal")
    else :
        abnormal_results.append("normal")



abnormal_results_ = []
for i in test_abnormal_data :
    with open(f"./preprocessed_dataset/{dataset}/train_vocab.json", "r") as f :
        vocab = json.load(f)
    log_sequence = str(i["text"]).split("', ")
    log_sequence = [i.replace("'","").strip() for i in log_sequence]
    checklist = []
    for log in log_sequence :
        if log not in vocab :
            checklist.append(log)
    checklist = list(set(checklist))
    if len(checklist) > 0 :
        abnormal_results_.append("abnormal")
    else :
        abnormal_results_.append("normal")

results = []
for i, j in zip(abnormal_results, abnormal_results_) :
    if i == "normal" and j == "normal" :
        results.append("normal")
    else :
        results.append("abnormal")
        
TN = results.count("abnormal") 


normal_results = []
for d in normal_d :
    if d > thresh :
        normal_results.append("abnormal")
    else :
        normal_results.append("normal")
        
normal_results_ = []
for i in test_normal_data :
    with open(f"./preprocessed_dataset/{dataset}/train_vocab.json", "r") as f :
        vocab = json.load(f)
    log_sequence = str(i["text"]).split("', ")
    log_sequence = [i.replace("'","").strip() for i in log_sequence]
    checklist = []
    for log in log_sequence :
        if log not in vocab :
            checklist.append(log)
    checklist = list(set(checklist))
    if len(checklist) > 0 :
        normal_results_.append("abnormal")
    else :
        normal_results_.append("normal")


results = []
for i, j in zip(normal_results, normal_results_) :
    if i == "normal" and j == "normal" :
        results.append("normal")
    else :
        results.append("abnormal")
        
TP = results.count("normal")


FN = 1000 - TP
FP = 1000 - TN
P = 100 * TP / (TP + FP)
R = 100 * TP / (TP + FN)
F1 = 2 * P * R / (P + R)

print(f"{thresh}")
print(f"P: {P}, R: {R}, F1: {F1}\n")