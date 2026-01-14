import json
from main import inference_result

_, unseen_abnormal = inference_result(True, f"./output/hdfs/batch_test_abnormal_1000_True_False_False_False.jsonl", "abnormal")
_, unseen_normal = inference_result(True, f"./output/hdfs/batch_test_normal_1000_True_False_False_False.jsonl", "normal")

_, few_abnormal = inference_result(True, f"./output/hdfs/batch_test_abnormal_1000_False_True_False_False_1.jsonl", "abnormal")
_, few_normal = inference_result(True, f"./output/hdfs/batch_test_normal_1000_False_True_False_False_1.jsonl", "normal")

_, stats_abnormal = inference_result(True, f"./output/hdfs/batch_test_abnormal_1000_False_False_True_False.jsonl", "abnormal")
_, stats_normal = inference_result(True, f"./output/hdfs/batch_test_normal_1000_False_False_True_False.jsonl", "normal")

results = []
for u, s, f in zip(unseen_abnormal, stats_abnormal, few_abnormal):
    cnt = 0
    if u == "abnormal" :
        cnt = cnt + 1
    if s == "abnormal" :
        cnt = cnt + 1
    if f == "abnormal" :
        cnt = cnt + 1
    if cnt >= 2 :
        results.append("abnormal")
    else :
        results.append("normal")
        
TP = results.count("abnormal")

results = []
for u, s, f in zip(unseen_normal, stats_normal, few_normal):
    cnt = 0
    if u == "normal" :
        cnt = cnt + 1
    if s == "normal" :
        cnt = cnt + 1
    if f == "normal" :
        cnt = cnt + 1
    if cnt >= 2 :
        results.append("normal")
    else :
        results.append("abnormal")
        
TN = results.count("normal")

FN = 1000 - TP
FP = 1000 - TN
P = 100 * TP / (TP + FP)
R = 100 * TP / (TP + FN)
F1 = 2 * P * R / (P + R)
print(f"@@@ Confusion Matrix @@@\nTP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}, P: {P}, R: {R}, F1: {F1}\n")