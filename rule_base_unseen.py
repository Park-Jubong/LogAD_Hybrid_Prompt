import json

dataset = "tbird"
with open(f"./preprocessed_dataset/{dataset}/base_test_normal_1000.json", "r") as f:
    test_normal_data = json.load(f)
with open(f"./preprocessed_dataset/{dataset}/base_test_abnormal_1000.json", "r") as f:
    test_abnormal_data = json.load(f)
    
results = []
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
        results.append("abnormal")
    else :
        results.append("normal")
        
TP = results.count("abnormal")


results = []
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
        results.append("abnormal")
    else :
        results.append("normal")

TN = results.count("normal")

FN = 1000 - TP
FP = 1000 - TN
P = 100 * TP / (TP + FP)
R = 100 * TP / (TP + FN)
F1 = 2 * P * R / (P + R)
print(f"@@@ Confusion Matrix @@@\nTP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}, P: {P}, R: {R}, F1: {F1}\n")