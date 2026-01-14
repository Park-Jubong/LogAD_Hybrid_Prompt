import json
from tqdm import tqdm
import pandas as pd


dataset = "hdfs"

with open(f"./preprocessed_dataset/{dataset}/train_vocab.json", "r") as f:
    vocab = json.load(f)
# print(vocab)
with open(f"./preprocessed_dataset/{dataset}/base_test_abnormal.json", "r") as f:
    data = json.load(f)
    
    
def point_conditional(data, normal_vocab):
    conditional = []
    if isinstance(data, pd.Series):
        data = data.apply(lambda x: eval(x))
    for line_ in tqdm(data, desc="point_conditional"):
        line = line_["text"].split(", ")
        # print(line)
        # break
        is_cond = True
        for log in line:
            if log not in normal_vocab:
                is_cond = False
                break
        conditional.append(is_cond)
    # count the number of conditional sequences
    print(f"Number of conditional sequences: {sum(conditional)} / {len(conditional)}")
    return conditional

conditional = point_conditional(data, vocab)