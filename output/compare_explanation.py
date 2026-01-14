import json

def get_results(results_) :
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
    return results

file_1 = "temp_sent"
file_2 = "desc_sent"

with open (f"./hdfs/test_normal_100_False_False_False_False_{file_1}.json", "r") as f:
    normal_1 = json.load(f)
with open (f"./hdfs/test_abnormal_100_False_False_False_False_{file_1}.json", "r") as f:
    abnormal_1 = json.load(f)
    
with open (f"./hdfs/test_normal_100_False_False_False_False_{file_2}.json", "r") as f:
    normal_2 = json.load(f)
with open (f"./hdfs/test_abnormal_100_False_False_False_False_{file_2}.json", "r") as f:
    abnormal_2 = json.load(f)
    
normal_1_result = get_results(normal_1)
abnormal_1_result = get_results(abnormal_1)
normal_2_result = get_results(normal_2)
abnormal_2_result = get_results(abnormal_2)


# file 2는 맞았는데 file 1은 틀린 경우
for i in range(len(normal_1)):
    if normal_1_result[i] != "normal" and normal_2_result[i] == "normal":
        print(f"=== {i} ===")
        print(normal_1[i]["prediction"])
        print(normal_2[i]["prediction"])
        
for i in range(len(normal_1)):
    if abnormal_1_result[i] == "abnormal" and abnormal_2_result[i] != "abnormal":
        print(f"=== {i} ===")
        print(abnormal_1[i]["prediction"])
        print(abnormal_2[i]["prediction"])
        
