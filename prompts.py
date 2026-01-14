import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from transformers import pipeline, AutoTokenizer, AutoModel
import json

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

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.squeeze().cpu().numpy()

# def make_description (dataset, pipe, tokenizer, model, log_data):
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
    
#     messages = [
#         {
#             "role": "system",
#             "content": dataset_info[dataset]["prompt"]
#         },
#         {
#             "role": "user",
#             "content": f"Log sequence: {log_data}\nDescription:"
#         }
#     ]

#     output = pipe(messages, max_new_tokens=512, do_sample=False, temperature=None, top_p=None)

#     model.eval()
    
#     generated_text = output[0]["generated_text"][2]["content"]
#     embedding = get_embedding(generated_text, tokenizer, model).tolist()

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
                "content": f"Log sequence: {log_data}\nDescription:"
            }
        ]
    )
    return "'" + completion.choices[0].message.content+"'"

def hdfs_prompt(input_data, train_normal_data, train_normal_data_embds, device, tokenizer, model, kmeans, cluster_info, compare_nv, seq_info, compare_sim, cot_prompt, vocab_desc_file) :
    with open(vocab_desc_file, "r") as f :
        vocab_descs = json.load(f)
    with open(f'./preprocessed_dataset/hdfs/total_vocab_desc.json', "r") as f :
        total_vocab_descs = json.load(f)
    
    log_sequence = str(input_data["text"]).split("', ")
    log_sequence = [i.replace("'","").strip() for i in log_sequence]
    log_sequence_ = ["'"+ i+"'" for i in log_sequence]
    # # 현재 방법
    # log_sequence = ", ".join(log_sequence)
    # 1. 2. 3. 
    # log_sequence_ = ""
    # for idx, i in enumerate(log_sequence) :
    #     log_sequence_ = log_sequence_ + str(idx) + ". " + i + " "
    # log_sequence = log_sequence_
    # # Log 1 is.
    # log_sequence_ = ""
    # for idx, i in enumerate(log_sequence) :
    #     log_sequence_ = log_sequence_ + "Log " + str(idx) + " is " + i + " "
    # log_sequence = log_sequence_
    # random shuffle
    # import random
    # random.shuffle(log_sequence)
    
    log_sequence_ = ", ".join(log_sequence_)
    
    
    desc_sequence = []
    for i in log_sequence :
        if i in vocab_descs.keys() :
            vocab_desc = vocab_descs[i]
            desc_sequence.append(vocab_desc)
        else : 
            if i in total_vocab_descs.keys():
                desc = total_vocab_descs[i]
            else:
                desc = make_description("hdfs", i)
                total_vocab_descs[i] = desc
            desc_sequence.append(desc)

            with open(f'./preprocessed_dataset/hdfs/total_vocab_desc.json', "w") as f :
                json.dump(total_vocab_descs, f, indent="\t")
    # 현재 방법
    desc_sequence = ", ".join(desc_sequence)
    # # 1. 2. 3. 
    # desc_sequence_ = ""
    # for idx, i in enumerate(desc_sequence) :
    #     desc_sequence_ = desc_sequence_ + str(idx) + ". " + i + " "
    # desc_sequence = desc_sequence_
    # Log 1 is.
    # desc_sequence_ = ""
    # for idx, i in enumerate(desc_sequence) :
    #     desc_sequence_ = desc_sequence_ + "Log " + str(idx) + " is " + i + " "
    # desc_sequence = desc_sequence_
    # random shuffle
    # import random
    # random.shuffle(desc_sequence)
    # desc_sequence = ", ".join(desc_sequence)     

    prompt_base = f'''## Instruction : 
Determine whether the log sequence is normal or abnormal by synthesizing the additional information given below.
log sequence : {log_sequence_}


'''
    prompt_dataset = '''## Domain Knowledge
# Dataset Explanation
HDFS(Hadoop Distributed File System) log dataset contains system logs generated by Hadoop clusters, capturing various events like file operations, errors, and access details.
These logs help in monitoring, debugging, and analyzing system performance and failures.
They are widely used in anomaly detection, fault diagnosis, and predictive maintenance in big data environments.
'''        
    if compare_nv == True : 
        with open(f"./preprocessed_dataset/hdfs/train_vocab.json", "r") as f :
            vocab = json.load(f)
        log_sequence = str(input_data["text"]).split("', ")
        log_sequence = [i.replace("'","").strip() for i in log_sequence]
        checklist = []
        for log in log_sequence :
            if log not in vocab :
                checklist.append(log)
        checklist = list(set(checklist))
        unseen_info = []
        for i in checklist : 
            unseen_info.append(f"- {i} : {total_vocab_descs[i]}")
#         if len(checklist) > 0 : 
#             vocab_info = f'''- Comparison Explanation: The log sequence is likely to contain the log of the Unseen Normal Vocab or the log of the Abnormal Vocab.
# - Unseen Normal Vocab or the log of the Abnormal Vocab in the provided log sequence: {checklist}'''
#         else :
#             vocab_info = f"- Comparison Explanation: The log sequence is likely to be normal because it does not contain the log of abnormal vocab, but it may be abnormal under certain conditions."
        prompt_vocab = f'''## Entry based Analysis
Normal Log Entry refers to the log templates that appear in the Normal Train Log Sequence.
If a log that is not in the normal log entry appears in the provided Log sequence, it provides what the log is.
# Comparison Results
- Unseen log from provided log sequence: {checklist}
'''


#         with open(f"./preprocessed_dataset/hdfs/base_train_normal.json", "r") as f :
#             train_data = json.load(f)
#         train_vocabs = []
#         for i in train_data :
#             temp = i["text"].split(", ")
#             for j in temp :
#                 train_vocabs.append(j)
#         from collections import Counter
#         train_vocabs = Counter(train_vocabs)
#         log_sequence = str(input_data["text"]).split(", ")
#         vocab_info = []
#         for i in log_sequence :
#             vocab_info.append(f"{i} : {train_vocabs[i]}")
#         # print(vocab_info)
#         prompt_vocab = f'''# Comparison with Normal Vocab
# This shows how many times each log in the provided log sequence appears in the log sequences of the training data. 
# If there are logs in the provided log sequence that never appear in the training data, you should examine them in detail.
# # Comparison Results
# {vocab_info}
# '''
            
        
    
    else : prompt_vocab = ""
    
    if compare_sim == True or seq_info == True :
        emb = model.encode(log_sequence_, normalize_embeddings=True)
    if compare_sim == True :
        # emb = get_embedding(input_data["text"], tokenizer, model)
        # emb = model.encode(log_sequence_, normalize_embeddings=True)

        # sims = cosine_similarity(np.vstack([emb]), np.vstack(train_normal_data_embds))[0]

        # centroid = np.vstack(train_normal_data_embds).mean(axis=0)
        # max_sims = cosine_similarity(np.vstack([centroid]), np.vstack(train_normal_data_embds))[0]

        # max_index = np.argmax(sims)
        # max_emb = train_normal_data_embds[max_index]
        # max_sims = cosine_similarity(np.vstack([max_emb]), np.vstack(train_normal_data_embds))[0]

        
        cluster_class = kmeans.predict([emb])
        centroid = np.vstack(cluster_info[cluster_class[0]]).mean(axis=0)
        # centroid = np.vstack(train_normal_data_embds).mean(axis=0)

        sims = cosine_similarity(np.vstack([emb]), np.vstack(cluster_info[cluster_class[0]]))[0]
        max_sims = cosine_similarity(np.vstack([centroid]), np.vstack(cluster_info[cluster_class[0]]))[0]
        # sims = cosine_similarity(np.vstack([emb]), np.vstack(train_normal_data_embds))[0]
        # max_sims = cosine_similarity(np.vstack([centroid]), np.vstack(train_normal_data_embds))[0]
        
        # print(sims[max_index])
        # print(train_normal_data[max_index])
        # print(np.mean(sims))
        # print(np.std(sims))
        
        # import random
        # max_index = random.randint(0, len(train_normal_data))
        
        import random
#         prompt_sim = f'''## Comparison with Example Normal Train Dataset
# We have a Train set consisting of 4369 normal log sequences.
# First, as an example, we provide the results of comparing the Similarity between one normal data and the log sequences of the train set. 
# Then, we provide the results of comparing the Similarity between the provided test data and the log sequences of the train set. 
# Depending on whether the test data is normal or abnormal, the results of comparing the Similarity may differ, so refer to the results of the two comparisons.
# # Example (Normal Log Sequence)
# Example of normal log sequence : "Receiving block blk_<*> src: <*> dest: /<*>:50010, Receiving block blk_<*> src: <*> dest: /<*>:50010, Receiving block blk_<*> src: <*> dest: /<*>:50010, BLOCK* NameSystem.allocateBlock: <*> blk_<*>, PacketResponder <*> for block blk_<*> <*>, Received block blk_<*> of size <*> from /<*>, PacketResponder <*> for block blk_<*> <*>, Received block blk_<*> of size <*> from /<*>, PacketResponder <*> for block blk_<*> <*>, Received block blk_<*> of size <*> from /<*>, BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:50010 is added to blk_<*> size <*>, BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:50010 is added to blk_<*> size <*>, BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:50010 is added to blk_<*> size <*>, Verification succeeded for blk_<*>, BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:50010, BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:50010, BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:50010, Deleting block blk_<*> file <*><*>, Deleting block blk_<*> file <*><*>, Deleting block blk_<*> file <*><*>"
# - The most similar log sequence in the Train set (log sequence / similarity): BLOCK* NameSystem.allocateBlock: <*> blk_<*>, Receiving block blk_<*> src: <*> dest: /<*>:50010, Receiving block blk_<*> src: <*> dest: /<*>:50010, Receiving block blk_<*> src: <*> dest: /<*>:50010, BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:50010 is added to blk_<*> size <*>, PacketResponder <*> for block blk_<*> <*>, Received block blk_<*> of size <*> from /<*>, BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:50010 is added to blk_<*> size <*>, BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:50010 is added to blk_<*> size <*>, PacketResponder <*> for block blk_<*> <*>, Received block blk_<*> of size <*> from /<*>, PacketResponder <*> for block blk_<*> <*>, Received block blk_<*> of size <*> from /<*>, Verification succeeded for blk_<*>, <*>:50010:Got exception while serving blk_<*> to /<*>:, <*>:50010 Served block blk_<*> to /<*>, <*>:50010:Got exception while serving blk_<*> to /<*>:, Verification succeeded for blk_<*>, BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:50010, BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:50010, BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:50010, Deleting block blk_<*> file <*><*>, Deleting block blk_<*> file <*><*>, Deleting block blk_<*> file <*><*> / 1.0000000000000004
# - The average similarity between the provided log sequence and the Train set : 0.9950641021805137
# - The standard deviation of similarity between the provided log sequence and the Train set : 0.005604737907004322
# # Comparison Result for the provided log sequence
# - The most similar log sequence in the Train set (log sequence / similarity): {train_normal_data[max_index]} / {sims[max_index]}
# - The average similarity between the provided log sequence and the Train set : {np.mean(sims)}
# - The standard deviation of similarity between the provided log sequence and the Train set : {np.std(sims)}


# '''  


#         prompt_sim = f'''## Comparison with Example Normal Train Dataset
# We have a Train set consisting of 4,855 normal log sequences.
# First, as an example, we provide the results of comparing the Similarity between one normal data and the log sequences of the train set. 
# Then, we provide the results of comparing the Similarity between the provided test data and the log sequences of the train set. 
# Depending on whether the test data is normal or abnormal, the results of comparing the Similarity may differ, so refer to the results of the two comparisons.
# # Example (Normal Log Sequence)
# Example of normal log sequence : "Receiving block blk_<*> src: <*> dest: /<*>:50010, Receiving block blk_<*> src: <*> dest: /<*>:50010, Receiving block blk_<*> src: <*> dest: /<*>:50010, BLOCK* NameSystem.allocateBlock: <*> blk_<*>, PacketResponder <*> for block blk_<*> <*>, Received block blk_<*> of size <*> from /<*>, PacketResponder <*> for block blk_<*> <*>, Received block blk_<*> of size <*> from /<*>, PacketResponder <*> for block blk_<*> <*>, Received block blk_<*> of size <*> from /<*>, BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:50010 is added to blk_<*> size <*>, BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:50010 is added to blk_<*> size <*>, BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:50010 is added to blk_<*> size <*>, BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:50010, BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:50010, BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:50010, Deleting block blk_<*> file <*><*>, Deleting block blk_<*> file <*><*>, Deleting block blk_<*> file <*><*>"
# - The most similar log sequence in the Train set (log sequence / similarity): Receiving block blk_<*> src: <*> dest: /<*>:50010, Receiving block blk_<*> src: <*> dest: /<*>:50010, Receiving block blk_<*> src: <*> dest: /<*>:50010, BLOCK* NameSystem.allocateBlock: <*> blk_<*>, PacketResponder <*> for block blk_<*> <*>, Received block blk_<*> of size <*> from /<*>, PacketResponder <*> for block blk_<*> <*>, Received block blk_<*> of size <*> from /<*>, PacketResponder <*> for block blk_<*> <*>, Received block blk_<*> of size <*> from /<*>, BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:50010 is added to blk_<*> size <*>, BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:50010 is added to blk_<*> size <*>, BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:50010 is added to blk_<*> size <*>, BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:50010, BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:50010, BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:50010, Deleting block blk_<*> file <*><*>, Deleting block blk_<*> file <*><*>, Deleting block blk_<*> file <*><*> / 1.0
# - The average similarity between the provided log sequence and the Train set : 0.99522
# - The standard deviation of similarity between the provided log sequence and the Train set : 0.00499
# # Comparison Result for the provided log sequence
# - The most similar log sequence in the Train set (log sequence / similarity): {train_normal_data[max_index]} / {round(sims[max_index], 5)}
# - The average similarity between the provided log sequence and the Train set : {round(np.mean(sims), 5)}
# - The standard deviation of similarity between the provided log sequence and the Train set : {round(np.std(sims), 5)}


# '''  

        prompt_sim = f'''## Sequence based Analysis
We have a Train set consisting of 4,855 normal log sequences.
First, as an example, we provide the results of comparing the Similarity between one normal data and the log sequences of the train set. 
Then, we provide the results of comparing the Similarity between the provided test data and the log sequences of the train set. 
Depending on whether the test data is normal or abnormal, the results of comparing the Similarity may differ, so refer to the results of the two comparisons.
# Example (Normal Log Sequence)
- The average similarity between the example normal log sequence and the Train set : {round(np.mean(max_sims), 5)}
- The standard deviation of similarity between the example normal log sequence and the Train set : {round(np.std(max_sims), 5)}
# Comparison Result for the provided log sequence
- The average similarity between the provided log sequence and the Train set : {round(np.mean(sims), 5)}
- The standard deviation of similarity between the provided log sequence and the Train set : {round(np.std(sims), 5)}


'''  



#         prompt_sim = f'''## Comparison with Example Normal Train Dataset
# We have a Train set consisting of 4855 normal log sequences.
# We generated descriptions for log sequences, and the results of comparing the similarity between the provided log sequence's description and the descriptions of log sequences in the Train set are as follows.
# # Example (Normal Log Sequence)
# Example of normal log sequence : Receiving block blk_<*> src: <*> dest: /<*>:50010, Receiving block blk_<*> src: <*> dest: /<*>:50010, BLOCK* NameSystem.allocateBlock: <*> blk_<*>, Receiving block blk_<*> src: <*> dest: /<*>:50010, PacketResponder <*> for block blk_<*> <*>, Received block blk_<*> of size <*> from /<*>, PacketResponder <*> for block blk_<*> <*>, Received block blk_<*> of size <*> from /<*>, PacketResponder <*> for block blk_<*> <*>, Received block blk_<*> of size <*> from /<*>, BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:50010 is added to blk_<*> size <*>, BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:50010 is added to blk_<*> size <*>, BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:50010 is added to blk_<*> size <*>, BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:50010, BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:50010, BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:50010, Deleting block blk_<*> file <*><*>, Deleting block blk_<*> file <*><*>, Deleting block blk_<*> file <*><*>"
# - The description of the provided log seqeuence : This log sequence appears to describe the process of receiving and storing blocks in a Hadoop Distributed File System (HDFS). The sequence is characterized by a series of repeated events, including:\n\n1. Receiving blocks from a source (src) and storing them at a destination (dest) on the HDFS.\n2. Allocating blocks in the NameSystem and updating the blockMap.\n3. Adding blocks to the blockMap and updating the block sizes.\n4. Deleting blocks from the blockMap and marking them as invalid.\n5. Deleting files associated with the blocks.\n\nThe sequence suggests that the HDFS is receiving a large number of blocks from a source, storing them, and then deleting them. The repeated events of receiving, allocating, adding, and deleting blocks indicate a continuous process of block management in the HDFS.\n\nThe sequence also implies that the HDFS is handling a large volume of data, as evidenced by the repeated events and the mention of block sizes. The sequence may be related to a Hadoop MapReduce job, where the HDFS is used to store and manage intermediate data during the processing of large datasets.\n\nOverall, the log sequence provides insight into the internal workings of the HDFS, highlighting its ability to manage large volumes of data and handle repeated events in a efficient manner.
# - The most similar log sequence in the Train set : Receiving block blk_<*> src: <*> dest: /<*>:50010, Receiving block blk_<*> src: <*> dest: /<*>:50010, BLOCK* NameSystem.allocateBlock: <*> blk_<*>, Receiving block blk_<*> src: <*> dest: /<*>:50010, Received block blk_<*> of size <*> from /<*>, PacketResponder <*> for block blk_<*> <*>, PacketResponder <*> for block blk_<*> <*>, Received block blk_<*> of size <*> from /<*>, PacketResponder <*> for block blk_<*> <*>, Received block blk_<*> of size <*> from /<*>, BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:50010 is added to blk_<*> size <*>, BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:50010 is added to blk_<*> size <*>, BLOCK* NameSystem.addStoredBlock: blockMap updated: <*>:50010 is added to blk_<*> size <*>, BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:50010, BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:50010, BLOCK* NameSystem.delete: blk_<*> is added to invalidSet of <*>:50010, Deleting block blk_<*> file <*><*>, Deleting block blk_<*> file <*><*>, Deleting block blk_<*> file <*><*> / 0.9848291763253438
# - The average similarity between the provided log sequence and the Train set : 0.9530413856029833
# - The standard deviation of similarity between the provided log sequence and the Train set : 0.017381068368228565
# # Comparison Result for the provided log sequence
# - The description of the provided log sequence : {description}
# - The most similar log sequence in the Train set (log sequence / similarity) : {train_normal_data[max_index]} / {sims[max_index]}
# - The average similarity between the provided log sequence and the Train set : {np.mean(sims)}
# - The standard deviation of similarity between the provided log sequence and the Train set : {np.std(sims)}


# '''     

#         prompt_sim = f'''## Few-shot Example
# Refer to the following example of a normal log sequence:
# # Example : [{train_normal_data[np.argsort(sims)[0]]}, {train_normal_data[np.argsort(sims)[1]]}, {train_normal_data[np.argsort(sims)[1]]}

# '''
    else : prompt_sim = ""
    

    if seq_info == True :
#         info_prompt = f'''## Instruction : 
# The Log Sequence is from HDFS Log Dataset. Please provide 3~5 key observations or features that can be obtained by looking at the entire flow rather than the information of each log in the log sequence.
# log sequence : {input_data["text"]}

# ## Domain Knowledge
# - Dataset Explanation : HDFS(Hadoop Distributed File System) log dataset contains system logs generated by Hadoop clusters, capturing various events like file operations, errors, and access details. These logs help in monitoring, debugging, and analyzing system performance and failures. They are widely used in anomaly detection, fault diagnosis, and predictive maintenance in big data environments.

# ## Response Format
# {{"sequence_wise_info" : 3~5 key observations}}'''

#         sequence_info = ask_llm(info_prompt)
#         # print(sequence_info)
#         prompt_seq = f'''# {sequence_info}
# '''
        # emb = get_embedding(input_data["text"], tokenizer, model)
        # emb = model.encode(log_sequence_, normalize_embeddings=True)
        sims = cosine_similarity(np.vstack([emb]), np.vstack(train_normal_data_embds))[0]
        prompt_seq = f'''## Few-shot Example
Refer to the following example of a normal log sequence
# Examples : [{str(train_normal_data[np.argsort(sims)[-1]])} / {str(train_normal_data[np.argsort(sims)[-2]])} / {str(train_normal_data[np.argsort(sims)[-3]])} / {str(train_normal_data[np.argsort(sims)[-4]])} / {str(train_normal_data[np.argsort(sims)[-5]])}]
'''
    else :
        prompt_seq = ""
    if cot_prompt == True : 
#         prompt_cot = f'''## Chain of Thought Prompt
# - Step 1 : Pay attention to the False Positive Rule and compare the Log Sequence and Normal Vocab to detect what meaning it has.
# - Step 2 : Compare the information comparing the Normal Train Data and the log sequence with the example, and note that even if the similarity is high, there is still a possibility that it is Conditional Abnormal.
# - Step 3 : Based on the information provided above and the information about abnormal cases that may occur in the HDFS Dataset you have, determine whether the log sequence is normal or abnormal and explain the reason.

# '''
        prompt_cot = f'''# Following the Steps
- Step 1 : Compare the Log Sequence and Normal Vocab to detect what meaning it has. 
- Step 2 : Compare the information comparing the Normal Train Data and the log sequence with the example, and note that even if the similarity is high, there is still a possibility that it is Conditional Abnormal.
- Step 3 : Based on the information provided above and the information about abnormal cases that may occur in the HDFS Dataset you have, determine whether the log sequence is normal or abnormal and explain the reason.

'''
    else : prompt_cot = ""

#     confidence_prompt = f'''## Classification Confidence
# "classification_confidence" means the confidence in the classification you generated.
# Due to the feature of log sequences, the classification_confidence of a sequence predicted as normal should be 95 or higher, and the classification_confidence of a sequence predicted as abnormal should be 80 or higher for a reliable prediction result.
# Assign a “classification_confidence” to the prediction result you generated and write a “confidence_explanation” that explains why the classification_confidence was deducted. 

# '''
    confidence_prompt = f''''''
    
    prompt_response = f'''## Response Format
Be sure to follow the Json response format below. No other explanations. Only output the Json format.
{{"classification" : normal/abnormal, "brief_explanation" : brief explanation of reason for judgment}}
'''
#     prompt_response = f'''## Response Format
# Be sure to follow the Json response format below.
# {{"classification" : normal/abnomal, "brief_explanation" : brief explanation of reason for judgment, "classification_confidence" : 0~100, "confidence_explanation" : brief explanation of reasons for the score reduction}}
# '''


    # if prompt_vocab == False and prompt_seq == False and prompt_cot == False and prompt_sim == False :
    #     prompt = prompt_base
    # else : 
    #     prompt = prompt_base + prompt_dataset + prompt_vocab + prompt_seq + prompt_cot + prompt_sim + prompt_response
    # prompt = {"prompt_base" : prompt_base, "prompt_dataset" : prompt_dataset, "prompt_vocab" : prompt_vocab, "prompt_seq" : prompt_seq, "prompt_cot" : prompt_cot, "prompt_sim" : prompt_sim, "prompt_response" : prompt_response}
    prompt = {"prompt_base" : prompt_base, "prompt_dataset" : prompt_dataset, "prompt_vocab" : prompt_vocab, "prompt_seq" : prompt_seq, "prompt_cot" : prompt_cot, "prompt_sim" : prompt_sim, "prompt_response" : prompt_response}

    return prompt


def bgl_prompt(input_data, train_normal_data, train_normal_data_embds, device, tokenizer, model, kmeans, cluster_info, compare_nv, seq_info, compare_sim, cot_prompt, vocab_desc_file) :
    log_sequence = str(input_data["text"]).split("', ")
    log_sequence = [i.replace("'","").strip() for i in log_sequence]
    log_sequence_ = ["'"+ i+"'" for i in log_sequence]
    log_sequence_ = ", ".join(log_sequence_)
    
    prompt_base = f'''## Instruction : 
Determine whether the log sequence is normal or abnormal by synthesizing the additional information given below.
log sequence : {log_sequence_}


'''
    prompt_dataset = '''## Domain Knowledge
# Dataset Explanation
BGL is an open dataset of logs collected from a BlueGene/L supercomputer system at Lawrence Livermore National Labs (LLNL) in Livermore, California, with 131,072 processors and 32,768GB memory.
It has been used in several studies on log parsing, anomaly detection, and failure prediction.
'''        
    if compare_nv == True : 
        with open(f"./preprocessed_dataset/bgl/train_vocab.json", "r") as f :
            vocab = json.load(f)
        log_sequence = str(input_data["text"]).split("', ")
        log_sequence = [i.replace("'","").strip() for i in log_sequence]
        checklist = []
        for log in log_sequence :
            if log not in vocab :
                checklist.append(log)
        checklist = list(set(checklist))
#         if len(checklist) > 0 : 
#             vocab_info = f'''- Comparison Explanation: The log sequence is likely to contain the log of the Unseen Normal Vocab or the log of the Abnormal Vocab.
# - Unseen Normal Vocab or the log of the Abnormal Vocab in the provided log sequence: {checklist}'''
#         else :
#             vocab_info = f"- Comparison Explanation: The log sequence is likely to be normal because it does not contain the log of abnormal vocab, but it may be abnormal under certain conditions."
        prompt_vocab = f'''## Entry based Analysis
Normal Log Entry refers to the log templates that appear in the Normal Train Log Sequence.
If a log that is not in the normal log entry appears in the provided Log sequence, it provides what the log is.
# Comparison Results
- Unseen log from provided log sequence: {checklist}
'''


#         with open(f"./preprocessed_dataset/hdfs/base_train_normal.json", "r") as f :
#             train_data = json.load(f)
#         train_vocabs = []
#         for i in train_data :
#             temp = i["text"].split(", ")
#             for j in temp :
#                 train_vocabs.append(j)
#         from collections import Counter
#         train_vocabs = Counter(train_vocabs)
#         log_sequence = str(input_data["text"]).split(", ")
#         vocab_info = []
#         for i in log_sequence :
#             vocab_info.append(f"{i} : {train_vocabs[i]}")
#         # print(vocab_info)
#         prompt_vocab = f'''# Comparison with Normal Vocab
# This shows how many times each log in the provided log sequence appears in the log sequences of the training data. 
# If there are logs in the provided log sequence that never appear in the training data, you should examine them in detail.
# # Comparison Results
# {vocab_info}
# '''
            
        
    
    else : prompt_vocab = ""
    if compare_sim == True or seq_info == True : 
        emb = model.encode(log_sequence_, normalize_embeddings=True)

    if compare_sim == True :
        # emb = get_embedding(input_data["text"], tokenizer, model)
        # emb = model.encode(log_sequence_, normalize_embeddings=True)

        # sims = cosine_similarity(np.vstack([emb]), np.vstack(train_normal_data_embds))[0]

        # centroid = np.vstack(train_normal_data_embds).mean(axis=0)
        # max_sims = cosine_similarity(np.vstack([centroid]), np.vstack(train_normal_data_embds))[0]

        # max_index = np.argmax(sims)
        # max_emb = train_normal_data_embds[max_index]
        # max_sims = cosine_similarity(np.vstack([max_emb]), np.vstack(train_normal_data_embds))[0]

        
        cluster_class = kmeans.predict([emb])
        # centroid = np.vstack(cluster_info[cluster_class[0]]).mean(axis=0)
        centroid = np.vstack(train_normal_data_embds).mean(axis=0)

        # sims = cosine_similarity(np.vstack([emb]), np.vstack(cluster_info[cluster_class[0]]))[0]
        # max_sims = cosine_similarity(np.vstack([centroid]), np.vstack(cluster_info[cluster_class[0]]))[0]
        sims = cosine_similarity(np.vstack([emb]), np.vstack(train_normal_data_embds))[0]
        max_sims = cosine_similarity(np.vstack([centroid]), np.vstack(train_normal_data_embds))[0]
        

        prompt_sim = f'''## Sequence based Analysis
We have a Train set consisting of 13,718 normal log sequences.
First, as an example, we provide the results of comparing the Similarity between one normal data and the log sequences of the train set. 
Then, we provide the results of comparing the Similarity between the provided test data and the log sequences of the train set. 
Depending on whether the test data is normal or abnormal, the results of comparing the Similarity may differ, so refer to the results of the two comparisons.
# Example (Normal Log Sequence)
- The average similarity between the example normal log sequence and the Train set : {round(np.mean(max_sims), 5)}
- The standard deviation of similarity between the example normal log sequence and the Train set : {round(np.std(max_sims), 5)}
# Comparison Result for the provided log sequence
- The average similarity between the provided log sequence and the Train set : {round(np.mean(sims), 5)}
- The standard deviation of similarity between the provided log sequence and the Train set : {round(np.std(sims), 5)}


'''  

    else : prompt_sim = ""
    

    if seq_info == True :
#         info_prompt = f'''## Instruction : 
# The Log Sequence is from HDFS Log Dataset. Please provide 3~5 key observations or features that can be obtained by looking at the entire flow rather than the information of each log in the log sequence.
# log sequence : {input_data["text"]}

# ## Domain Knowledge
# - Dataset Explanation : HDFS(Hadoop Distributed File System) log dataset contains system logs generated by Hadoop clusters, capturing various events like file operations, errors, and access details. These logs help in monitoring, debugging, and analyzing system performance and failures. They are widely used in anomaly detection, fault diagnosis, and predictive maintenance in big data environments.

# ## Response Format
# {{"sequence_wise_info" : 3~5 key observations}}'''

#         sequence_info = ask_llm(info_prompt)
#         # print(sequence_info)
#         prompt_seq = f'''# {sequence_info}
# '''
        # emb = get_embedding(input_data["text"], tokenizer, model)
        # emb = model.encode(log_sequence_, normalize_embeddings=True)
        sims = cosine_similarity(np.vstack([emb]), np.vstack(train_normal_data_embds))[0]
        prompt_seq = f'''## Few-shot Example
Refer to the following example of a normal log sequence
# Examples : [{train_normal_data[np.argsort(sims)[-1]]}]
'''
    else :
        prompt_seq = ""
    if cot_prompt == True : 
#         prompt_cot = f'''## Chain of Thought Prompt
# - Step 1 : Pay attention to the False Positive Rule and compare the Log Sequence and Normal Vocab to detect what meaning it has.
# - Step 2 : Compare the information comparing the Normal Train Data and the log sequence with the example, and note that even if the similarity is high, there is still a possibility that it is Conditional Abnormal.
# - Step 3 : Based on the information provided above and the information about abnormal cases that may occur in the HDFS Dataset you have, determine whether the log sequence is normal or abnormal and explain the reason.

# '''
        prompt_cot = f'''# Following the Steps
- Step 1 : Compare the Log Sequence and Unseen Log Entry to detect what meaning it has. Due to the nature of the dataset, the presence of unseen logs strongly indicates an abnormal event, while the absence of unseen logs strongly indicates a normal event.
- Step 2 : Compare the information comparing the Normal Train Data and the log sequence with the example, and note that even if the similarity is high, there is still a possibility that it is Conditional Abnormal.
- Step 3 : Based on the information provided above and the information about abnormal cases that may occur in the BGL Dataset you have, determine whether the log sequence is normal or abnormal and explain the reason.

'''
    else : prompt_cot = ""

    
    prompt_response = f'''## Response Format
Be sure to follow the Json response format below.
{{"classification" : normal/abnormal, "brief_explanation" : brief explanation of reason for judgment}}
'''
#     prompt_response = f'''## Response Format
# Be sure to follow the Json response format below.
# {{"classification" : normal/abnomal, "brief_explanation" : brief explanation of reason for judgment, "classification_confidence" : 0~100, "confidence_explanation" : brief explanation of reasons for the score reduction}}
# '''


    # if prompt_vocab == False and prompt_seq == False and prompt_cot == False and prompt_sim == False :
    #     prompt = prompt_base
    # else : 
    #     prompt = prompt_base + prompt_dataset + prompt_vocab + prompt_seq + prompt_cot + prompt_sim + prompt_response
    # prompt = {"prompt_base" : prompt_base, "prompt_dataset" : prompt_dataset, "prompt_vocab" : prompt_vocab, "prompt_seq" : prompt_seq, "prompt_cot" : prompt_cot, "prompt_sim" : prompt_sim, "prompt_response" : prompt_response}
    prompt = {"prompt_base" : prompt_base, "prompt_dataset" : prompt_dataset, "prompt_vocab" : prompt_vocab, "prompt_seq" : prompt_seq, "prompt_cot" : prompt_cot, "prompt_sim" : prompt_sim, "prompt_response" : prompt_response}

    return prompt


def tbird_prompt(input_data, train_normal_data, train_normal_data_embds, device, tokenizer, model, kmeans, cluster_info, compare_nv, seq_info, compare_sim, cot_prompt, vocab_desc_file) :
    log_sequence = str(input_data["text"]).split("', ")
    log_sequence = [i.replace("'","").strip() for i in log_sequence]
    log_sequence_ = ["'"+ i+"'" for i in log_sequence]
    log_sequence_ = ", ".join(log_sequence_)

    prompt_base = f'''## Instruction : 
Determine whether the log sequence is normal or abnormal by synthesizing the additional information given below.
log sequence : {log_sequence_}


'''
    prompt_dataset = '''## Domain Knowledge
# Dataset Explanation
Thunderbird is an open dataset of logs collected from a Thunderbird supercomputer system at Sandia National Labs (SNL) in Albuquerque, with 9,024 processors and 27,072GB memory.
'''        
    if compare_nv == True : 
        with open(f"./preprocessed_dataset/tbird/train_vocab.json", "r") as f :
            vocab = json.load(f)
        log_sequence = str(input_data["text"]).split("', ")
        log_sequence = [i.replace("'","").strip() for i in log_sequence]
        checklist = []
        for log in log_sequence :
            if log not in vocab :
                checklist.append(log)
        checklist = list(set(checklist))  
#         if len(checklist) > 0 : 
#             vocab_info = f'''- Comparison Explanation: The log sequence is likely to contain the log of the Unseen Normal Vocab or the log of the Abnormal Vocab.
# - Unseen Normal Vocab or the log of the Abnormal Vocab in the provided log sequence: {checklist}'''
#         else :
#             vocab_info = f"- Comparison Explanation: The log sequence is likely to be normal because it does not contain the log of abnormal vocab, but it may be abnormal under certain conditions."
        prompt_vocab = f'''## Entry based Analysis
Normal Log Entry refers to the log templates that appear in the Normal Train Log Sequence.
If a log that is not in the normal log entry appears in the provided Log sequence, it provides what the log is.
# Comparison Results
- Unseen log from provided log sequence: {checklist}
'''


#         with open(f"./preprocessed_dataset/hdfs/base_train_normal.json", "r") as f :
#             train_data = json.load(f)
#         train_vocabs = []
#         for i in train_data :
#             temp = i["text"].split(", ")
#             for j in temp :
#                 train_vocabs.append(j)
#         from collections import Counter
#         train_vocabs = Counter(train_vocabs)
#         log_sequence = str(input_data["text"]).split(", ")
#         vocab_info = []
#         for i in log_sequence :
#             vocab_info.append(f"{i} : {train_vocabs[i]}")
#         # print(vocab_info)
#         prompt_vocab = f'''# Comparison with Normal Vocab
# This shows how many times each log in the provided log sequence appears in the log sequences of the training data. 
# If there are logs in the provided log sequence that never appear in the training data, you should examine them in detail.
# # Comparison Results
# {vocab_info}
# '''
            
        
    
    else : prompt_vocab = ""
    
    if compare_sim == True or seq_info == True :
        emb = model.encode(log_sequence_, normalize_embeddings=True)
    if compare_sim == True :
        # emb = get_embedding(input_data["text"], tokenizer, model)
        # emb = model.encode(log_sequence_, normalize_embeddings=True)

        # sims = cosine_similarity(np.vstack([emb]), np.vstack(train_normal_data_embds))[0]

        # centroid = np.vstack(train_normal_data_embds).mean(axis=0)
        # max_sims = cosine_similarity(np.vstack([centroid]), np.vstack(train_normal_data_embds))[0]

        # max_index = np.argmax(sims)
        # max_emb = train_normal_data_embds[max_index]
        # max_sims = cosine_similarity(np.vstack([max_emb]), np.vstack(train_normal_data_embds))[0]

        
        cluster_class = kmeans.predict([emb])
        # centroid = np.vstack(cluster_info[cluster_class[0]]).mean(axis=0)
        centroid = np.vstack(train_normal_data_embds).mean(axis=0)

        # sims = cosine_similarity(np.vstack([emb]), np.vstack(cluster_info[cluster_class[0]]))[0]
        # max_sims = cosine_similarity(np.vstack([centroid]), np.vstack(cluster_info[cluster_class[0]]))[0]
        sims = cosine_similarity(np.vstack([emb]), np.vstack(train_normal_data_embds))[0]
        max_sims = cosine_similarity(np.vstack([centroid]), np.vstack(train_normal_data_embds))[0]
        

        prompt_sim = f'''## Sequence based Analysis
We have a Train set consisting of 6,000 normal log sequences.
First, as an example, we provide the results of comparing the Similarity between one normal data and the log sequences of the train set. 
Then, we provide the results of comparing the Similarity between the provided test data and the log sequences of the train set. 
Depending on whether the test data is normal or abnormal, the results of comparing the Similarity may differ, so refer to the results of the two comparisons.
# Example (Normal Log Sequence)
- The average similarity between the example normal log sequence and the Train set : {round(np.mean(max_sims), 5)}
- The standard deviation of similarity between the example normal log sequence and the Train set : {round(np.std(max_sims), 5)}
# Comparison Result for the provided log sequence
- The average similarity between the provided log sequence and the Train set : {round(np.mean(sims), 5)}
- The standard deviation of similarity between the provided log sequence and the Train set : {round(np.std(sims), 5)}


'''  

    else : prompt_sim = ""
    

    if seq_info == True :
#         info_prompt = f'''## Instruction : 
# The Log Sequence is from HDFS Log Dataset. Please provide 3~5 key observations or features that can be obtained by looking at the entire flow rather than the information of each log in the log sequence.
# log sequence : {input_data["text"]}

# ## Domain Knowledge
# - Dataset Explanation : HDFS(Hadoop Distributed File System) log dataset contains system logs generated by Hadoop clusters, capturing various events like file operations, errors, and access details. These logs help in monitoring, debugging, and analyzing system performance and failures. They are widely used in anomaly detection, fault diagnosis, and predictive maintenance in big data environments.

# ## Response Format
# {{"sequence_wise_info" : 3~5 key observations}}'''

#         sequence_info = ask_llm(info_prompt)
#         # print(sequence_info)
#         prompt_seq = f'''# {sequence_info}
# '''
        # emb = get_embedding(input_data["text"], tokenizer, model)
        # emb = model.encode(log_sequence_, normalize_embeddings=True)
        sims = cosine_similarity(np.vstack([emb]), np.vstack(train_normal_data_embds))[0]
        prompt_seq = f= f'''## Few-shot Example
Refer to the following example of a normal log sequence
# Examples : [{train_normal_data[np.argsort(sims)[-1]]}]
'''
    else :
        prompt_seq = ""
    if cot_prompt == True : 
#         prompt_cot = f'''## Chain of Thought Prompt
# - Step 1 : Pay attention to the False Positive Rule and compare the Log Sequence and Normal Vocab to detect what meaning it has.
# - Step 2 : Compare the information comparing the Normal Train Data and the log sequence with the example, and note that even if the similarity is high, there is still a possibility that it is Conditional Abnormal.
# - Step 3 : Based on the information provided above and the information about abnormal cases that may occur in the HDFS Dataset you have, determine whether the log sequence is normal or abnormal and explain the reason.

# '''
        prompt_cot = f'''# Following the Steps
- Step 1 : Compare the Log Sequence and Unseen Log Entry to detect what meaning it has. Due to the nature of the dataset, the presence of unseen logs strongly indicates an abnormal event, while the absence of unseen logs strongly indicates a normal event.
- Step 2 : Compare the information comparing the Normal Train Data and the log sequence with the example, and note that even if the similarity is high, there is still a possibility that it is Conditional Abnormal.
- Step 3 : Based on the information provided above and the information about abnormal cases that may occur in the Thunderbird Dataset you have, determine whether the log sequence is normal or abnormal and explain the reason.

'''
    else : prompt_cot = ""

    
    prompt_response = f'''## Response Format
Be sure to follow the Json response format below.
{{"classification" : normal/abnormal, "brief_explanation" : brief explanation of reason for judgment}}
'''
#     prompt_response = f'''## Response Format
# Be sure to follow the Json response format below.
# {{"classification" : normal/abnomal, "brief_explanation" : brief explanation of reason for judgment, "classification_confidence" : 0~100, "confidence_explanation" : brief explanation of reasons for the score reduction}}
# '''


    # if prompt_vocab == False and prompt_seq == False and prompt_cot == False and prompt_sim == False :
    #     prompt = prompt_base
    # else : 
    #     prompt = prompt_base + prompt_dataset + prompt_vocab + prompt_seq + prompt_cot + prompt_sim + prompt_response
    # prompt = {"prompt_base" : prompt_base, "prompt_dataset" : prompt_dataset, "prompt_vocab" : prompt_vocab, "prompt_seq" : prompt_seq, "prompt_cot" : prompt_cot, "prompt_sim" : prompt_sim, "prompt_response" : prompt_response}
    prompt = {"prompt_base" : prompt_base, "prompt_dataset" : prompt_dataset, "prompt_vocab" : prompt_vocab, "prompt_seq" : prompt_seq, "prompt_cot" : prompt_cot, "prompt_sim" : prompt_sim, "prompt_response" : prompt_response}

    return prompt

