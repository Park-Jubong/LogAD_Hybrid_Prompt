# Pre-processing
python main.py --mode preprocess --dataset hdfs
# python main.py --mode preprocess --dataset bgl
# python main.py --mode preprocess --dataset tbird

# python main.py --mode inference --dataset bgl --num_samples 100 --compare_nv --cot_prompt --compare_sim --seq_info
# python main.py --mode confusion_matrix --dataset bgl --num_samples 100 --compare_nv --cot_prompt --compare_sim --seq_info

python main.py --mode inference --dataset hdfs --num_samples 100 --compare_nv --cot_prompt --compare_sim --seq_info
# python main.py --mode confusion_matrix --dataset hdfs --num_samples 1000 --compare_nv --cot_prompt --compare_sim --seq_info --llm_batch

# python main.py --mode inference --dataset bgl --num_samples 1000 --compare_nv --cot_prompt --compare_sim --seq_info --llm_batch
# python main.py --mode confusion_matrix --dataset bgl --num_samples 1000 --compare_nv --cot_prompt --compare_sim --seq_info --llm_batch

# python main.py --mode inference --dataset tbird --num_samples 1000 --compare_nv --cot_prompt --compare_sim --seq_info --llm_batch
# python main.py --mode confusion_matrix --dataset tbird --num_samples 1000 --compare_nv --cot_prompt --compare_sim --seq_info --llm_batch

# Ablation Studies
# Only Instruction
# python main.py --mode inference --dataset hdfs --num_samples 1000 --llm_batch
# python main.py --mode confusion_matrix --dataset hdfs --num_samples 1000 --llm_batch

# python main.py --mode inference --dataset hdfs --num_samples 1000 --compare_nv --llm_batch
# python main.py --mode confusion_matrix --dataset hdfs --num_samples 1000 --compare_nv --llm_batch

# python main.py --mode inference --dataset hdfs --num_samples 1000 --compare_sim --llm_batch
# python main.py --mode confusion_matrix --dataset hdfs --num_samples 1000 --compare_sim --llm_batch

# python main.py --mode inference --dataset hdfs --num_samples 1000 --seq_info --llm_batch
# python main.py --mode confusion_matrix --dataset hdfs --num_samples 1000 --seq_info --llm_batch