# Evaluating DExperts
size=13
python -m eval.triviaqa.run_eval \
    --data_dir data/eval/triviaqa \
    --save_dir results/triviaqa/dexperts-triviaqa-${size}B \
    --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --expert_model_name_or_path models/llama2-triviaqa-7b \
    --tokenizer_name_or_path meta-llama/Llama-2-${size}b-hf \
    --eval_batch_size 32

# Evaluating Llama 2
size=13
python -m eval.triviaqa.run_eval \
    --data_dir data/eval/triviaqa \
    --save_dir results/triviaqa/llama2-${size}B \
    --model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --eval_batch_size 32

# Evaluating expert
size=13
python -m eval.triviaqa.run_eval \
    --data_dir data/eval/triviaqa \
    --save_dir results/triviaqa/llama2-triviaqa-${size}B \
    --model_name_or_path models/llama2-triviaqa-${size}b \
    --eval_batch_size 32
