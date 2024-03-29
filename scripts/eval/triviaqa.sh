# Evaluating DExperts
size=13
python -m eval.triviaqa.run_eval \
    --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --expert_model_name_or_path models/llama2-triviaqa-7b \
    --save_dir results/triviaqa/dexperts-triviaqa-${size}B \
    --tokenizer_name_or_path meta-llama/Llama-2-${size}b-hf \
    --eval_batch_size 32


# Evaluating Llama 2
size=13
python -m eval.triviaqa.run_eval \
    --model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --save_dir results/triviaqa/llama2-${size}B \
    --eval_batch_size 32


# Evaluating Llama 2 chat
size=13
python -m eval.triviaqa.run_eval \
    --model_name_or_path models/Llama-2-${size}b-chat-hf \
    --save_dir results/triviaqa/llama2-chat-${size}B \
    --eval_batch_size 32


# Evaluating expert
size=13
python -m eval.triviaqa.run_eval \
    --model_name_or_path models/llama2-triviaqa-${size}b \
    --save_dir results/triviaqa/llama2-triviaqa-${size}B \
    --eval_batch_size 32


# Evaluating model tuned on TriviaQA with LORA
size=13
python -m eval.triviaqa.run_eval \
    --model_name_or_path models/llama2-triviaqa-${size}b-lora-merged \
    --save_dir results/triviaqa/llama2-triviaqa-lora-${size}B \
    --eval_batch_size 32