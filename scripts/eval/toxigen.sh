# Evaluating DExperts
size=13
python -m eval.toxigen.run_eval \
    --data_dir data/eval/toxigen/ \
    --save_dir results/toxigen/dexperts-${size}B \
    --eval_batch_size 32 \
    --max_examples_per_group 200 \
    --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --expert_model_name_or_path meta-llama/Llama-2-7b-chat-hf

# Evaluating Llama 2
size=13
python -m eval.toxigen.run_eval \
    --data_dir data/eval/toxigen/ \
    --save_dir results/toxigen/llama2-${size}B \
    --eval_batch_size 32 \
    --max_examples_per_group 200 \
    --model_name_or_path meta-llama/Llama-2-${size}b-hf

# Evaluating Llama 2 chat
size=13
python -m eval.toxigen.run_eval \
    --data_dir data/eval/toxigen/ \
    --save_dir results/toxigen/llama2-chat-${size}B \
    --model_name_or_path meta-llama/Llama-2-${size}b-chat-hf \
    --use_chat_format \
    --eval_batch_size 32 \
    --max_examples_per_group 200 \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format
