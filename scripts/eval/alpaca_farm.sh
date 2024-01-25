# evaluating DExperts
size=13
python -m eval.alpaca_farm.run_eval \
    --save_dir results/alpaca_farm/dexperts-${size}B \
    --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --expert_model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --eval_batch_size 8

# evaluating Llama 2
size=13
python -m eval.alpaca_farm.run_eval \
    --save_dir results/alpaca_farm/llama2-${size}B \
    --model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --eval_batch_size 4

# evaluating Llama 2 chat
size=13
python -m eval.alpaca_farm.run_eval \
    --save_dir results/alpaca_farm/llama2-${size}B-chat \
    --model_name_or_path meta-llama/Llama-2-${size}b-chat-hf \
    --eval_batch_size 8 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format
