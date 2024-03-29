# Evaluating DExperts with chat expert
size=13
echo "Results dir: results/gsm/dexperts-${size}B"
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --save_dir results/gsm/dexperts-${size}B \
    --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --expert_model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --eval_batch_size 20


# Evaluating DExperts with GSM expert
size=13
echo "Results dir: results/gsm/dexperts-gsm-${size}B"
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --save_dir results/gsm/dexperts-gsm-${size}B \
    --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --expert_model_name_or_path models/llama2-gsm-7b \
    --eval_batch_size 20


# Evaluating Llama 2
size=13
echo "Results dir: results/gsm/llama2-${size}B"
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --save_dir results/gsm/llama2-${size}B \
    --model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --eval_batch_size 20


# Evaluating Llama 2 chat
size=13
echo "Results dir: results/gsm/llama2-chat-${size}B"
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --save_dir results/gsm/llama2-chat-${size}B \
    --model_name_or_path meta-llama/Llama-2-${size}b-chat-hf \
    --eval_batch_size 20 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format


# Evaluating GSM expert
size=7
echo "Results dir: results/gsm/llama2-gsm-${size}B"
python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm/ \
    --save_dir results/gsm/llama2-gsm-${size}B \
    --model_name_or_path models/llama2-gsm-${size}b \
    --eval_batch_size 20
