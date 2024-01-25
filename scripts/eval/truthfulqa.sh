# Evaluating DExperts
size=13
python -m eval.truthfulqa.run_eval \
    --data_dir data/eval/truthfulqa \
    --save_dir results/truthfulqa/dexperts-${size}B \
    --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --expert_model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --settings open mc \
    --eval_batch_size 20 \
    --system_prompt "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information." \
    --gpt_true_model_name curie:ft-allennlp:gpt-judge-2023-07-26-09-37-48 \
    --gpt_info_model_name curie:ft-allennlp:gpt-info-2023-07-26-11-38-18


# Evaluating Llama 2 model
size=13
python -m eval.truthfulqa.run_eval \
    --data_dir data/eval/truthfulqa \
    --save_dir results/truthfulqa/llama2-${size}B \
    --model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --settings open mc \
    --gpt_true_model_name curie:ft-allennlp:gpt-judge-2023-07-26-09-37-48 \
    --gpt_info_model_name curie:ft-allennlp:gpt-info-2023-07-26-11-38-18 \
    --eval_batch_size 20


# Evaluating Llama 2 chat model
size=13
python -m eval.truthfulqa.run_eval \
    --data_dir data/eval/truthfulqa \
    --save_dir results/truthfulqa/llama2-chat-${size}B \
    --model_name_or_path meta-llama/Llama-2-${size}b-chat-hf \
    --settings open mc \
    --gpt_true_model_name curie:ft-allennlp:gpt-judge-2023-07-26-09-37-48 \
    --gpt_info_model_name curie:ft-allennlp:gpt-info-2023-07-26-11-38-18 \
    --system_prompt "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information." \
    --eval_batch_size 20 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format
