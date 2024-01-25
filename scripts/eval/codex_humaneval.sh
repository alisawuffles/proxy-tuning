# Evaluating DExperts
temperature=0.8
k=10
size=13
python -m eval.codex_humaneval.run_eval \
    --save_dir results/codex_humaneval/dexperts-${size}B-t$temperature \
    --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --expert_model_name_or_path codellama/CodeLlama-7b-Python-hf \
    --eval_pass_at_ks $k \
    --unbiased_sampling_size_n 20 \
    --temperature $temperature \
    --eval_batch_size 32

# Evaluating Llama 2
temperature=0.8
k=10
size=13
python -m eval.codex_humaneval.run_eval \
    --save_dir results/codex_humaneval/llama2-${size}B-t$temperature \
    --model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --eval_pass_at_ks $k \
    --unbiased_sampling_size_n 20 \
    --temperature $temperature \
    --eval_batch_size 8

# Evaluating CodeLlama
temperature=0.8
k=10
size=13
python -m eval.codex_humaneval.run_eval \
    --save_dir results/codex_humaneval/codellama-${size}B-Python-t$temperature \
    --model_name_or_path codellama/CodeLlama-${size}b-Python-hf \
    --eval_pass_at_ks $k \
    --unbiased_sampling_size_n 20 \
    --temperature $temperature \
    --eval_batch_size 8
