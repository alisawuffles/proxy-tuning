# Evaluating DExperts
temperature=0.8
k=10
size=13
echo "Results dir: results/codex_humaneval/dexperts-${size}B-t$temperature"
python -m eval.codex_humaneval.run_eval \
    --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --expert_model_name_or_path codellama/CodeLlama-7b-Python-hf \
    --save_dir results/codex_humaneval/dexperts-${size}B-t$temperature \
    --eval_pass_at_ks $k \
    --unbiased_sampling_size_n 20 \
    --temperature $temperature \
    --eval_batch_size 32


# Evaluating Llama 2
temperature=0.8
k=10
size=13
echo "Results dir: results/codex_humaneval/llama2-${size}B-t$temperature"
python -m eval.codex_humaneval.run_eval \
    --model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --save_dir results/codex_humaneval/llama2-${size}B-t$temperature \
    --eval_pass_at_ks $k \
    --unbiased_sampling_size_n 20 \
    --temperature $temperature \
    --eval_batch_size 8


# Evaluating Code Llama
temperature=0.8
k=10
size=13
echo "Results dir: results/codex_humaneval/codellama-${size}B-Python-t$temperature"
python -m eval.codex_humaneval.run_eval \
    --model_name_or_path codellama/CodeLlama-${size}b-Python-hf \
    --save_dir results/codex_humaneval/codellama-${size}B-t$temperature \
    --eval_pass_at_ks $k \
    --unbiased_sampling_size_n 20 \
    --temperature $temperature \
    --eval_batch_size 8
