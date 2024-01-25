# Evaluating DExperts
temperature=0.8
size=13
echo "Results dir: results/ds1000/dexperts-${size}B-t$temperature"
python -m eval.ds1000.run_inference \
    --save_dir results/ds1000/dexperts-${size}B-t$temperature \
    --base_model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --expert_model_name_or_path codellama/CodeLlama-7b-Python-hf \
    --temperature $temperature \
    --eval_batch_size 8 \
    --max_examples 200 \
    --unbiased_sampling_size_n 20


# Evaluating Llama
size=13
temperature=0.8
python -m eval.ds1000.run_inference \
--save_dir results/ds1000/llama2-${size}B-t$temperature \
    --model_name_or_path meta-llama/Llama-2-${size}b-hf \
    --temperature $temperature \
    --eval_batch_size 8 \
    --max_examples 200 \
    --unbiased_sampling_size_n 20


# Evaluating CodeLlama
size=13
temperature=0.8
echo "Results dir: results/ds1000/codellama2-${size}B-t$temperature"
python -m eval.ds1000.run_inference \
    --save_dir results/ds1000/codellama-${size}B-t$temperature \
    --model_name_or_path codellama/CodeLlama-${size}b-Python-hf \
    --temperature $temperature \
    --eval_batch_size 8 \
    --max_examples 200 \
    --unbiased_sampling_size_n 20
