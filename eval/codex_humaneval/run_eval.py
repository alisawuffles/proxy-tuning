import argparse
import os
import json
import random
import torch
import pandas as pd
from eval.utils import (
    generate_completions,
    load_hf_lm_and_tokenizer,
    load_dexperts_model_and_tokenizer,
    ensure_dir
)
from eval.codex_humaneval.data import write_jsonl, read_problems
from eval.codex_humaneval.evaluation import evaluate_functional_correctness, clean_generation


def main(args):
    random.seed(42)

    ensure_dir(args.save_dir)

    test_data = list(read_problems(args.data_file).values())
    if args.max_examples is not None and len(test_data) > args.max_examples:
        test_data = random.sample(test_data, args.max_examples)
    print("Number of examples:", len(test_data))

    prompts = [example["prompt"] for example in test_data]

    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    print(prompts[0], flush=True)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
    elif args.base_model_name_or_path:
        model, tokenizer = load_dexperts_model_and_tokenizer(
            args.base_model_name_or_path,
            args.expert_model_name_or_path,
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=not args.use_slow_tokenizer,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
        )

    # We will write intermediate output for every sampling iteration
    # This allows us to use ckpt resources on hyak :)
    ensure_dir(os.path.join(args.save_dir, 'sampling_iterations'))
    # Because many tokenizers will treat the word after space differently from the original word alone,
    # to be consistent, we add a space before tokenization and remove it after tokenization.
    stop_sequences = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]
    stop_sequences = [tokenizer.encode(" " + x, add_special_tokens=False)[1:] for x in stop_sequences]
    banned_sequences = ['pass', '...']
    banned_id_sequences = [tokenizer.encode(x, add_special_tokens=False) for x in banned_sequences]
    outputs_per_sampling_iter = []
    for sampling_iter in range(args.unbiased_sampling_size_n):
        print(f"Sampling iter: {sampling_iter} / {args.unbiased_sampling_size_n}")
        iter_save_path = os.path.join(args.save_dir, 'sampling_iterations', f'{sampling_iter}.jsonl')
        if os.path.exists(iter_save_path) and not args.overwrite:  # read results, if they exist
            sampling_outputs = pd.read_json(iter_save_path, lines=True).output.tolist()
        else:
            sampling_outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=512,
                batch_size=args.eval_batch_size,
                stop_id_sequences=stop_sequences,
                banned_id_sequences=banned_id_sequences,
                num_return_sequences=1,  # we don't use the hf num_return_sequences, because otherwise the real batch size will be multiplied by it and often cause oom.
                do_sample=True,  # if only pass@1 is evaluated, we do greedy decoding.
                top_p=0.95,
                temperature=args.temperature,
            )
            sampling_outputs = [clean_generation(o) for o in sampling_outputs]
            pd.DataFrame({'output': sampling_outputs}).to_json(iter_save_path, lines=True, orient='records')  # save results
        outputs_per_sampling_iter.append(sampling_outputs)

    # regroup the outputs to match the number of test data
    outputs = []
    for i in range(len(prompts)):
        for j in range(args.unbiased_sampling_size_n):
            outputs.append(outputs_per_sampling_iter[j][i])

    # duplicates test data to match the number of outputs.
    duplicate_test_data = [
        example for example in test_data for _ in range(args.unbiased_sampling_size_n)
    ]
    assert len(duplicate_test_data) == len(outputs)
    predictions = [{"task_id": example["task_id"], "prompt": example["prompt"], "completion": output} for example, output in zip(duplicate_test_data, outputs)]
    prediction_save_path = os.path.join(args.save_dir, "predictions.jsonl")
    write_jsonl(prediction_save_path, predictions)

    pass_at_k_results = evaluate_functional_correctness(
        sample_file=prediction_save_path,
        k=args.eval_pass_at_ks,
        problems={example["task_id"]: example for example in test_data},
        n_workers=64
    )

    print(pass_at_k_results)

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        json.dump(pass_at_k_results, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/eval/codex_humaneval/HumanEval.jsonl",
        help="Path to the HumanEval data file."
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/codex_eval",
        help="Directory to save the results."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--eval_pass_at_ks",
        nargs="+",
        type=int,
        default=[1],
        help="Multiple k's that we will report pass@k."
    )
    parser.add_argument(
        "--unbiased_sampling_size_n",
        type=int,
        default=20,
        help="Codex HumanEval requires `n` sampled generations per prompt, to estimate the unbiased pass@k."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for sampling. This is should be low for evaluating smaller pass@k, and high for larger pass@k."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default='meta-llama/Llama-2-13b-chat-hf',
    )
    parser.add_argument(
        "--expert_model_name_or_path",
        type=str,
        default='meta-llama/Llama-2-7b-chat-hf',
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If given, we will use the vllm library, which will likely increase the inference throughput."
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help="Whether to overwrite previous results, if they exist."
    )
    args = parser.parse_args()
    main(args)
