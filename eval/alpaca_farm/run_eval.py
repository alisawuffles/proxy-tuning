import os
import json
import argparse
import logging
import random
import datasets
import pandas as pd
from alpaca_eval import evaluate as alpaca_farm_evaluate
from eval.utils import (
    generate_completions,
    dynamic_import_function,
    load_lm_and_tokenizer,
    load_dexperts_model_and_tokenizer,
    ensure_dir
)


def main(args):
    random.seed(42)
    ensure_dir(args.save_dir)

    logging.info("loading data and model...")
    if args.data_path:
        alpaca_eval_data = pd.read_json(args.data_path, lines=True).to_dict(orient="records")
    else:
        alpaca_eval_data = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

    prompts = []
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    for example in alpaca_eval_data:
        prompt = example["instruction"]
        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
        prompts.append(prompt)

    prompts = prompts[:args.max_examples]

    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    if args.model_name_or_path:
        model, tokenizer = load_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
    else:
        model, tokenizer = load_dexperts_model_and_tokenizer(
            args.base_model_name_or_path,
            args.expert_model_name_or_path,
            system_prompt=args.system_prompt,
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=not args.use_slow_tokenizer
        )

    stop_sequences = ["\n\nComment:"]  # degenerate stuff for llama 2
    stop_sequences = [tokenizer.encode(" " + x, add_special_tokens=False)[1:] for x in stop_sequences]
    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        stop_id_sequences=stop_sequences,
        do_sample=False,
        batch_size=args.eval_batch_size if args.eval_batch_size else 1,
    )

    model_results = []
    model_name = os.path.basename(args.save_dir)
    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
        for example, output in zip(alpaca_eval_data, outputs):
            example["output"] = output.strip()
            example["generator"] = model_name
            fout.write(json.dumps(example) + "\n")
            model_results.append(example)

    evaluation_args = {
        "model_outputs": model_results,
        "annotators_config": "alpaca_eval_gpt4_0314",
        "output_path": args.save_dir,
        "is_return_instead_of_print": True,
        "precomputed_leaderboard": None,
        "is_cache_leaderboard": False,
        "caching_path": os.path.join(args.save_dir, "alpaca_eval_annotator_cache.json")
    }

    if args.reference_path:
        evaluation_args["reference_outputs"] = args.reference_path

    df_leaderboard, annotations = alpaca_farm_evaluate(**evaluation_args)

    # save to json
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fout:
        for k in df_leaderboard.to_dict():
            df_leaderboard[k] = df_leaderboard[k][model_name]
        json.dump(df_leaderboard, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference_path",
        type=str,
        default=None,
        help="Path to the reference outputs."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/alpaca_farm")
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="The number of instances to evaluate. If not given, we will evaluate all instances."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default='meta-llama/Llama-2-13b-hf',
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
    args = parser.parse_args()

    main(args)
