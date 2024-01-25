import argparse
import os
import torch
import pandas as pd
import numpy as np
import json
from eval.utils import (
    ensure_dir,
    generate_completions,
    load_hf_lm_and_tokenizer,
    load_dexperts_model_and_tokenizer,
)


def create_prompt(row):
    return f'Question: {row["question"]}\nAnswer:'


def main(args):
    ensure_dir(args.save_dir)

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
        )

    # use dev set because test set answers are hidden
    test_df = pd.read_json(os.path.join(args.data_dir, "dev.jsonl"), lines=True)
    if args.max_examples and args.max_examples < test_df.shape[0]:
        test_df = test_df.sample(args.max_examples, random_state=42)

    # Create prompts
    prompts = []
    for i, row in test_df.iterrows():
        prompts.append(create_prompt(row))

    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])
    print(prompts[0], flush=True)

    new_line_token = tokenizer.encode("\n\n", add_special_tokens=False)[-1]
    outputs = generate_completions(
        model,
        tokenizer,
        prompts,
        batch_size=args.eval_batch_size,
        do_sample=False,
        max_new_tokens=20,
        stop_id_sequences=[[new_line_token]],
    )

    test_df['output'] = [o.strip() for o in outputs]
    cors = []
    for i, row in test_df.iterrows():
        # ignore casing
        pred = row['output'].lower()
        answers = [a.strip().lower() for a in row['answers']]
        cors.append(pred in answers)

    test_df['correct'] = cors
    acc = np.nanmean(cors)
    print(f"Accuracy: {np.round(acc, 3)}")

    test_df.to_json(os.path.join(args.save_dir, "predictions.jsonl"), lines=True, orient='records')

    # save results
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as fo:
        json.dump({
            "acc": acc,
            "tot": len(test_df)
        }, fo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/eval/triviaqa"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/triviaqa/llama-7B/"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        help="if specified, a maximum of max_examples for evaluation"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default='meta-llama/Llama-2-13b-hf',
    )
    parser.add_argument(
        "--expert_model_name_or_path",
        type=str,
        default='UW/llama2-7b-triviaqa',
    )
    args = parser.parse_args()

    main(args)
