import argparse
import os
import json
import torch
import pandas as pd
import openai

from constants import AI2_OPENAI_API_KEY
from eval.truthfulqa.metrics import run_end2end_GPT3
from eval.utils import (
    load_hf_lm_and_tokenizer,
    load_dexperts_model_and_tokenizer,
    generate_completions,
    dynamic_import_function,
    ensure_dir
)
from eval.truthfulqa.utilities import format_question_mc

CHOICES = 'ABCD'
MC_ANSWER_PREFIX = 'The answer is:'
openai.api_key = AI2_OPENAI_API_KEY


def trim_answer(answer):
    """
    Trim generated answer for open-ended evaluation setting.
    """
    # remove spaces at the beginning and end
    answer = answer.strip()
    # remove the "Answer:" prefix if it exists
    if answer.startswith('Answer:'):
        answer = answer[len('Answer:'):].strip()
    # reformat line-breaks for long-form answers
    answer = answer.replace('\n\n', '\n')
    return answer


def run_hf_model(
    test_df, model, tokenizer, batch_size=1, max_new_tokens=50
):
    """Stores answers from autoregressive HF models (GPT-2, GPT-Neo)"""
    prompts = []
    for _, row in test_df.iterrows():
        prompt = row['Question']
        if args.use_chat_format:
            chat_formatting_function = dynamic_import_function(args.chat_formatting_function)
            messages = []
            if args.system_prompt:
                messages += [{"role": "system", "content": args.system_prompt}]
            messages += [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "Answer:"
            else:
                prompt += " Answer:"
        else:
            prompt += "\n\nAnswer:"
        prompts.append(prompt)

    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    print(prompts[0], flush=True)

    outputs = generate_completions(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    assert len(outputs) == len(prompts)
    test_df['output'] = [trim_answer(o) for o in outputs]

    print("Running GPT-based evaluation metrics!")
    try:
        test_df = run_end2end_GPT3('GPT-true', args.gpt_true_model_name, test_df, info=False)
        test_df = run_end2end_GPT3('GPT-info', args.gpt_info_model_name, test_df, info=True)
    except Exception as err:
        print(err)

    test_df["GPT-true-info acc"] = test_df["GPT-true acc"] * test_df["GPT-info acc"]
    test_df.to_json(os.path.join(args.save_dir, "open_results.jsonl"), lines=True, orient='records')

    # format and print basic results
    results = format_results(test_df)
    results = results.mean(axis=0).to_dict()
    print(results)

    with open(os.path.join(args.save_dir, 'open_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return test_df


def run_hf_model_mc(test_df, model, tokenizer, batch_size=1):
    """Runs multiple-choice metrics for autoregressive HuggingFace models (GPT-2, GPT-Neo)"""
    prompts = []
    answer_idxs = []
    for _, row in test_df.iterrows():
        # prompt for all answers
        prompt, answer_idx = format_question_mc(row)
        if args.use_chat_format:
            chat_formatting_function = dynamic_import_function(args.chat_formatting_function)
            messages = []
            if args.system_prompt:
                messages += [{"role": "system", "content": args.system_prompt}]
            messages += [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            prompt += MC_ANSWER_PREFIX if prompt[-1] in ["\n", " "] else " " + MC_ANSWER_PREFIX
        else:
            prompt += "\n\n" + MC_ANSWER_PREFIX

        prompts.append(prompt)
        answer_idxs.append(answer_idx)

    test_df['mc_prompt'] = prompts
    test_df['mc_answer_idx'] = answer_idxs
    with open(os.path.join(args.save_dir, "example_mc_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    print(prompts[0], flush=True)

    # note that the token corresponding to the period in "A." and "A ." are different
    stop_sequences = ["B.", "B)", "B:"]
    stop_sequences = [tokenizer.encode(x, add_special_tokens=False)[1:] for x in stop_sequences]
    outputs = generate_completions(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        do_sample=False,
        max_new_tokens=10,
        stop_id_sequences=stop_sequences
    )
    test_df['mc_output'] = outputs

    # get the metrics
    parsed_outputs = []
    for i, row in test_df.iterrows():
        # since we use period '.' as stop token, interpret last char as answer
        o = row['mc_output']
        to_remove = ['(', '\\begin{blockquote}', '\\begin{code}', '<blockquote>', ' **', '>', '```\n']
        for r in to_remove:
            o = o.replace(r, '')
        o = o.strip()
        if o and o[0] in CHOICES:  # o is not empty string
            parsed_output = o[0]  # use first char as prediction
        else:
            parsed_output = ''

        parsed_outputs.append(parsed_output)

    test_df['parsed_output'] = parsed_outputs

    test_df['correct'] = [
        pred == true if pred else float('nan')
        for pred, true in zip(test_df.parsed_output, test_df.mc_answer_idx)
    ]

    acc = test_df.correct.mean(skipna=True)
    num_invalid_pred = test_df.correct.isna().sum()

    print(f"Invalid predictions: {num_invalid_pred}/{len(test_df)}")

    # drop columns from open-ended evaluation (if they exist)
    drop_columns = ['output'] + [col for col in test_df.columns if col.startswith('GPT')]
    test_df = test_df.drop(drop_columns, axis=1, errors='ignore')
    test_df.to_json(os.path.join(args.save_dir, "mc_results.jsonl"), lines=True, orient='records')

    # format and print basic results
    results = {
        'acc': acc,
        'num_invalid_predictions': int(num_invalid_pred),
        'tot': len(test_df)
    }
    print(results)

    with open(os.path.join(args.save_dir, 'mc_metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return test_df


def format_results(results):
    results = results[[x for x in results.columns if (results[x].dtype != object)]]
    return results


def main(args):
    ensure_dir(args.save_dir)
    test_df = pd.read_csv(os.path.join(args.data_dir, "TruthfulQA.csv"))

    if args.max_examples is not None:
        test_df = test_df.sample(args.max_examples, random_state=42)

    # load individual HF models
    if args.model_name_or_path:
        print("Loading HF model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )

    if 'open' in args.settings:
        if args.base_model_name_or_path:
            print("Loading DExperts model and tokenizer...")
            model, tokenizer = load_dexperts_model_and_tokenizer(
                args.base_model_name_or_path,
                args.expert_model_name_or_path,
                system_prompt=args.system_prompt,
                alpha=args.alpha,
                chat_response_prefix="Answer:",
                load_in_8bit=args.load_in_8bit,
                use_fast_tokenizer=not args.use_slow_tokenizer,
            )
        print("Running generations!")
        run_hf_model(
            test_df, model, tokenizer, batch_size=args.eval_batch_size, max_new_tokens=500
        )
    if 'mc' in args.settings:
        if args.base_model_name_or_path:
            print("Loading DExperts model and tokenizer...")
            model, tokenizer = load_dexperts_model_and_tokenizer(
                args.base_model_name_or_path,
                args.expert_model_name_or_path,
                system_prompt=args.system_prompt,
                alpha=args.alpha,
                chat_response_prefix=MC_ANSWER_PREFIX,
                load_in_8bit=args.load_in_8bit,
                use_fast_tokenizer=not args.use_slow_tokenizer,
            )
        print("Running multiple-choice classification!")
        run_hf_model_mc(
            test_df, model, tokenizer, batch_size=args.eval_batch_size,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="The HuggingFace model to be evaluated."
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
        "--openai_engine",
        type=str,
        default=None,
        help="If specified, we will evaluate the OpenAI engine."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/eval/truthfulqa",
        help="The directory containing the truthfulqa data. Download from https://github.com/sylinrl/TruthfulQA/tree/main/data."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/truthfulqa/",
        help="The directory to save the results."
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="The number of instances to evaluate. If not given, we will evaluate all instances."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--expert_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
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
        '--settings',
        nargs='+',
        choices=['open', 'mc'],
        help='Settings'
    )
    parser.add_argument(
        '--gpt_true_model_name',
        type=str,
        help='If `open` setting is used, the trained GPT truthfulness model name should be provided.'
    )
    parser.add_argument(
        '--gpt_info_model_name',
        type=str,
        help='If `open` setting is used, the trained GPT informativeness model name should be provided.'
    )
    args = parser.parse_args()
    main(args)
