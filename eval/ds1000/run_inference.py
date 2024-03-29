import argparse
import os
import random
import pandas as pd
from eval.ds1000.ds1000 import DS1000Dataset
from eval.utils import (
    ensure_dir,
    load_lm_and_tokenizer,
    load_dexperts_model_and_tokenizer,
    generate_completions
)
from eval.codex_humaneval.evaluation import clean_generation


def preprocess_data():
    ds1000 = DS1000Dataset("data/eval/ds1000", mode='Completion')
    examples = []
    for lib in ds1000.libs:
        for problem in ds1000[lib]:
            examples.append({
                'lib': lib,
                'problem_id': problem.problem_id,
                'prompt': problem['prompt'],
                'solution': problem['reference_code']
            })
    return pd.DataFrame(examples)


def main(args):
    random.seed(42)
    ensure_dir(args.save_dir)

    test_df = preprocess_data()

    if args.max_examples:
        test_df = test_df.sample(args.max_examples, random_state=42)

    prompts = test_df.prompt.tolist()
    prompts = [p if p.endswith('\n') else p + '\n' for p in prompts]

    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    print(prompts[0], flush=True)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        model, tokenizer = load_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
    elif args.base_model_name_or_path:
        model, tokenizer = load_dexperts_model_and_tokenizer(
            args.base_model_name_or_path,
            args.expert_model_name_or_path,
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )

    # We will write intermediate output for every sampling iteration
    # This allows us to use ckpt resources on hyak :)
    ensure_dir(os.path.join(args.save_dir, 'sampling_iterations'))

    # Because many tokenizers will treat the word after space differently from the original word alone,
    # to be consistent, we add a space before tokenization and remove it after tokenization.
    stop_sequences = ["\n</code>", '\n# SOLUTION END', '\nEND SOLUTION']
    stop_sequences = [tokenizer.encode(x, add_special_tokens=False)[1:] for x in stop_sequences]
    banned_id_sequences = [tokenizer.encode(x, add_special_tokens=False) for x in ['pass', '...']]
    banned_begin_ids = [tokenizer.encode('\n</code>', add_special_tokens=False)[2]]
    outputs_per_sampling_iter = []
    for sampling_iter in range(args.unbiased_sampling_size_n):
        print(f"Sampling iter: {sampling_iter} / {args.unbiased_sampling_size_n}")
        iter_save_path = os.path.join(args.save_dir, 'sampling_iterations', f'{sampling_iter}.jsonl')
        if os.path.exists(iter_save_path) and not args.overwrite:  # read results, if they exist
            sampling_outputs = pd.read_json(iter_save_path, lines=True).output.tolist()
            pd.DataFrame({'output': sampling_outputs}).to_json(
                iter_save_path, lines=True, orient='records'
            )  # save results
        else:
            sampling_outputs = generate_completions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=512,
                batch_size=args.eval_batch_size,
                stop_id_sequences=stop_sequences,
                banned_id_sequences=banned_id_sequences,
                banned_begin_ids=banned_begin_ids,
                num_return_sequences=1,
                temperature=args.temperature,
                top_p=0.95,
                do_sample=True,
            )
            sampling_outputs = [clean_generation(o) for o in sampling_outputs]
            pd.DataFrame({'output': sampling_outputs}).to_json(
                iter_save_path, lines=True, orient='records'
            )  # save results
        outputs_per_sampling_iter.append(sampling_outputs)

    # regroup the outputs to match the number of test data
    outputs = []
    for i in range(len(prompts)):
        outputs.append([outputs_per_sampling_iter[j][i] for j in range(args.unbiased_sampling_size_n)])

    print(outputs[0][0])
    assert len(outputs) == len(test_df)

    test_df['output'] = outputs
    prediction_save_path = os.path.join(args.save_dir, "predictions.jsonl")
    test_df.to_json(prediction_save_path, orient='records', lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference."
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
        "--overwrite",
        type=bool,
        default=False,
        help="Whether to overwrite previous results, if they exist."
    )
    args = parser.parse_args()
    main(args)
