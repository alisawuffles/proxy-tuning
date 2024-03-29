import torch
from tqdm import tqdm
import pandas as pd
from eval.utils import load_dexperts_model_and_tokenizer
from analysis.utils import flatten_batch_results, summarize_results


def get_equation_lhs_rhs_indices(tokens):
    """
    Returns two lists of indices, one for tokens in the LHS of equations and one for those in the RHS.

    Args:
        tokens: list of str
    """
    equal_indices = [i for i, x in enumerate(tokens) if x == '=']
    lhs_idx, rhs_idx = [], []

    for equal_idx in equal_indices:
        # go left until it's no longer a number or symbol
        left_idx, right_idx = equal_idx - 1, equal_idx + 1
        while True:
            if left_idx < 0 or not (tokens[left_idx].isdigit() or tokens[left_idx] in ",$€+-x*/"):
                break
            lhs_idx.append(left_idx)
            left_idx -= 1

        # go right until it's no longer a number or symbol
        while True:
            if right_idx >= len(tokens) or \
                 not (tokens[right_idx].isdigit() or tokens[right_idx] in ",$€+-x*/"):
                break
            rhs_idx.append(right_idx)
            right_idx += 1

    return lhs_idx, rhs_idx


@torch.inference_mode()
def main():
    # load model
    model, tokenizer = load_dexperts_model_and_tokenizer(
        base_model_name_or_path='meta-llama/Llama-2-13b-hf',
        expert_model_name_or_path='meta-llama/Llama-2-7b-chat-hf',
        chat_response_prefix='Answer:'
    )

    # load dataset
    gsm_df = pd.read_json('data/eval/gsm/test.jsonl', lines=True)

    # construct prompts
    prompt_prefix = "Answer the following question.\n\n"
    prompts = [prompt_prefix + 'Question: ' + row['question'].strip() + '\nAnswer:' for _, row in gsm_df.iterrows()]

    # get token probabilities
    batch_size = 16
    all_results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Batches"):
        batch_prompts = prompts[i: i + batch_size]
        batch_inputs = tokenizer(batch_prompts, return_tensors='pt', padding='longest')
        input_ids = batch_inputs.input_ids.cuda()
        attention_mask = batch_inputs.attention_mask.cuda()
        _, results = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=False,
            return_logits_for_analysis=True
        )

        # flatten batch results into a list of results for each prompt
        results = flatten_batch_results(results)
        shortened_results = summarize_results(results)
        all_results.extend(shortened_results)

    torch.save(all_results, 'analysis/pkl/gsm_analysis.pkl')


if __name__ == "__main__":
    main()
