import torch
from tqdm import tqdm
import pickle
import pandas as pd
from eval.utils import load_dexperts_model_and_tokenizer


def get_equation_lhs_rhs_indices(token_list):
    equal_indices = [i for i, x in enumerate(token_list) if x == '=']
    lhs_idx, rhs_idx = [], []

    for equal_idx in equal_indices:
        # go left until it's no longer a number or symbol
        left_idx, right_idx = equal_idx - 1, equal_idx + 1
        while True:
            if left_idx < 0 or not (token_list[left_idx].isdigit() or token_list[left_idx] in ",$€+-x*/"):
                break
            left_idx -= 1

        # go right until it's no longer a number or symbol
        while True:
            if right_idx >= len(token_list) or \
                 not (token_list[right_idx].isdigit() or token_list[right_idx] in ",$€+-x*/"):
                break
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
    batch_size = 32
    batched_results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Batches"):
        batch_prompts = prompts[i: i + batch_size]
        batch_inputs = tokenizer(batch_prompts, return_tensors='pt', padding='longest')
        input_ids = batch_inputs.input_ids.cuda()
        attention_mask = batch_inputs.attention_mask.cuda()
        results = model.get_probs_for_analysis(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            do_sample=False
        )
        batched_results.append(results)

    # flatten batch results into a list of results for each prompt
    all_results = []
    for batch in batched_results:
        this_batch_size = len(batch['token'][0])
        for i in range(this_batch_size):
            ex = {}
            ex['token'] = [x[i] for x in batch['token']]
            if '</s>' in ex['token']:
                output_len = ex['token'].index('</s>')
            else:
                output_len = len(ex['token'])
            ex['token'] = ex['token'][:output_len]
            for k in batch.keys():
                if k != 'token':
                    ex[k] = batch[k][i, :].tolist()[:output_len]
            all_results.append(ex)

    # save as pickle
    with open('analysis/gsm_token_probs.pkl', 'wb') as fo:
        pickle.dump(all_results, fo)
    print('Results saved to analysis/gsm_token_probs.pkl', flush=True)


if __name__ == "__main__":
    main()
