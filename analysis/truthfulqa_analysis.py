import torch
from tqdm import tqdm
import pandas as pd
from eval.utils import load_dexperts_model_and_tokenizer
from analysis.utils import flatten_batch_results, summarize_results


@torch.inference_mode()
def main():
    # load model
    model, tokenizer = load_dexperts_model_and_tokenizer(
        base_model_name_or_path='meta-llama/Llama-2-13b-hf',
        expert_model_name_or_path='meta-llama/Llama-2-7b-chat-hf',
        chat_response_prefix='Answer:'
    )

    # load dataset
    truthfulqa_df = pd.read_csv('data/eval/truthfulqa/TruthfulQA.csv')

    # construct prompts
    prompts = [row['Question'] + '\n\nAnswer:' for _, row in truthfulqa_df.iterrows()]

    # get token probabilities
    batch_size = 32
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

    torch.save(all_results, 'analysis/pkl/truthfulqa_analysis.pkl')


if __name__ == "__main__":
    main()
