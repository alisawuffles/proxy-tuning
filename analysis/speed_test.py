import torch
from tqdm import tqdm
from eval.utils import load_lm_and_tokenizer, load_dexperts_model_and_tokenizer
from eval.templates import create_prompt_with_llama2_chat_format
from transformers import LogitsProcessorList, MinNewTokensLengthLogitsProcessor
import argparse
import time
import numpy as np


@torch.inference_mode()
def main(args):
    if args.setting == 'tuned':
        model_name = f'meta-llama/Llama-2-{args.size}b-chat-hf'
        model, tokenizer = load_lm_and_tokenizer(
            model_name_or_path=model_name,
            tokenizer_name_or_path=model_name,
            device_map='balanced',
            use_fast_tokenizer=True,
        )
    elif args.setting == 'dexperts':
        base_model_name = f'meta-llama/Llama-2-{args.size}b-hf'
        model, tokenizer = load_dexperts_model_and_tokenizer(
            base_model_name,
            expert_model_name_or_path='meta-llama/Llama-2-7b-chat-hf',
            device_map='balanced',
            use_fast_tokenizer=True,
        )

    long_len, short_len = 512, 8
    for prompt_len, output_len in [(short_len, long_len), (long_len, short_len), (short_len, short_len)]:
        # construct & tokenize prompt
        prompt = ('hi ' * (prompt_len-1)).strip()
        if args.setting == 'tuned':
            prompt = create_prompt_with_llama2_chat_format([{"role": "user", "content": prompt}], add_bos=False)
        input_ids = tokenizer([prompt], return_tensors='pt').input_ids.cuda()

        # use logits processor to force generation to be at least output_len
        logits_processor = MinNewTokensLengthLogitsProcessor(
            prompt_length_to_skip=prompt_len,
            min_new_tokens=output_len,
            eos_token_id=tokenizer.eos_token_id
        )
        logits_processor = LogitsProcessorList([logits_processor])

        # use max_new_tokens to force generation to be no more than output_len
        max_new_tokens = output_len

        # measure generation time
        times = []
        num_trials = 100
        for trial in tqdm(range(num_trials)):
            start_time = time.time()
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                logits_processor=logits_processor,
                do_sample=False
            )
            assert len(output_ids[0][len(input_ids[0]):]) == output_len, f"output has length {len(output_ids[0][len(input_ids[0]):])} instead of {output_len}"
            times.append(time.time() - start_time)
            if trial == 0:
                print(tokenizer.decode(output_ids[0], skip_special_tokens=True), flush=True)

        print(f'prompt_len={prompt_len}, output_len={output_len}, mean={np.mean(times)}, std={np.std(times)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting",
        choices=['dexperts', 'tuned']
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=[13, 70]
    )
    args = parser.parse_args()
    main(args)
