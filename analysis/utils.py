import torch


def flatten_batch_results(batch):
    """
    Flatten batch results into a list of results for each prompt
    """
    all_results = []
    batch_size = len(batch['tokens'][0])
    for i in range(batch_size):
        ex = {}
        ex['tokens'] = [x[i] for x in batch['tokens']]  # list of tokens
        if '</s>' in ex['tokens']:
            output_len = ex['tokens'].index('</s>')
        else:
            output_len = len(ex['tokens'])
        ex['token_ids'] = [x.squeeze(dim=0)[i].item() for x in batch['token_ids']][:output_len]  # list of tokens
        ex['tokens'] = ex['tokens'][:output_len]
        for k in batch.keys():
            if k.startswith('logits'):
                ex[k] = batch[k][i, ...][:output_len, ...]
        all_results.append(ex)
    return all_results


def summarize_results(results):
    """
    Logit vectors are huge, so let's just extract the key information: the probability of the 
    DExperts next-token and the top prediction from each model.
    """
    shortened_results = []
    logit_keys = [k for k in results[0].keys() if k.startswith('logits')]
    for ex in results:
        for k in logit_keys:
            model = '_'.join(k.split('_')[1:])
            probs = ex[k].softmax(dim=-1)
            ex[f'p_{model}'] = probs.gather(-1, torch.tensor(ex['token_ids']).unsqueeze(-1).cuda()).squeeze()
            ex[f'preds_{model}'] = ex[k].argmax(dim=-1)
            del ex[k]

        shortened_results.append(ex)
    return shortened_results
