"""Controlled small-data learning probes; no production checkpoint is resumed.

Compare normalization/schedule changes on identical real cells and masks.
Each probe keeps the full gene vocabulary, count cap, depth and context length.
"""
import argparse
import gc
import json
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import ScBaseCountDataset, collate_fn
from model import Andre
from train import predict


def measure(model, batch):
    model.eval()
    targets = batch['targets'] - 2
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        logits = predict(model, batch).float()
        no_context = dict(batch)
        no_context['attention_mask'] = torch.zeros_like(batch['attention_mask'])
        no_context['attention_mask'][torch.arange(len(targets), device='cuda'), batch['mask_positions']] = True
        contextless = predict(model, no_context).float()
    p = logits.softmax(-1)
    mean_p = p.mean(0).clamp_min(1e-30)
    result = {'loss': nn.functional.cross_entropy(logits, targets).item(),
              'accuracy': (logits.argmax(-1) == targets).float().mean().item(),
              'top1_unique': logits.argmax(-1).unique().numel(),
              'prediction_kl_to_mean': (p * (p.clamp_min(1e-30).log()-mean_p.log())).sum(-1).mean().item(),
              'constant_mean_loss': nn.functional.nll_loss(mean_p.log().expand_as(logits), targets).item(),
              'no_context_loss': nn.functional.cross_entropy(contextless, targets).item(),
              'no_context_probability_l1': (p-contextless.softmax(-1)).abs().sum(-1).mean().item(),
              'shuffled_target_loss': nn.functional.cross_entropy(logits, targets.roll(1)).item()}
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--width', type=int, default=512)
    parser.add_argument('--cells', type=int, default=64)
    parser.add_argument('--microbatch', type=int, default=32)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--variants', nargs='+', default=['post_original', 'pre_same_schedule', 'post_conservative'])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    ds = ScBaseCountDataset(args.data_root, 'train')
    indices = random.Random(1234).sample(range(len(ds)), args.cells)
    torch.manual_seed(1234)
    cpu = collate_fn(ds.__getitems__(indices), length=512)
    rows = torch.arange(args.cells)
    assert torch.all(cpu['gene_ids'][rows, cpu['mask_positions']] == 1)
    assert torch.all((cpu['gene_ids'] == 1).sum(-1) == 1)
    assert not torch.any(cpu['gene_ids'] == cpu['targets'][:, None])
    assert torch.all(cpu['attention_mask'][rows, cpu['mask_positions']])
    assert torch.all(cpu['counts'][rows, cpu['mask_positions']] > 0)
    counts = torch.bincount(cpu['targets']-2, minlength=36601).float()
    frequency = counts[counts > 0] / args.cells
    data_info = {'cells': args.cells, 'context_length': 512,
                 'distinct_targets': int((counts > 0).sum()),
                 'empirical_target_entropy': float(-(frequency*frequency.log()).sum()),
                 'target_absent_from_input': True, 'exactly_one_mask_per_cell': True,
                 'indices': indices, 'seed': 1234}
    (args.out/'data-checks.json').write_text(json.dumps(data_info, indent=2)+'\n')
    batch = {k:v.to('cuda') for k,v in cpu.items()}
    variants = {'post_original': (False, 3e-4, 100),
                'pre_same_schedule': (True, 3e-4, 100),
                'post_conservative': (False, 1e-4, 300),
                'pre_conservative': (True, 1e-4, 300)}
    for name in args.variants:
        norm_first, lr, warmup = variants[name]
        torch.manual_seed(42)
        model = Andre(width=args.width, norm_first=norm_first).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01)
        config = {'variant': name, 'width': args.width, 'norm_first': norm_first,
                  'lr': lr, 'warmup': warmup, 'steps': args.steps, 'cells': args.cells}
        print(json.dumps(config), flush=True)
        started = time.monotonic()
        with (args.out/f'{name}.jsonl').open('w') as log:
            metrics = {'step': 0, **measure(model, batch)}
            log.write(json.dumps(metrics)+'\n'); log.flush()
            print(json.dumps(metrics), flush=True)
            for step in range(1, args.steps+1):
                optimizer.param_groups[0]['lr'] = lr * min(1., step/warmup)
                optimizer.zero_grad(set_to_none=True)
                for start in range(0, args.cells, args.microbatch):
                    micro = {k:v[start:start+args.microbatch] for k,v in batch.items()}
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        logits = predict(model, micro)
                        loss = nn.functional.cross_entropy(logits, micro['targets']-2)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(name)
                    (loss * (len(micro['targets'])/args.cells)).backward()
                first_rms = model.transformer_layers[0].self_attn.in_proj_weight.grad.float().square().mean().sqrt().item()
                last_rms = model.transformer_layers[-1].self_attn.in_proj_weight.grad.float().square().mean().sqrt().item()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True).item()
                optimizer.step()
                if step % 50 == 0 or step == args.steps:
                    metrics = {'step': step, 'first_attention_grad_rms': first_rms,
                               'last_attention_grad_rms': last_rms, 'grad_norm': grad_norm,
                               'seconds': time.monotonic()-started, **measure(model, batch)}
                    log.write(json.dumps(metrics)+'\n'); log.flush()
                    print(json.dumps(metrics), flush=True)
        (args.out/f'{name}-summary.json').write_text(json.dumps({**config, **metrics}, indent=2)+'\n')
        del model, optimizer
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
