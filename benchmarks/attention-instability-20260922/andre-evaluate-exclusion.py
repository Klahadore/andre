"""Read-only paired held-out context audit. Never updates model weights."""
import argparse
from functools import partial
import json
from pathlib import Path
import random
import sys
import time
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset
sys.path.insert(0,'/home/ubuntu/andre')
from dataset import ScBaseCountDataset,collate_fn
from model import Andre
from train import predict,move_batch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',required=True)
p.add_argument('--data-root',required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--cells',type=int,default=6400)
p.add_argument('--batch-size',type=int,default=128)
a=p.parse_args()
torch.set_num_threads(4)
torch.set_float32_matmul_precision('high')
checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
assert checkpoint['model_architecture']=='pre_ln_v1'
cfg=checkpoint['config']; seed=cfg['seed']+1
model=Andre(width=cfg['hidden_dim'], qk_norm=False).cuda().eval()
model.load_state_dict(checkpoint['model'])
step=checkpoint['step'];del checkpoint
val=ScBaseCountDataset(a.data_root,'val')
indices=random.Random(seed).sample(range(len(val)),min(len(val),a.cells))
loader=DataLoader(Subset(val,indices),batch_size=a.batch_size,num_workers=0,
 collate_fn=partial(collate_fn,length=cfg['context_length']),generator=torch.Generator().manual_seed(seed))
# Match the trainer's fixed validation cells, sampled genes and MASK positions.
torch.manual_seed(seed)
excluded_losses=[];normal_losses=[];shuffled_losses=[];targets_all=[];predictions=[]
prob_sum=torch.zeros(36601,device='cuda'); entropy_sum=0.
started=time.monotonic()
with torch.no_grad():
 for cpu in loader:
  batch=move_batch(cpu,torch.device('cuda'));targets=batch['targets']-2
  rows=torch.arange(len(targets),device='cuda')
  shuffled={k:v.roll(1,0) for k,v in batch.items()}
  shuffled['counts'][rows,shuffled['mask_positions']]=batch['counts'][rows,batch['mask_positions']]
  with torch.autocast('cuda',dtype=torch.bfloat16):
   logits=predict(model,batch).float()
   wrong_context=predict(model,shuffled).float()
  normal_losses.append(nn.functional.cross_entropy(logits,targets,reduction='none').cpu())
  excluded=logits.clone();ids=batch['gene_ids'];real=ids>=2
  row_indices=torch.arange(len(targets),device='cuda')[:,None].expand_as(ids)
  excluded[row_indices[real],ids[real]-2]=-torch.inf
  excluded_losses.append(nn.functional.cross_entropy(excluded,targets,reduction='none').cpu())
  shuffled_losses.append(nn.functional.cross_entropy(wrong_context,targets,reduction='none').cpu())
  probabilities=logits.softmax(-1)
  prob_sum+=probabilities.sum(0)
  entropy_sum+=float(-(probabilities*probabilities.clamp_min(1e-30).log()).sum())
  targets_all.append(targets.cpu());predictions.append(logits.argmax(-1).cpu())
normal=torch.cat(normal_losses);shuffled=torch.cat(shuffled_losses)
targets=torch.cat(targets_all);preds=torch.cat(predictions);delta=shuffled-normal
assert torch.isfinite(normal).all() and torch.isfinite(shuffled).all()
mean_prob=(prob_sum/len(targets)).cpu().clamp_min(1e-30)
result={'checkpoint':a.checkpoint,'checkpoint_step':step,'width':cfg['hidden_dim'],
 'loss_excluding_visible_genes':torch.cat(excluded_losses).mean().item(),'seed':seed,'cells':len(targets),'batch_size':a.batch_size,'loss':normal.mean().item(),
 'shuffled_context_loss':shuffled.mean().item(),'shuffled_context_gain':delta.mean().item(),
 'paired_gain_standard_error_iid':(delta.std()/len(delta)**.5).item(),
 'uncertainty_note':'Naive paired standard error treats cells as independent; experiment clustering can increase uncertainty.',
 'fraction_cells_helped_by_correct_context':(delta>0).float().mean().item(),
 'constant_mean_prediction_loss':(-mean_prob.log()[targets]).mean().item(),
 'prediction_kl_to_mean':float(-(mean_prob*mean_prob.log()).sum())-entropy_sum/len(targets),
 'accuracy':(preds==targets).float().mean().item(),'unique_top1_predictions':preds.unique().numel(),
 'seconds':time.monotonic()-started}
a.out.parent.mkdir(exist_ok=True,parents=True)
a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
