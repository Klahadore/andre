import sys,json,random,gc,torch
from pathlib import Path
sys.path.insert(0,'/home/ubuntu/andre')
from model import Andre
from dataset import ScBaseCountDataset,collate_fn
from train import predict
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
ds=ScBaseCountDataset('/opt/dlami/nvme/andre/lmdb','val');torch.manual_seed(43)
b={k:v.cuda() for k,v in collate_fn(ds.__getitems__(random.Random(43).sample(range(len(ds)),8)),512).items()}
results=[]
for filename in ['last.pt','best.pt']:
 c=torch.load(Path('/opt/dlami/nvme/andre/runs/width-768-preln-compiled-10b-tokens')/filename,map_location='cpu',weights_only=True,mmap=True)
 m=Andre(width=768, qk_norm=False).cuda();m.load_state_dict(c['model']);step=c['step'];del c
 m.train()
 for mode in ['eager_bf16','eager_fp32','compiled_bf16']:
  if mode=='compiled_bf16':
   for layer in m.transformer_layers:layer.compile(dynamic=False)
  for dropout in [0.,.1]:
   for layer in m.modules():
    if isinstance(layer,torch.nn.Dropout):layer.p=dropout
    if isinstance(layer,torch.nn.MultiheadAttention):layer.dropout=dropout
   for seed in [42,123]:
    torch.manual_seed(seed);m.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mode!='eager_fp32'):
     loss=torch.nn.functional.cross_entropy(predict(m,b),b['targets']-2)
    loss.backward()
    norms=torch.stack([p.grad.float().norm() for p in m.parameters() if p.grad is not None])
    row={'step':step,'mode':mode,'dropout':dropout,'seed':seed,'loss':float(loss.detach()),'grad_norm':float(norms.norm()),'layer_grad_rms':[float(l.self_attn.in_proj_weight.grad.square().mean().sqrt()) for l in m.transformer_layers]}
    results.append(row);print(json.dumps(row),flush=True)
 del m;gc.collect();torch.cuda.empty_cache()
Path('/opt/dlami/nvme/andre/dropout-probe.json').write_text(json.dumps(results,indent=2)+'\n')
