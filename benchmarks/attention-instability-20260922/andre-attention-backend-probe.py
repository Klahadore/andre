import sys,json,random,torch,contextlib
from pathlib import Path
sys.path.insert(0,'/home/ubuntu/andre')
from model import Andre
from dataset import ScBaseCountDataset,collate_fn
from train import predict
from torch.nn.attention import sdpa_kernel,SDPBackend
torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
ds=ScBaseCountDataset('/opt/dlami/nvme/andre/lmdb','val');torch.manual_seed(43)
b={k:v.cuda() for k,v in collate_fn(ds.__getitems__(random.Random(43).sample(range(len(ds)),8)),512).items()}
c=torch.load('/opt/dlami/nvme/andre/runs/width-768-preln-compiled-10b-tokens/last.pt',map_location='cpu',weights_only=True,mmap=True)
m=Andre(width=768).cuda();m.load_state_dict(c['model']);del c;m.train()
results=[]
for backend in ['default','math','efficient','cudnn','flash']:
 for which in ['all','attention_only','residual_only']:
  for l in m.modules():
   if isinstance(l,torch.nn.Dropout):l.p=0. if which=='attention_only' else .1
   if isinstance(l,torch.nn.MultiheadAttention):l.dropout=0. if which=='residual_only' else .1
  torch.manual_seed(42);m.zero_grad(set_to_none=True)
  ctx=contextlib.nullcontext() if backend=='default' else sdpa_kernel({'math':SDPBackend.MATH,'efficient':SDPBackend.EFFICIENT_ATTENTION,'cudnn':SDPBackend.CUDNN_ATTENTION,'flash':SDPBackend.FLASH_ATTENTION}[backend])
  try:
   with ctx,torch.autocast('cuda',dtype=torch.bfloat16):
    loss=torch.nn.functional.cross_entropy(predict(m,b),b['targets']-2)
   loss.backward()
   row={'backend':backend,'dropout':which,'loss':loss.item(),'grad_norm':torch.stack([p.grad.float().norm() for p in m.parameters() if p.grad is not None]).norm().item()}
  except RuntimeError as e:row={'backend':backend,'dropout':which,'error':str(e)}
  results.append(row);print(json.dumps(row),flush=True)
Path('/opt/dlami/nvme/andre/attention-backend-probe.json').write_text(json.dumps(results,indent=2)+'\n')
