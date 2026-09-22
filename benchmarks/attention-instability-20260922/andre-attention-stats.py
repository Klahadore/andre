import sys,json,random,torch,argparse,gc,math
from pathlib import Path
sys.path.insert(0,'/home/ubuntu/andre')
from model import Andre
from dataset import ScBaseCountDataset,collate_fn
from train import predict
p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);a=p.parse_args()
torch.set_num_threads(2)
ds=ScBaseCountDataset('/opt/dlami/nvme/andre/lmdb','val');torch.manual_seed(43)
b={k:v.cuda() for k,v in collate_fn(ds.__getitems__(random.Random(43).sample(range(len(ds)),8)),512).items()}
results=[]
for name in ['best.pt','last.pt']:
 c=torch.load(f'/opt/dlami/nvme/andre/runs/width-{a.width}-preln-compiled-10b-tokens/{name}',map_location='cpu',weights_only=True,mmap=True)
 m=Andre(width=a.width, attention_dropout=.1, qk_norm=False).cuda();m.load_state_dict(c['model']);step=c['step'];del c;m.train();stats=[];handles=[]
 def hook(index):
  def inspect(module,args,kwargs):
   x=args[0]
   q,k,v=torch.nn.functional.linear(x,module.in_proj_weight,module.in_proj_bias).chunk(3,-1)
   q=q.reshape(8,512,8,-1).transpose(1,2)[:,:,:16].float();k=k.reshape(8,512,8,-1).transpose(1,2).float()
   scores=q@k.transpose(-1,-2)/math.sqrt(a.width//8)
   scores.masked_fill_(~b['attention_mask'][:,None,None,:],float('-inf'))
   probs=scores.softmax(-1);valid=scores[torch.isfinite(scores)]
   stats.append({'layer':index,'qk_logit_abs_max':valid.abs().max().item(),'qk_logit_std':valid.std().item(),'attention_rows_gt_99_percent':(probs.max(-1).values>.99).float().mean().item(),'attention_entropy':(-(probs*probs.clamp_min(1e-30).log()).sum(-1)).mean().item(),'qk_weight_rms':module.in_proj_weight[:2*a.width].square().mean().sqrt().item(),'norm_input_rms':x.square().mean().sqrt().item()})
  return inspect
 for i,l in enumerate(m.transformer_layers):handles.append(l.self_attn.register_forward_pre_hook(hook(i),with_kwargs=True))
 with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):loss=torch.nn.functional.cross_entropy(predict(m,b),b['targets']-2)
 row={'width':a.width,'step':step,'file':name,'loss':loss.item(),'layers':stats};results.append(row);print(json.dumps(row),flush=True)
 for h in handles:h.remove()
 del m;gc.collect();torch.cuda.empty_cache()
Path(f'/opt/dlami/nvme/andre/attention-stats-{a.width}.json').write_text(json.dumps(results,indent=2))
