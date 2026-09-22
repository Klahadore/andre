"""Paired continuation experiment; production checkpoints are read only."""
import sys,json,random,time,gc,argparse,statistics
from pathlib import Path
from functools import partial
import torch
from torch.utils.data import DataLoader,Subset
sys.path.insert(0,'/home/ubuntu/andre')
from model import Andre
from dataset import ScBaseCountDataset,collate_fn,BlockShuffleSampler
from train import predict,move_batch,evaluate

def main():
 p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);p.add_argument('--steps',type=int,default=1500);a=p.parse_args()
 torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
 root=Path('/opt/dlami/nvme/andre');out=root/f'random-sampling-ablation-{a.width}';out.mkdir(exist_ok=True)
 train=ScBaseCountDataset(root/'lmdb','train');val=ScBaseCountDataset(root/'lmdb','val')
 indices=random.Random(43).sample(range(len(val)),6400)
 vl=DataLoader(Subset(val,indices),batch_size=128,collate_fn=partial(collate_fn,length=512),generator=torch.Generator().manual_seed(43))
 short=DataLoader(Subset(val,indices[:1024]),batch_size=128,collate_fn=partial(collate_fn,length=512),generator=torch.Generator().manual_seed(43))
 for attention_dropout in [0.]:
  torch.manual_seed(4242)
  c=torch.load(root/f'runs/width-{a.width}-preln-compiled-10b-tokens/best.pt',map_location='cpu',weights_only=True,mmap=True)
  m=Andre(width=a.width, qk_norm=False).cuda();m.load_state_dict(c['model']);o=torch.optim.AdamW(m.parameters(),lr=.0003);o.load_state_dict(c['optimizer']);source_step=c['step'];del c
  for g in o.param_groups:g['lr']=.0003
  for l in m.transformer_layers:l.self_attn.dropout=attention_dropout;l.compile(dynamic=False)
  sampler=random.Random(4242).sample(range(len(train)),a.steps*128)
  loader=DataLoader(train,batch_size=128,sampler=sampler,collate_fn=partial(collate_fn,length=512),num_workers=8,persistent_workers=True,prefetch_factor=2,multiprocessing_context='spawn',pin_memory=True,generator=torch.Generator().manual_seed(4242))
  log=(out/f'attention-dropout-{attention_dropout}.jsonl').open('w')
  def emit(r):
   r.update(width=a.width,attention_dropout=attention_dropout,source_step=source_step,sampler='global_random_without_replacement');s=json.dumps(r);print(s,flush=True);log.write(s+'\n');log.flush()
  emit({'step':0,**evaluate(m,vl,torch.device('cuda'),True,43)})
  batches=iter(loader);norms=[];losses=[];start=time.monotonic();m.train()
  for step in range(1,a.steps+1):
   b=move_batch(next(batches),torch.device('cuda'));o.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=torch.nn.functional.cross_entropy(predict(m,b),b['targets']-2)
   loss.backward();g=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);o.step()
   norms.append(float(g));losses.append(float(loss.detach()))
   if step%25==0:
    row={'step':step,'train/loss':statistics.mean(losses),'grad_norm_median':statistics.median(norms),'grad_norm_max':max(norms),'seconds':time.monotonic()-start}
    if step%250==0:row.update(evaluate(m,vl if step==a.steps else short,torch.device('cuda'),True,43))
    emit(row);norms=[];losses=[]
  torch.save({'model':m.state_dict(),'optimizer':o.state_dict(),'step':source_step+a.steps,'model_architecture':m.architecture,'experiment_attention_dropout':attention_dropout},out/f'attention-dropout-{attention_dropout}.pt')
  log.close();del batches,loader,m,o;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
