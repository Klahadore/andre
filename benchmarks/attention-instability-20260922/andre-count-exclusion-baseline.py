"""Training-split empirical gene priors, with and without MASK count bins."""
import sys,json,random,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,'/home/ubuntu/andre')
from dataset import ScBaseCountDataset,collate_fn
from torch.utils.data import DataLoader,Subset
from functools import partial
torch.set_num_threads(2)
root='/opt/dlami/nvme/andre/lmdb';V=36601;B=64;N=50000

def bins(c):
 c=np.minimum(c,100000).astype(np.int64)
 return np.where(c<=32,c,33+np.floor(np.log2(np.maximum(c,1)/32)).astype(np.int64))

train=ScBaseCountDataset(root,'train');indices=random.Random(2718).sample(range(len(train)),N)
freq=np.zeros((B,V),dtype=np.float64);start=time.monotonic()
# A MASK is uniform among expressed genes. Integrate over that choice exactly:
# every cell contributes total weight one, regardless of its number of genes.
for offset in range(0,N,256):
 rows=train.__getitems__(indices[offset:offset+256]);encoded=[];weights=[]
 for cell in rows:
  ids=cell['gene_ids'].numpy()-2;c=cell['counts'].numpy()
  encoded.append(bins(c)*V+ids);weights.append(np.full(len(ids),1/len(ids)))
 freq+=np.bincount(np.concatenate(encoded),weights=np.concatenate(weights),minlength=B*V).reshape(B,V)
 if offset%5120==0:print('training cells',offset,flush=True)
np.savez_compressed('/opt/dlami/nvme/andre/count-baseline-frequencies.npz',freq=freq)
prior=(freq.sum(0)+.01)/(freq.sum()+.01*V)
val=ScBaseCountDataset(root,'val');vidx=random.Random(43).sample(range(len(val)),6400)
loader=DataLoader(Subset(val,vidx),batch_size=128,collate_fn=partial(collate_fn,length=512),num_workers=0,generator=torch.Generator().manual_seed(43))
torch.manual_seed(43);ys=[];bs=[];visible=[]
for batch in loader:
 ys.append(batch['targets'].numpy()-2)
 for row,target in zip(batch['gene_ids'].numpy(),batch['targets'].numpy()):
  ids=row[row>=2]-2
  assert target-2 not in ids
  visible.append(ids)
 c=batch['counts'][torch.arange(len(batch['targets'])),batch['mask_positions']].numpy();bs.append(bins(c))
y=np.concatenate(ys);b=np.concatenate(bs)
report={'train_cells':N,'validation_cells':len(y),'fit_split':'train','validation_seed':43,
 'fit_method':'Each training cell contributes 1/number_of_expressed_genes per gene, integrating over the uniform MASK choice.',
 'count_bins':'Exact counts 1..32; powers-of-two bins above 32, capped at 100000.',
 'unconditional_loss':float(-np.log(prior[y]).mean()),'unconditional_accuracy':float((prior.argmax()==y).mean()),'count_only':[]}
for strength in [1.,10.,100.,1000.]:
 probs=(freq+strength*prior[None,:])/(freq.sum(1,keepdims=True)+strength)
 report['count_only'].append({'prior_smoothing_cells_per_bin':strength,'loss':float(-np.log(probs[b,y]).mean()),'accuracy':float((probs.argmax(1)[b]==y).mean())})
report['excluding_visible_genes']=[]
for strength in [None,1.,10.,100.,1000.]:
 probs=np.repeat(prior[None,:],B,axis=0) if strength is None else (freq+strength*prior[None,:])/(freq.sum(1,keepdims=True)+strength)
 losses=[];correct=0
 for i,ids in enumerate(visible):
  row=probs[b[i]].copy();row[ids]=0.;mass=row.sum()
  losses.append(-np.log(row[y[i]]/mass));correct+=row.argmax()==y[i]
 report['excluding_visible_genes'].append({'prior_smoothing_cells_per_bin':strength,'loss':float(np.mean(losses)),'accuracy':float(correct/len(y))})
report['seconds']=time.monotonic()-start
p=Path('/opt/dlami/nvme/andre/count-exclusion-baseline.json');p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
