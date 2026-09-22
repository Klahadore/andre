import sys,json,random,time,statistics
from itertools import islice
from functools import partial
from pathlib import Path
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,'/home/ubuntu/andre')
from dataset import ScBaseCountDataset,BlockShuffleSampler,collate_fn

def main():
 torch.set_num_threads(1);ds=ScBaseCountDataset('/opt/dlami/nvme/andre/lmdb','train');n=128*129;results=[]
 for name,indices in [('block',list(islice(iter(BlockShuffleSampler(ds,seed=42)),n))),('random',random.Random(42).sample(range(len(ds)),n))]:
  diversity=[len(set(ds._locate(i)[0]for i in indices[j:j+128]))for j in range(0,n,128)]
  loader=DataLoader(ds,sampler=indices,batch_size=128,collate_fn=partial(collate_fn,length=512),num_workers=8,multiprocessing_context='spawn',prefetch_factor=2)
  t=time.monotonic();it=iter(loader);next(it);startup=time.monotonic()-t;t=time.monotonic()
  cells=0
  for b in it:cells+=len(b['targets'])
  elapsed=time.monotonic()-t
  r={'sampler':name,'measured_cells':cells,'seconds':elapsed,'cells_per_second':cells/elapsed,'startup_seconds':startup,'mean_shards_per_batch':statistics.mean(diversity),'median_shards_per_batch':statistics.median(diversity),'note':'No explicit cache prewarming; concurrent GPU training may contend for CPU/IO.'};results.append(r);print(json.dumps(r),flush=True)
 Path('/opt/dlami/nvme/andre/random-loader-probe.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
