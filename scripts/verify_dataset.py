"""Check every published shard and independently compare sampled source cells."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time

import anndata as ad
import h5py
import lmdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_codec import CELLS_PER_RECORD, SPLITS, cell_key, unpack_cell
from scripts.build_dataset import notebook_splits


def verify_shard(root, shard, vocab_size, vocab_hash):
    path = root / shard['path']
    meta = json.loads((path/'metadata.json').read_text())
    if meta['counts'] != shard['counts'] or meta['signature']['vocab_sha256'] != vocab_hash:
        raise ValueError(f'Metadata mismatch: {path}')
    if (path/'data.mdb').stat().st_size != meta['database_bytes']:
        raise ValueError(f'Database size mismatch: {path}')
    with lmdb.open(str(path),readonly=True,lock=False,readahead=False,max_spare_txns=0) as env:
        expected = sum((n+CELLS_PER_RECORD-1)//CELLS_PER_RECORD for n in meta['counts'].values())
        with env.begin() as txn:
            if txn.stat()['entries'] != expected:
                raise ValueError(f'Packed-record count mismatch: {path}')
            for split, count in meta['counts'].items():
                for index in sorted({0,count//2,count-1}) if count else []:
                    sample = unpack_cell(txn.get(cell_key(split,index//CELLS_PER_RECORD)),index%CELLS_PER_RECORD)
                    ids, values = sample['gene_ids'],sample['counts']
                    if not (len(ids) and np.all((ids>=2)&(ids<vocab_size))
                            and np.all(ids[1:]>ids[:-1]) and np.all(np.isfinite(values))
                            and np.all(values>0)):
                        raise ValueError(f'Invalid cell: {path}/{split}/{index}')
    return meta['nnz']


def compare_source(root, data_root, shard, vocab):
    accession=Path(shard['path']).name
    # Use AnnData/h5py's standard decoder as an independent reference.
    with h5py.File(data_root/'h5ad'/f'{accession}.h5ad') as f:
        matrix=ad.io.read_elem(f['X']).tocsr()
        matrix.sum_duplicates();matrix.eliminate_zeros()
        obs=f['obs']; var=f['var']
        keep=np.flatnonzero((obs['gene_count_Unique'][:]>=300)&(obs['umi_count_Unique'][:]>=500))
        barcodes=np.asarray(ad.io.read_elem(obs[obs.attrs['_index']]),dtype=str)
        names=np.asarray(ad.io.read_elem(var[var.attrs['_index']]),dtype=str)
        tokens=np.array([vocab[g] for g in names])
        split_ids=notebook_splits(accession,barcodes[keep])
    checked=0
    max_row=int(np.searchsorted(matrix.indptr, int(matrix.data.argmax()),side='right')-1) if matrix.nnz else -1
    with lmdb.open(str(root/shard['path']),readonly=True,lock=False) as env:
        with env.begin() as txn:
            for sid,split in enumerate(SPLITS):
                rows=keep[split_ids==sid]
                if len(rows)!=shard['counts'][split]:
                    raise ValueError(f'Source QC/split mismatch: {accession}')
                choices={0,len(rows)//2,len(rows)-1} if len(rows) else set()
                choices.update(np.flatnonzero(rows==max_row).tolist())
                for index in sorted(choices):
                    sample=unpack_cell(txn.get(cell_key(split,index//CELLS_PER_RECORD)),index%CELLS_PER_RECORD)
                    a,b=matrix.indptr[rows[index]:rows[index]+2]
                    expected_ids=tokens[matrix.indices[a:b]]
                    order=np.argsort(expected_ids)
                    np.testing.assert_array_equal(sample['gene_ids'],expected_ids[order])
                    np.testing.assert_array_equal(sample['counts'],matrix.data[a:b][order])
                    checked+=1
    return checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--source-samples',type=int,default=12)
    parser.add_argument('--expected-cells',type=int)
    parser.add_argument('--expected-nnz',type=int)
    args=parser.parse_args()
    catalog=json.loads((args.root/'catalog.json').read_text())
    shards=catalog['shards']
    counts={split:sum(s['counts'][split] for s in shards) for split in SPLITS}
    if counts!=catalog['counts']:
        raise ValueError('Catalog totals disagree')
    cells=sum(counts.values())
    if args.expected_cells is not None and cells!=args.expected_cells:
        raise ValueError(f'Expected {args.expected_cells} cells, got {cells}')
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as pool:
        nnz=sum(pool.map(lambda s:verify_shard(args.root,s,catalog['vocab_size'],catalog['vocab_sha256']),shards))
    if args.expected_nnz is not None and nnz!=args.expected_nnz:
        raise ValueError(f'Expected {args.expected_nnz} nonzeros, got {nnz}')
    source_cells=0
    if args.data_root and args.source_samples:
        vocab=json.loads((args.root/'vocabulary.json').read_text())
        ordered=sorted(shards,key=lambda s:s['database_bytes'])
        selected=[ordered[i] for i in sorted(set(np.linspace(0,len(ordered)-1,min(args.source_samples,len(ordered)),dtype=int)))]
        # Include the source containing the independently audited maximum count.
        maximum=next((s for s in shards if Path(s['path']).name=='SRX18243180'),None)
        if maximum is not None and maximum not in selected:
            selected.append(maximum)
        for shard in selected:
            source_cells+=compare_source(args.root,args.data_root,shard,vocab)
    result={'verified_shards':len(shards),'cells':cells,'counts':counts,'nnz':nnz,
            'source_cells_compared':source_cells,'elapsed_seconds':round(time.monotonic()-started,2)}
    (args.root/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
