import json
import fcntl
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import anndata as ad
import h5py
import lmdb
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix

from dataset_codec import cell_key, encode_cell, decode_cell, unpack_cell, SPLITS
from scripts.fast_encode import prepare_cells
from scripts.fast_h5 import read_vector
from scripts.build_dataset import build_shard, load_vocab, notebook_splits


class FastBuildTests(unittest.TestCase):
    def test_compiled_bytes_match_reference(self):
        rng = np.random.default_rng(42)
        dense = rng.integers(0, 8, (73, 300)).astype(np.float32)
        dense[dense < 6] = 0
        dense[0, 5] = 213654
        dense[1, 10] = 1.5
        matrix = csr_matrix(dense)
        tokens = np.arange(2, 302, dtype=np.uint16)
        rows = np.arange(len(dense))
        raw, offsets = prepare_cells(matrix, tokens, rows)
        import zlib
        for i in rows:
            a, b = matrix.indptr[i:i+2]
            expected = zlib.decompress(encode_cell(tokens[matrix.indices[a:b]], matrix.data[a:b]))
            self.assertEqual(bytes(raw[offsets[i]:offsets[i+1]]), expected)

    def test_gzip_and_fallback_readers(self):
        with tempfile.TemporaryDirectory() as tmp:
            with h5py.File(Path(tmp)/'sample.h5', 'w') as f:
                expected = np.arange(1031, dtype=np.float32)
                for name, options in [('gzip',dict(compression='gzip',chunks=(128,))),
                                      ('shuffle',dict(compression='gzip',shuffle=True,chunks=(128,))),
                                      ('plain',{})]:
                    ds = f.create_dataset(name, data=expected, **options)
                    np.testing.assert_array_equal(read_vector(ds), expected)

    def test_parallel_cli_publication_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data'
            (data/'h5ad').mkdir(parents=True)
            (data/'metadata').mkdir()
            vocab = data/'gene_to_id.csv'
            pd.DataFrame({'token_id':range(5),'gene':['<pad>','<mask>','gA','gB','gC']}).to_csv(vocab,index=False)
            (data/'accessions.txt').write_text('SRX1\nSRX2\n')
            import duckdb
            with duckdb.connect() as db:
                db.execute("COPY (SELECT acc AS srx_accession, 'Homo sapiens' organism, '3_prime_gex' tech_10x, 'single_cell' cell_prep FROM (VALUES ('SRX1'), ('SRX2')) t(acc)) TO ? (FORMAT PARQUET)", [str(data/'metadata/sample_metadata.parquet')])
            obs = pd.DataFrame({'gene_count_Unique':[300]*83,'umi_count_Unique':[501]*82+[499]},index=[f'b{i}' for i in range(83)])
            values = np.tile(np.array([213654, 1, 3],np.float32),(83,1))
            for acc in ['SRX1','SRX2']:
                ad.AnnData(csc_matrix(values),obs=obs,var=pd.DataFrame(index=['gC','gA','gB'])).write_h5ad(data/'h5ad'/f'{acc}.h5ad')
            command=[sys.executable,'scripts/build_dataset.py','--data-root',str(data),'--vocab',str(vocab),
                     '--out',str(root/'stage'),'--publish-to',str(root/'published'),'--workers','2','--reserve-gib','0']
            subprocess.run(command, check=True, capture_output=True)
            catalog=json.loads((root/'published/catalog.json').read_text())
            self.assertEqual(sum(catalog['counts'].values()),164)
            self.assertFalse(list((root/'stage/shards').glob('SRX*')))
            for shard in catalog['shards']:
                with lmdb.open(str(root/'published'/shard['path']),readonly=True,lock=False) as env:
                    with env.begin() as txn:
                        for split in SPLITS:
                            for index in range(shard['counts'][split]):
                                sample=unpack_cell(txn.get(cell_key(split,index//32)),index%32)
                                np.testing.assert_array_equal(sample['gene_ids'],[2,3,4])
                                np.testing.assert_array_equal(sample['counts'],[1,3,213654])
            subprocess.run(command,check=True,capture_output=True)
            self.assertEqual(catalog,json.loads((root/'published/catalog.json').read_text()))
            subprocess.run([sys.executable, 'scripts/verify_dataset.py', str(root/'published'),
                            '--data-root', str(data), '--source-samples', '2',
                            '--expected-cells', '164', '--expected-nnz', '492'],
                           check=True, capture_output=True)
            with (root/'published/.build.lock').open('w') as lock:
                fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                blocked = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(blocked.returncode, 0)
                self.assertIn('BlockingIOError', blocked.stderr)

    def test_missing_vocab_has_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError,'Copy the notebook'):
            load_vocab('/does-not-exist/andre-vocab.csv')


if __name__ == '__main__':
    unittest.main()
