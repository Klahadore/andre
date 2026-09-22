"""Integration checks using real LMDB records and a smaller version of Andre."""
import json
from functools import partial
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import lmdb
import numpy as np
import torch
from torch.utils.data import DataLoader

import dataset
import model as model_module
import train


class TrainingLoopTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        shard = self.data / "shards" / "tiny"
        shard.mkdir(parents=True)
        records = [
            dataset.encode_cell(np.array([2, 10, 36602]), np.array([1., 5., 213654.])),
            dataset.encode_cell(np.array([3]), np.array([7.])),
            dataset.encode_cell(np.array([4, 12]), np.array([2., 3.])),
        ]
        with lmdb.open(str(shard), map_size=1024 * 1024) as env:
            with env.begin(write=True) as txn:
                for split in ("train", "val", "test"):
                    txn.put(dataset.cell_key(split, 0), dataset.pack_cells(records))
        (self.data / "catalog.json").write_text(json.dumps({
            "format_version": dataset.FORMAT_VERSION, "vocab_size": 36603,
            "shards": [{"path": "shards/tiny", "counts": {s: 3 for s in ("train", "val", "test")}}],
        }))
        # Preserve the real model's vocabulary, count cap, masks, and head while
        # keeping tests fast. Production model.py is never changed.
        self.width = patch.object(model_module, "hidden_dim", 16)
        self.depth = patch.object(model_module, "num_transformer_layers", 1)
        self.width.start()
        self.depth.start()

    def tearDown(self):
        dataset.close_environments()
        self.width.stop()
        self.depth.stop()
        self.tmp.cleanup()

    def command(self, out, steps, workers=0):
        return ["--data-root", str(self.data), "--out", str(out),
                "--steps", str(steps), "--batch-size", "2", "--context-length", "4",
                "--workers", str(workers), "--val-batches", "2", "--eval-every", "2",
                "--log-every", "1", "--warmup-steps", "2", "--device", "cpu"]

    def test_training_checkpoint_resume_and_offline_wandb(self):
        out = self.root / "run"
        train.main(self.command(out, 3) + ["--wandb-mode", "offline"])
        checkpoint = torch.load(out / "last.pt", weights_only=True)
        self.assertEqual(checkpoint["step"], 3)
        self.assertEqual(checkpoint["epoch"], 1)  # Tiny loader exhausted and reshuffled.
        self.assertTrue((out / "best.pt").is_file())
        self.assertTrue(list((out / "wandb").glob("offline-run-*")))
        metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([m["step"] for m in metrics], [1, 2, 3])
        self.assertEqual(metrics[-1]["val/cells"], 3)  # Includes short final batch.
        self.assertTrue(np.isfinite(metrics[-1]["val/loss"]))
        self.assertAlmostEqual(metrics[0]["train/lr"], 0.00015)
        self.assertAlmostEqual(metrics[-1]["train/lr"], 0.0003)
        self.assertTrue(all(s["step"].item() == 3 for s in checkpoint["optimizer"]["state"].values()))

        train.main(self.command(out, 4) + ["--resume", str(out / "last.pt")])
        resumed = torch.load(out / "last.pt", weights_only=True)
        self.assertEqual(resumed["step"], 4)
        self.assertEqual(resumed["epoch"], 2)
        self.assertTrue(all(s["step"].item() == 4 for s in resumed["optimizer"]["state"].values()))
        self.assertFalse(torch.equal(checkpoint["model"]["out_layer.weight"], resumed["model"]["out_layer.weight"]))

    def test_validation_repeats_and_preserves_training_rng(self):
        data = dataset.ScBaseCountDataset(self.data, "val")
        loader = DataLoader(data, batch_size=2, collate_fn=partial(dataset.collate_fn, length=4))
        model = model_module.Andre()
        state = torch.get_rng_state().clone()
        first = train.evaluate(model, loader, torch.device("cpu"), False, 77)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        second = train.evaluate(model, loader, torch.device("cpu"), False, 77)
        self.assertEqual(first, second)
        self.assertTrue(model.training)

    def test_spawn_workers_and_short_final_run(self):
        out = self.root / "workers"
        train.main(self.command(out, 1, workers=2))
        checkpoint = torch.load(out / "last.pt", weights_only=True)
        self.assertEqual(checkpoint["step"], 1)

    def test_unfinished_dataset_is_rejected(self):
        args = self.command(self.root / "incomplete-run", 1)
        args[1] = str(self.root / "not-built")
        with self.assertRaisesRegex(FileNotFoundError, "catalog.json"):
            train.main(args)


if __name__ == "__main__":
    unittest.main()
