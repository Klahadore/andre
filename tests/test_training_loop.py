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
                "--hidden-dim", "16",
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

    def test_accumulation_matches_full_batch_with_unequal_microbatches(self):
        # Three cells: compare one batch of 3 against batches of 2 and 1.
        # Disable dropout so RNG differences cannot obscure gradient averaging.
        models = []

        def make_model(width):
            model = model_module.Andre(width=width)
            for layer in model.modules():
                if isinstance(layer, torch.nn.Dropout):
                    layer.p = 0
                elif isinstance(layer, torch.nn.MultiheadAttention):
                    layer.dropout = 0
            models.append(model)
            return model

        with patch.object(train, "Andre", side_effect=make_model):
            full = self.root / "full"
            train.main(self.command(full, 1) + ["--batch-size", "3", "--grad-clip", "100000"])
            accumulated = self.root / "accumulated"
            train.main(self.command(accumulated, 1) + ["--grad-accum-steps", "2", "--grad-clip", "100000"])

        for full_param, accumulated_param in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(full_param.grad, accumulated_param.grad, rtol=2e-5, atol=1e-6)
        config = json.loads((accumulated / "config.json").read_text())
        self.assertEqual(config["hidden_dim"], 16)
        self.assertEqual(config["effective_batch_size"], 4)
        metrics = json.loads((accumulated / "metrics.jsonl").read_text())
        self.assertEqual(metrics["train/cells_per_update"], 3)
        checkpoint = torch.load(accumulated / "last.pt", weights_only=True)
        self.assertTrue(all(s["step"].item() == 1 for s in checkpoint["optimizer"]["state"].values()))
        # Resume also keeps optimizer steps distinct from microbatch count.
        train.main(self.command(accumulated, 2) + ["--grad-accum-steps", "2", "--resume", str(accumulated / "last.pt")])
        checkpoint = torch.load(accumulated / "last.pt", weights_only=True)
        self.assertTrue(all(s["step"].item() == 2 for s in checkpoint["optimizer"]["state"].values()))

    def test_explicit_model_width_and_invalid_configuration(self):
        with torch.device("meta"):
            model = model_module.Andre(width=768)
        self.assertEqual(model.gene_embedding.embedding_dim, 768)
        self.assertEqual(model.transformer_layers[0].linear1.out_features, 768 * 4)
        self.assertEqual(model.out_layer.in_features, 768)
        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            model_module.Andre(width=767)
        for extra in (["--hidden-dim", "767"], ["--grad-accum-steps", "0"]):
            with self.assertRaises(SystemExit):
                train.parse_args(["--data-root", str(self.data)] + extra)

    def test_context_only_learning_on_fixed_examples(self):
        # All MASK counts are identical; only the visible gene context can
        # distinguish these examples. This catches input-independent predictors.
        torch.manual_seed(12)
        with patch.object(model_module, "num_transformer_layers", 8), patch.object(model_module, "output_dim", 8):
            model = model_module.Andre(width=16)
        for layer in model.modules():
            if isinstance(layer, torch.nn.Dropout):
                layer.p = 0
            elif isinstance(layer, torch.nn.MultiheadAttention):
                layer.dropout = 0
        ids = torch.tensor([[1, 100+i, 200, 201] for i in range(8)])
        counts = torch.ones_like(ids)
        attention = torch.ones_like(ids, dtype=torch.bool)
        positions = torch.zeros(8, dtype=torch.long)
        targets = torch.arange(8)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0)
        for _ in range(100):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(ids, counts, attention, positions), targets)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(ids, counts, attention, positions)
            self.assertLess(torch.nn.functional.cross_entropy(logits, targets).item(), .15)
            self.assertTrue(torch.equal(logits.argmax(-1), targets))
            attention[:, 1:] = False
            without_context = model(ids, counts, attention, positions)
            self.assertGreater(torch.nn.functional.cross_entropy(without_context, targets).item(), 1.)
        metrics = train.evaluate(model, [{"gene_ids": ids, "counts": counts,
            "attention_mask": torch.ones_like(attention), "mask_positions": positions,
            "targets": targets + 2}], torch.device("cpu"), False, 77)
        self.assertGreater(metrics["val/shuffled_context_gain"], 1.)

    def test_legacy_postln_checkpoint_is_rejected(self):
        out = self.root / "legacy"
        train.main(self.command(out, 1))
        checkpoint = torch.load(out / "last.pt", weights_only=True)
        checkpoint.pop("model_architecture")
        torch.save(checkpoint, out / "legacy.pt")
        with self.assertRaisesRegex(ValueError, "normalization architecture differs"):
            train.main(self.command(out, 2) + ["--resume", str(out / "legacy.pt")])

    def test_compiled_blocks_save_and_resume_without_compilation(self):
        # The eager backend exercises Dynamo's wrapper on CPU without requiring
        # a local C++ compiler. H100 kernel speed/correctness is measured separately.
        original_compile = torch.nn.Module.compile

        def compile_eager(module, **kwargs):
            return original_compile(module, backend="eager", **kwargs)

        out = self.root / "compiled"
        with patch.object(torch.nn.Module, "compile", compile_eager):
            train.main(self.command(out, 2) + ["--compile-layers"])
        checkpoint = torch.load(out / "last.pt", weights_only=True)
        self.assertTrue(checkpoint["config"]["compile_layers"])
        self.assertFalse(any("_orig_mod" in key for key in checkpoint["model"]))
        train.main(self.command(out, 3) + ["--resume", str(out / "last.pt")])
        resumed = torch.load(out / "last.pt", weights_only=True)
        self.assertEqual(resumed["step"], 3)
        self.assertTrue(all(s["step"].item() == 3 for s in resumed["optimizer"]["state"].values()))


if __name__ == "__main__":
    unittest.main()
