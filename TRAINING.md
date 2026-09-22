# Training the masked-gene model

Start with `train.py`, especially the numbered comments inside `main()`.
The model architecture stays in `model.py`; the dataset and collator stay in
`dataset.py`. This script trains on one device and does not build the dataset.

## A first run

Wait for the other agent's dataset build to publish `catalog.json`. Use the
directory containing that file as `--data-root`; this is the **LMDB output**,
not the downloaded H5AD directory. Replace the example path if the build uses
a different name.

On the H100 node:

```bash
uv run train.py \
  --data-root /opt/dlami/nvme/andre/lmdb \
  --out /opt/dlami/nvme/andre/runs/first-run \
  --steps 1000 \
  --batch-size 16 \
  --workers 4
```

This is a starting configuration, not a tuned recipe. One step is one batch
and one optimizer update. At batch size 16, 1,000 full steps process 16,000
cell visits, not an epoch through the entire dataset. Increase `--steps` for
a longer experiment. Reduce `--batch-size` if GPU memory is insufficient.

CUDA uses bfloat16 autocast when supported (including H100); model parameters
and AdamW state remain float32. CPU uses float32. You can request
`--precision float32` to disable autocast. No training is launched just by
importing this script.

## What happens in one step?

1. The loader supplies gene IDs, raw counts, an attention mask, one masked
   position per cell, and the original gene identity at that position.
2. `model(...)` returns `[batch_size, 36601]` scores. Its count embedding caps
   counts at 100,000; its Transformer ignores padding and selects the MASK
   position's hidden vector before the final linear layer.
3. `targets - 2` converts gene token IDs into prediction class indices, because
   PAD and MASK occupy input IDs 0 and 1 but are not prediction classes.
4. Cross-entropy compares the raw scores with those target classes. Do not
   apply softmax first.
5. `loss.backward()` calculates gradients. Gradient clipping limits their
   combined norm; `optimizer.step()` applies the AdamW update.
6. The next iteration clears the old gradients with `zero_grad()`.

The learning rate starts small, increases linearly for `--warmup-steps` (100
by default), then stays at `--lr` (0.0003 by default). There is no gradient
accumulation, distributed training, or compilation in this introductory loop.

## Validation and logging

Every `--eval-every` steps (100 by default), and at the final step, the script
evaluates a fixed random subset of the **val** split. It uses the same sampled
genes and MASK positions each time, with dropout and gradient recording off.
The default subset has at most `--val-batches * --batch-size` cells (800).
This is a quick progress check, not a full validation-set evaluation. The test
split is left unused. The existing dataset split is at the cell level, not a
held-out-study evaluation.

Loss and masked-gene accuracy are logged for training and validation. Accuracy
means the fraction of cells whose top-scoring gene is the hidden gene. Training
metrics average over the cells since the previous log. Learning rate, gradient
norm before clipping, and training cells/second are also logged.

W&B is disabled by default; console output and `metrics.jsonl` are always
available. For online logging, log in **on the machine doing the training**:

```bash
uv run wandb login
```

Then append these flags to the training command:

```bash
--wandb-mode online --wandb-project andre --run-name first-run
```

Use `--wandb-mode offline` to save W&B logs locally without uploading, or
`--wandb-entity YOUR_TEAM` to choose a team. The script logs metrics and config;
it does not upload the dataset or model checkpoints. Each invocation creates
a new W&B run, including when resuming training.

## Checkpoints and resume

`--out` contains:

- `last.pt`: model, optimizer, completed step, shuffle epoch, best validation
  loss, and command-line settings, saved at every validation and the end.
- `best.pt`: the same state at the best observed validation loss.
- `config.json` and `metrics.jsonl`: readable configuration and metric history.
- W&B's local files when logging is enabled.

The checkpoints include AdamW state, so they can be several GB for this model.
They are written to a temporary file and renamed into place. The script keeps
only the latest and best files. An interruption loses steps since the last
save; reduce `--eval-every` if more frequent checkpoints are needed.

To continue from step 1,000 until a total of 2,000 steps, repeat your settings
and add:

```bash
--resume /opt/dlami/nvme/andre/runs/first-run/last.pt --steps 2000
```

Resume restores the model and optimizer but starts a fresh data shuffle. It
does not restore the exact prefetched batches or RNG state. Use the same
dataset and model architecture. The current `--lr` and `--warmup-steps`
control the resumed learning-rate schedule; other optimizer settings restore
from the checkpoint. Avoid reusing an output directory for an unrelated run.

The EC2 instance's local NVMe is temporary. Copy useful checkpoints and logs to
durable storage before the instance terminates.

## Check the loop locally

```bash
uv run python -m unittest discover -s tests -p test_training_loop.py -v
```

These checks create temporary LMDB records and use a smaller version of the
same model. They cover real reader/collator integration, finite updates,
short batches, shuffle rollover, checkpoint resume, deterministic validation,
spawned workers, and offline W&B logging. They do not benchmark H100 throughput.

API references: [W&B run modes](https://docs.wandb.ai/models/ref/python/functions/init)
and [PyTorch automatic mixed precision](https://docs.pytorch.org/docs/2.14/amp.html).
