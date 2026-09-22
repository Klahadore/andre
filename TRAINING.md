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

This is a starting configuration, not a tuned recipe. One step is one optimizer
update. By default it uses one batch. At batch size 16, 1,000 full steps process 16,000
cell visits, not an epoch through the entire dataset. Increase `--steps` for
a longer experiment. Reduce `--batch-size` if GPU memory is insufficient.

CUDA uses bfloat16 autocast when supported (including H100); model parameters
and AdamW state remain float32. CPU uses float32. You can request
`--precision float32` to disable autocast. No training is launched just by
importing this script.

## Change model width and keep the same effective batch

Use `--hidden-dim` to select model width without editing `model.py`. It must be
a positive multiple of 8, the number of attention heads; depth stays at 30 layers.
Direct Python callers can use `Andre(width=768)`; `Andre()` still defaults to 512.

`--batch-size` is the number of cells processed at once. `--grad-accum-steps`
is the number of batches processed before one weight update. Their product is
the effective batch size: 64 cells times 2 batches gives 128 cells per update,
matching a width-512 run that uses a batch of 128 with no accumulation.

For the width-768 run with approximately 10 billion input positions:

```bash
uv run train.py \
  --data-root /opt/dlami/nvme/andre/lmdb \
  --out runs/width-768-10b-tokens \
  --hidden-dim 768 \
  --batch-size 64 \
  --grad-accum-steps 2 \
  --context-length 512 \
  --steps 152588 \
  --eval-every 1000 \
  --val-batches 100 \
  --device cuda \
  --precision bfloat16 \
  --wandb-mode online \
  --run-name width-768-10b-tokens
```

This retains 6,400 validation cells (100 batches of 64), the same subset as the
width-512 run's 50 batches of 128 when using the same seed and dataset.
Steps, warmup, evaluation and checkpoint intervals count **optimizer updates**,
not individual microbatches. Each full update presents 128 x 512 = 65,536 input
positions, including padding. The command starts fresh; it does not resume the
width-512 checkpoint.

At an epoch boundary a microbatch can be shorter, so that update can contain
fewer cells than the configured effective batch. The loop weights each loss
by its actual cell count; it neither overweights the short batch nor drops it.
W&B records the configured `effective_batch_size` and actual `train/cells_per_update`.

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
5. For each microbatch, `(loss * (microbatch_cells / update_cells)).backward()`
   adds its share of the average gradient. The weights stay unchanged between
   microbatches. With two full batches, each loss is divided by two.
6. After all microbatches, gradient clipping limits the combined norm and
   `optimizer.step()` applies one AdamW update. The next update clears the old
   gradients with `zero_grad()`.

The learning rate starts small, increases linearly for `--warmup-steps` (100
by default), then stays at `--lr` (0.0003 by default). There is no distributed
training or compilation in this introductory loop.

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
dataset and model architecture, including `--hidden-dim`. The current `--lr` and `--warmup-steps`
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
spawned workers, offline W&B logging, and accumulated gradients matching a full
batch even when the microbatches have unequal sizes. They do not benchmark H100 throughput.

API references: [W&B run modes](https://docs.wandb.ai/models/ref/python/functions/init)
and [PyTorch automatic mixed precision](https://docs.pytorch.org/docs/2.14/amp.html).
