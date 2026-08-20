# Rules for working in this repo

## Never release a Slurm allocation

**Do not give up an allocated node unless Kai explicitly asks.** This cluster is contended and a
released node does not come back — on 2026-08-20, 26 of 27 GPU nodes were `alloc` with other users
already queued on `(Resources)` and `(Priority)`. Losing an allocation costs hours to days of
wall-clock and cannot be undone.

This covers every way an allocation can end, not just the obvious one:

* no `scancel` on the job
* don't exit or let an `salloc`/shell session end when that drops the job
* don't let a job die by inaction when it could be kept alive
* don't hand a node back "since we're done with it" — being done is not a reason

Killing an inner *step* is fine and is not releasing the node: `squeue -s -j <jobid>` to list steps,
then `scancel <jobid>.<step>`.

**Getting a node needs no permission — but reuse before you allocate.** Check for a live allocation
of ours first and use it if it isn't busy; only allocate a new one if there is none free. Don't ask
either way, and don't stall on it.

```bash
squeue -u "$(whoami)" -o '%.8i %.20j %.8T %.10M %R'   # our live allocations
squeue -s -j <jobid>                                  # steps running inside one -> is it busy?
srun --overlap --jobid=<jobid> --nodes=1 --ntasks=1 nvidia-smi   # are the GPUs actually free?
```

"Occupied" means something of ours is really using it — running steps, or GPUs with memory in use.
An idle allocation is the thing to grab. Reuse it with `srun --overlap --jobid=<id> ...` rather than
requesting a second node.

If nothing is free, allocate (same shape as before: 1 node, `ml.p5en.48xlarge`, `--no-shell`). Note
that the queue is often saturated, so a request may pend for a long time — report that rather than
waiting silently.

## Ask before `git push`

Commit freely; never push without asking. Push over SSH, not HTTPS. Verify a push landed with
`git ls-remote origin` rather than trusting the absence of an error.

## Secrets

`tau-voice-2/.env` holds the ElevenLabs key and is gitignored (`.gitignore:153`). Never stage it and
never print the key value — length, prefix, charset and quota numbers only.
