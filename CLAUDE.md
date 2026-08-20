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

**Also ask before requesting a new allocation.** Approval to run something is not approval to
allocate for it. State the exact shape (nodes / time / partition) and wait for an answer.

Reuse a live allocation instead of asking for a second one:

```bash
srun --overlap --jobid=<id> --nodes=1 --ntasks=1 ...
```

When a job *is* already gone, say so plainly with the `sacct` evidence and present the options —
don't quietly re-allocate.

## Ask before `git push`

Commit freely; never push without asking. Push over SSH, not HTTPS. Verify a push landed with
`git ls-remote origin` rather than trusting the absence of an error.

## Secrets

`tau-voice-2/.env` holds the ElevenLabs key and is gitignored (`.gitignore:153`). Never stage it and
never print the key value — length, prefix, charset and quota numbers only.
