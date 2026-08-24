# Program plan: expand → distill → SFT → RL on τ-voice

Status: **draft, 2026-08-24.** Assessment of a four-stage proposal, with the corrections the
evidence forces. Not yet agreed.

**Relationship to the existing plan.** `TAU_VOICE_SFT_PLAN.md` (agreed 2026-08-14) remains the
record for the eval harness (its phase 0), the three-arm experimental design, and risks 1–11. This
document *extends* it with stages 1 and 4 and revises the ordering; it does not replace it. The
three-arm design (A baseline / B upper bound / C cross-domain, with model selection on a **fourth**
held-out domain) is inherited unchanged and applies to the RL stage too — an RL policy selected on
the test domains is not a cross-domain result.

Every claim here carries a receipt: a file:line, an episode id, or a measured number. Do not add
claims without one.

---

## 0. The proposal, and the verdict

As stated:

1. Expand τ-voice with more domains.
2. Collect successful trajectories by running frontier models (possibly very strong ones) on the
   expanded domains.
3. SFT the NeMo 11B on those trajectories.
4. RL the SFT'd model in the τ-voice environment using the final reward (success / not success).

| stage | verdict | what changes |
|---|---|---|
| 1 — expand domains | **right, wrong position** | It is a *stage-4* prerequisite (RL generalization), not a stage-2 one. Do it after stage 3's decision gate. |
| 2 — collect | **supported** | Yield arithmetic is tight; teacher strength is the highest-leverage knob, not selection tuning. |
| 3 — SFT | **supported** | Blocked on one prompt-alignment fix (§2.3). Expect format modes to move first. |
| 4 — RL | **not as specified** | The benchmark's final reward is the one instrument we have *proven* invalid. Substitute a shaped, action-match reward (§3). |

Plus **two prerequisites nobody has costed** (§4): mode H, and rollout throughput. The second is
the largest cost item in the whole program and it currently sits behind a one-line dispatch.

**Overall: the core bet is supported.** The bet is that τ-voice failure is a specification and
coverage problem rather than a capability wall, and §1 is the evidence for it. Fix stage 4's reward
and price the two prerequisites and this is a sound program.

---

## 1. The bet, and why the evidence supports it

The model card reports FDB-v3 Tool Selection **82.5%**, argument accuracy **44.2%**, Pass@1 **33%**.
On τ-voice, arm A scored **0/14** real task completions (`NEMO_FAILURE_MODES.md` §1). The apparent
contradiction dissolves on inspection, and what it leaves behind is a *mechanically identifiable*
gap rather than a wall:

* **82.5% is a tool-*name* metric.** Mean F1 of multiset recall/precision over names, pure set
  arithmetic, no judge, arguments never inspected (`FDB_V3_REPRODUCTION.md:102`). The card's own
  argument number is 44.2% — the model gets the argument wrong more often than right, on the
  benchmark it is said to pass.
* **The same defect is visible inside FDB-v3.** Container run, `ecommerce_01`: user spells "a B C
  one two three", model calls `track_order({"order_id": "A-BC123"})`. That is mode A/B.
* **It is not perception.** An independent ASR recovers **4/4** spelled ids from the same waveform
  that produced the model's `SIN-555` (`scripts/mode_a_probe.py`). The audio/codec explanation is
  retracted. The value is lost on the way into the argument slot.
* **The function-calling channel has never been trained in our finetune.** The synthetic shards in
  `data/voicechat/synth_train` contain **no tool calls**: `function_loss` flat at ~0.001,
  `val_txt_bleu_tool_call` 0. Stage 3 starts from zero on exactly the channel that matters.

Four mechanisms turn 44% per-argument into 0% per-episode. They are also the four things any
version of this program has to attack:

1. **Compounding.** τ-voice needs a chain of correct calls carrying state (identify → fetch →
   look up → mutate). At ~0.44/argument, three chained calls is ~0.09.
2. **A wrong argument is free in FDB-v3 and terminal in τ-voice.** FDB-v3's mock registry has
   **one error path across 16 functions**, and it fires only for an unknown function *name*
   (`Full-Duplex-Bench/v3/mock_apis.py:77`). Every well-formed call to a real tool succeeds however
   wrong the argument. In τ-voice a wrong id returns `is_error: true`, recovery is required (mode G:
   the model reissues verbatim), and consecutive errors end the episode **unscored** (§2.2). That
   cap killed **22 of 64** episodes.
3. **The metric rewards what τ-voice punishes.** Name-F1 rewards eagerness — our own path
   comparison shows the two inference paths sit at different points on an eagerness/caution
   trade-off and reverse order once arguments are scored. FDB-v3's instructions say "Execute the
   tool unconditionally!" and "DO NOT ASK CLARIFYING QUESTIONS" (`FDB_V3_REPRODUCTION.md:199`).
   τ-voice's most damaging mode is C — fabricating a format-plausible id instead of asking, 5/14.
   Same disposition, opposite sign.
4. **Mode H.** The container refuses τ-voice domain policies and loops the refusal while running
   FDB-v3 prompts fine. Unexplained. See §4.1.

---

## 2. Stage by stage

### 2.1 Stage 1 — expand domains: right, but it is a stage-4 prerequisite

**Not the SFT bottleneck.** We have **16 domains / 4,923 tasks** and have measured 14 episodes.
Telecom alone has 2,285 tasks and is **entirely unmeasured on both arms**. What limits SFT is
behaviour *coverage*, not task diversity: the current corpus yields **one** error-recovery span
(§2.3). More domains do not produce error-recovery demonstrations; raising `MAX_ERRORS` and choosing
tasks that fail interestingly does.

**It is the RL bottleneck.** 16 distinct tool schemas and policies is very few environments for
policy-gradient training, and an RL policy will overfit them fast — memorizing 16 policies is a
much easier objective than learning to read an unseen one, and the reward cannot tell the two
apart. Domain count is what buys generalization in stage 4.

**Therefore: do it after stage 3's decision gate.** Domain authoring is the most expensive effort in
the program (`TAU_VOICE_SFT_PLAN.md` §6: "the binding constraint is task authoring, not episode
collection"). Spending it before knowing whether SFT moves argument accuracy puts the largest cost
on the least-validated assumption. Risk 7 of the old register also applies: stylistically cloned
domains give optimistically high transfer numbers.

### 2.2 Stage 2 — collection: supported, and the lever is teacher strength

The pipeline is built and pushed (`tau-voice-2` `64fbeca`, runbook `350c504`). Yield arithmetic from
the one pass we have measured:

| | measured |
|---|---|
| episodes collected | 64 |
| usable after selection | **6 (9%)** |
| never scored (error cap / max steps) | 43 of 58 rejects |
| zero tool calls | 23 |
| §6 coverage | p1: 2 ep / 4 spans · p2: 2 ep / 2 spans · **p3: 1 ep / 1 span** |

Even assuming the fixes triple yield, the default 322-task plan gives ~80 episodes. Thousands of
SFT episodes requires thousands of task-trials, which runs into §4.2.

**The highest-leverage single change is the teacher, not the selection threshold.** Gemini Live is
31% published (τ-Voice paper Table 6, and the *lowest* of the three providers there); our control
measured 37.5% at n=8. Sierra's blog puts `grok-voice-think-fast-1.0` (Apr 2026) at **67%**. A
teacher at 67% roughly doubles both yield and per-episode quality for no extra engineering. The
user's instinct to use "super powerful" teachers is correct and should be acted on before any
threshold tuning.

Caveat to price: a much stronger teacher widens the distillation gap. Unknown sign for an 11B
student; worth one small A/B (same tasks, two teachers, same SFT recipe) rather than an assumption.

### 2.3 Stage 3 — SFT: supported, one hard prerequisite

Expected movement, in order:

| modes | expectation |
|---|---|
| F (invented names), B (wrong slot), A-as-copy | **first and largest** — pure format/copy behaviour, and the FC channel starts untrained |
| D (nested arguments) | improves; true rate unknown (1/14, undertested) |
| G (error recovery) | **only if the corpus contains the spans** |
| C (invent vs ask), E (act at all) | least — judgement, not format |

Two things gate it:

1. **Prompt alignment — must be fixed before training, not after.** Because our cuts set
   `custom['system_prompt']`, `augment_fc_system_prompt` is a **no-op**: `collate_system_prompt`
   (`s2s_dataset.py:2118`) resolves the prompt from the cut on its **first** branch (`:2149`) and
   only augments on the following `elif` (`:2151`). The model therefore trains on the raw
   tools-only prompt with **no `<TOOLCALL>` scaffold**. If inference adds a scaffold, we have
   rebuilt the train/test mismatch at a new offset. Decide the prompt once and assert it is
   byte-identical on both sides.
2. **Priority 3 gates everything downstream.** If the model cannot recover from a wrong call it hits
   the error cap, and the episode is then **never scored** — so gains in A/B/F do not appear in any
   aggregate. `build_sft_dataset.sh` fails the build on a zero-span priority for this reason.

**Ceiling: behaviour cloning inherits the teacher.** SFT alone cannot exceed ~31–37% by imitation,
and realistically lands well below. That is the correct argument for stage 4 existing — so the
four-stage shape is justified. It is not an argument for stage 4 as specified.

Also price, from the old plan's risk 3: the agent audio in these cuts is the *teacher's* speech.
Training the 11B's decoder toward another model's voice and turn-timing risks speech quality and the
scored duplex metrics (`L_R`, `L_Y`, `R_Y`, `I_A`). Decide explicitly whether to down-weight the
audio loss relative to text and function rather than discovering it in the samples.

### 2.4 Stage 4 — RL: the reward must be substituted

**The problem is not that final-reward RL is weak here. It is that we have receipts showing it would
train in a defect we are trying to remove.**

* **§2.1 — inaction scores 1.0.** `airline__9`: reward **1.0**, **zero tool calls**, customer
  quitting mid-complaint ("I've given you my user ID three times now"). The only expected action is
  a read, so the correct final DB state is the unchanged one — exactly what an agent that does
  nothing leaves behind. `db_check` passes; COMMUNICATE returns 1.0 via "No communicate_info to
  evaluate". `action_match` is false and does not count toward reward. RL reliably finds hacks of
  this shape, and here we do not have to speculate that one exists — we have the episode. On the
  read-only subset, "say nothing, call nothing" is a high-value policy. **That is mode E**, 5/14 of
  the current failures.
* **§2.3 — the gradient is wrong in the other direction too.** `airline__3` matched **both**
  expected actions, including a `user_id` read off a spelled-out id — the exact §6 priority 1 target
  — and scored **0.0** for escalating instead of answering.
* **§2.2 — false zeros on nearly half the rollouts.** Error-capped and max-steps episodes are never
  scored: `reward_breakdown`, `db_check`, `nl_assertions`, `action_checks` all `null`, reward
  defaults to 0.0. Of the 50 episodes carrying a `reward_info`, **29 were never actually scored**
  (22 `too_many_errors` + 7 `max_steps`). An episode that died at 46 s is indistinguishable from one
  that ran to completion and failed — and the signal is silent on error recovery, the bottleneck
  behaviour.

This is a change to one design decision, not to the plan.

---

## 3. The reward you can actually train on

Do not use `reward` from the results file. Build the RL reward out of the instrument we already
trust — `tau-voice-2/scripts/score_voice_run.py` action match, which is reward-blind by
construction and is what §3's mode readouts are computed from.

**Terminal component.** Fraction of expected actions matched, not the benchmark's 0/1. Already
computed per episode; the existing `--min_action_match` selection threshold uses the same quantity.

**Shaping, per call.** The four things that compound (§1.1), each individually observable in the
tick data:

| signal | source |
|---|---|
| tool name is in the domain schema | `data/tool_schemas.json` — already required by the SFT build |
| required arguments present, schema-valid | same sidecar |
| argument value matches the grounded truth | expected-action list |
| after `is_error: true`: spoke **and** changed the call | `select_training_episodes.py` p3 detector, already written |

The p3 detector is worth reusing verbatim: it requires **both** a speech act and a changed call, so
a verbatim retry (mode G) scores zero rather than being credited as a recovery attempt.

**Batch hygiene, not optional.**

* **Filter, do not zero.** Drop rollouts where `reward_breakdown is None` from the batch. Feeding
  them as 0.0 teaches the model that being cut off is equivalent to failing, which is the opposite
  of the lesson.
* **Never `reward_info is not None`** as the scored/unscored test — it misclassifies all 29
  unscored episodes. `reward_breakdown` is the discriminator (verified: null for exactly the 29 that
  ended `too_many_errors`/`max_steps`, set for all 21 that ended `user_stop`/`agent_stop`, no
  overlap). Note also that `action_checks` is JSON `null`, not `[]`, on those episodes, so
  `.get(k, [])` returns `None` and iterating raises.
* **Penalize inaction explicitly.** Because §2.1 is a property of the environment's scoring and not
  of our reward, a zero-tool-call episode on a read-only task must be given negative or zero value
  by *our* reward even though the environment would pay 1.0.
* **Keep the duplex metrics in the objective or in a constraint.** `L_R`, `L_Y`, `R_Y`, `I_A` are
  scored by the benchmark and are trivially gamed by a policy that stops talking. A reward that only
  counts tool correctness will silently trade them away.

**Report the benchmark's own reward alongside, never optimize it.** It is the number the outside
world compares, and §2.1 means an improvement in mode E can register as a *regression* in it.

---

## 4. Two prerequisites, currently unpriced

### 4.1 Mode H blocks stage 4 entirely

The container refuses τ-voice domain policies and loops the refusal, while running FDB-v3 prompts
fine (`NEMO_FAILURE_MODES.md` §3 mode H). If it is unresolved you cannot run RL rollouts in the
container at all, and nothing suggests it is a weights problem — so an SFT'd checkpoint in the same
container may refuse identically. Already excluded: an appended authorization instruction (arm A″,
no effect), and the timeout/truncation gates (all four cleared or identified).

Three cheap tests, in order of explanatory power (hours, not days):

| test | isolates | cost |
|---|---|---|
| one episode with `USE_JINJA_TEMPLATE_PROMPT=0` | prompt template vs policy content | 1 episode + server restart |
| send an FDB-v3 prompt to *this* server | server config vs prompt | no TTS, no episode |
| truncate a τ-voice policy to ~2k chars | prompt length vs subject matter | 1 episode |

Run these in parallel with collection. They are the cheapest remaining information in the program.

### 4.2 Rollout throughput is the largest cost in the program

The user simulator's TTS is **hard-bound to ElevenLabs**: `src/tau2/voice/synthesis/synthesize.py:22`
is the only dispatch branch and `cli.py:66` lists one supported provider. Measured on our key:
**2 concurrent requests**, 429 beyond that; and retries cannot fix a concurrency ceiling — a request
that opens against a saturated limit burns its retries against a wall that only clears when another
episode finishes, so it fails permanently rather than waiting.

At 2 concurrent and 3–5 min/episode:

| | |
|---|---|
| throughput | **24–40 episodes/hour** |
| per day, flat out | 576–960 |
| 10⁴ rollouts | **~11–17 days of pure rollout wall-clock** |

Real RL runs want more than 10⁴. For comparison, a text τ-bench RL loop does thousands of episodes
per hour. This is a two-orders-of-magnitude throughput deficit, and it is the difference between
stage 4 being a multi-week experiment and a multi-month one.

This is already registered as risk 10 in `TAU_VOICE_SFT_PLAN.md` ("a local TTS behind
`synthesize.py:22`") where it materialized as a *quota* problem. For RL it returns as a *throughput*
problem, which a paid key does not fix. **Making TTS pluggable and running a local voice for
training rollouts is the highest-leverage engineering work available.**

Constraint on that fix: changing the user's voice changes the acoustics the model's front end sees,
so **eval runs must keep ElevenLabs and the stock/Sierra voice ids** for comparability. Local TTS is
for training rollouts only, and any arm using it must say so.

---

## 5. Revised order of work

Decision gates matter more than the order — the point is not to spend stage-1 effort before stage 3
has reported.

**Phase P — prerequisites (days, all parallel, all cheap).**
1. Mode H: the three tests in §4.1.
2. Prompt alignment: one prompt, asserted byte-identical in training and inference (§2.3).
3. TTS made pluggable behind `synthesize.py:22`; local voice validated for rollouts only.

**Phase C — collect (existing 16 domains).** Strongest available audio-native teacher. Gate on the
coverage table, not the cut count. Run the teacher A/B (§2.2) inside this phase.

**Phase S — SFT, and the program's decision gate.** Read argument accuracy and the per-mode
readouts, not reward. **If argument accuracy does not move, more domains and RL will not save it** —
stop and re-diagnose instead of proceeding.

**Phase D — expand domains,** for RL generalization, with deliberate stylistic diversity (old risk
7). Only after Phase S passes.

**Phase R — RL** on the §3 reward. Never the benchmark's final reward.

---

## 6. Success criteria, and what would falsify this

Inherits the old plan's three-arm design and its secondary metrics. Additions:

**Primary gate at Phase S:** FDB-v3-style argument accuracy on τ-voice tool calls, and per-mode
counts A/B/C/D/E/F/G, arm C vs arm A. Task success is reported but is *not* the gate, for §2
reasons.

**Falsifiers — state these before running, not after:**

* Argument accuracy flat after SFT on a corpus with non-zero p1 coverage → the defect is not
  copy-fidelity and mode A is an encoder problem after all. Kills the cheap path; §7 readouts 1–3
  are the tiebreaker and should ideally run *before* Phase S.
* Mode H reproduces with a 2k-char policy and with `USE_JINJA_TEMPLATE_PROMPT=0` → prompt capacity
  or subject-matter refusal is structural, and the container path is unusable for RL regardless of
  training.
* Coverage report still shows p3 = 0 spans after a large collection with `MAX_ERRORS=25` → error
  recovery is not demonstrable by collection and needs synthetic construction instead.
* RL improves our shaped reward while the benchmark's own reward and the duplex metrics fall →
  we are hacking our own reward; add the missing constraint before continuing.

---

## 7. Risk register (additions to `TAU_VOICE_SFT_PLAN.md` §8)

| # | risk | severity | mitigation |
|---|---|---|---|
| 12 | **Final-reward RL trains in mode E.** §2.1 pays 1.0 for zero tool calls on read-only tasks; RL will find it. | **blocking** | §3 shaped reward; explicit inaction penalty |
| 13 | **False zeros on ~half of rollouts.** 29 of 50 reward-carrying episodes were never scored; reward defaults to 0.0. | **blocking** | filter on `reward_breakdown`, do not zero |
| 14 | **Rollout throughput.** 24–40 episodes/hour, TTS-bound. 10⁴ rollouts ≈ 11–17 days. | **high** | pluggable local TTS for rollouts; eval keeps ElevenLabs |
| 15 | **Mode H makes the container unusable for RL.** | **high** | §4.1 three tests, before Phase R |
| 16 | **RL overfits 16 policies** instead of learning to read an unseen one; reward cannot distinguish. | high | Phase D before Phase R; hold out domains, not tasks |
| 17 | **Train/inference prompt mismatch survives SFT** (trap 3). | high | assert byte-identical prompts, §2.3 |
| 18 | **Cloning the teacher's voice/timing** degrades the scored duplex metrics. | medium | down-weight audio loss; track `L_R`/`L_Y`/`R_Y`/`I_A` per checkpoint |
| 19 | **Distillation gap** from a much stronger teacher. | medium | two-teacher A/B in Phase C |
| 20 | Domain authoring cost spent before the Phase S gate. | medium | ordering in §5 |

---

## 8. Open questions

1. **Encoder vs copy for mode A** (old plan §7 readouts 1–3). This should run *before* Phase S — it
   decides whether copy-fidelity supervision is the fix or whether more spelled-id audio is needed,
   and the stimulus is free (τ-voice persisted `both.wav` per episode).
2. **Is an on-policy RL loop affordable at all in this modality**, or is the realistic version
   offline/iterated-SFT (rejection sampling on the shaped reward)? Iterated SFT needs no serving
   loop and reuses the whole Phase C pipeline; it is the fallback if §4.2 cannot be fixed, and it
   should be costed as an alternative rather than a consolation.
3. **Does mode C survive when the id is supplied cleanly** — grounding failure or "always fill the
   slot" policy failure? Changes whether stage 3 or stage 4 is the right tool for it.
4. **Telecom is entirely unmeasured** (2,285 tasks, 0 episodes on either arm). Mode proportions may
   not hold there, and it is 46% of the task pool.
5. Which teacher, concretely, and is it reachable from the collection machine's gateway with a
   **bidirectional audio streaming** endpoint? A text-completions proxy cannot serve stage 2.

---

## 9. Pointers

| what | where |
|---|---|
| failure modes, reward pathologies, §6 priorities | `NEMO_FAILURE_MODES.md` |
| FDB-v3 metric definitions and the two-path comparison | `FDB_V3_REPRODUCTION.md` |
| harness, three-arm design, risks 1–11 | `TAU_VOICE_SFT_PLAN.md` |
| collection + data prep, how to run it | `tau-voice-2/user_docs/sft_data_collection_runbook.md` |
| the data contract and its seven consumer traps | `tau-voice-2/user_docs/training_data_generation.md` |
| reward-blind scoring instrument | `tau-voice-2/scripts/score_voice_run.py` |
