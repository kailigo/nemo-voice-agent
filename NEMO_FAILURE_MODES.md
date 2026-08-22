# Failure modes of NVIDIA-NemotronLabs-VoiceChat-11B on τ-voice

**Living document.** Add to it as evidence lands; do not delete superseded claims, mark them.
Every claim here must carry a receipt — an episode id and the actual call the model made.
See the changelog at the bottom.

Status as of 2026-08-21: **14 of 24 episodes of the stage-2 subset finished, real task
completion 0/14.** Six distinct failure modes plus one amplifier, all with receipts. The
Gemini Live control is in flight; on the 4 episodes both arms have now run it scores 1 real
pass to arm A's 0, and 12 of 25 expected actions to arm A's 0 of 4.

---

## 1. What was measured

| | arm A (subject) | arm B (control) |
|---|---|---|
| agent | NemotronLabs-VoiceChat-11B, research path | Gemini Live `gemini-3.1-flash-live-preview` |
| run | `stage2_subset_0821` | `gemini_baseline_0821b` |
| user simulator | `bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0` | identical |
| speech complexity | `control` | identical |
| voices | `scripts/tau2_stock_voices.env` | identical |
| episode cap | 200 s | 1200 s |
| tool-error cap | `DEFAULT_MAX_ERRORS = 10` (untouched) | identical |

Both arms use the same user LLM, the same TTS voices and the same complexity, so an
episode-level comparison is clean. The cap differs and is *not* a confound for arm A's
failures: every error-capped episode died at 46–99 s, less than half its own budget.

Score both arms with `tau-voice-2/scripts/score_voice_run.py`, never with raw reward — see §2.

---

## 2. Two ways the benchmark's zero-or-one lies

**Neither of these is a bug in τ-voice. Both must be handled before any number from this
benchmark is quoted, for either arm or for a post-SFT checkpoint.**

### 2.1 Inaction scores 1.0 on a read-only task

`airline__9` scored **reward 1.0 having made zero tool calls**, while the customer was
quitting mid-complaint ("I've given you my user ID three times now"). The mechanism:

* `reward_basis` is `[DB, COMMUNICATE]`, so `action_checks` contributes nothing.
* The task's only expected action is `search_direct_flight`, `tool_type: "read"`. The
  correct final DB state is therefore the *unchanged* one — exactly what an agent that does
  nothing leaves behind. `db_check.db_match: true`, `DB: 1.0`.
* `COMMUNICATE` returns 1.0 via *"No communicate_info to evaluate"*.
* The one check that would have caught it, `action_match`, is `false` and does not count.

Product: 1.0 for doing nothing. Aggregate reward is inflated by however many read-only
tasks the set contains, and the inflation lands precisely on the failure mode we care most
about (mode E below).

### 2.2 An error-capped episode is never evaluated at all

When an episode terminates on `too_many_errors` **or `max_steps`**, `reward_breakdown`,
`db_check` and `nl_assertions` all come back `null` and `action_checks` comes back empty.
Reward defaults to `0.0`. That is **11 of arm A's 14 episodes** — arm A has only 3 scored
episodes, so most of its "measured" zeros are unmeasured. It happens to be the right answer
here, but:

* any per-check *rate* computed over those episodes is computed over missing data, and
* an arm that aborts more episodes looks *better* on some checks purely by being absent
  from them.

The scorer prints the scored denominator next to the average for this reason.

### 2.3 …and a reward of 0.0 can sit on top of a perfect action list

The converse of §2.1, found in the control. Gemini's `airline__3` matched **both** of the
task's expected actions — `get_reservation_details {"reservation_id": "JMO1MG"}` and
`get_user_details {"user_id": "anya_garcia_5901"}`, the latter from a spelled-out id — and
scored **0.0**, because `reward_basis` is `[DB, COMMUNICATE]` and it escalated to a human
instead of answering the baggage question.

So reward and action-match are decoupled in *both* directions. **For "can this model ground
an identifier from speech", read the reward-blind action-match row, not the reward.** That is
the instrument this document uses for modes A–C. Reward answers a different question — did
the conversation end in the right world-state — and both are worth reporting.

---

## 3. The failure modes

Counts are over the 14 finished episodes. Modes co-occur; one episode can appear twice.

### Mode A — spelled-out alphanumeric IDs are mis-transcribed (4/14)

The single highest-yield mode. The user spells an identifier character by character and the
model writes down something else.

| episode | user said (spoken) | correct value | model sent |
|---|---|---|---|
| `airline__28` | "S, I, five, U, K, W" | `SI5UKW` | `SIN-555` |
| `airline__3` | "anya underscore garcia underscore five, nine, zero, one" | `anya_garcia_5901` | `JFK59001` |
| `airline__40` | "3, R, K, 2, T, 9" | `3RK2T9` | `3RGRK2T` |
| `retail__78` | "W, five, zero, five, six, five, one, nine" | `#W5056519` | `WDL500019` |

Note the error shapes: `3RK2T9` → `3RGRK2T` inserts a character and duplicates a substring
(a decoding error, not a random string); `SI5UKW` → `SIN-555` collapses letters into a
plausible *airport-code plus number* shape. `retail__78` also shows the model does not know
retail order ids carry a `#W` prefix.

**The audio is not the problem — measured 2026-08-22, `scripts/mode_a_probe.py` readout 0.**
Parakeet TDT 0.6b, run over the *same* `both.wav` user channel the model heard (cut to the
labelled turn, no re-synthesis), recovered **4 of 4** ids exactly:

| episode | expected | independent ASR heard | model sent | model edit distance |
|---|---|---|---|---|
| `airline__28` | `SI5UKW` | `SI5UKW` | `SIN-555` | 4 |
| `airline__3` | `anya_garcia_5901` | `Anya underscore Garcia underscore 5901` | `JFK59001` | 11 |
| `airline__40` | `3RK2T9` | `3RK2T9` | `3RGRK2T` | 3 |
| `retail__78` | `#W5056519` | `W5056519` | `WDL500019` | 5 |

A 0.6b ASR reads every id off 8 kHz G.711 telephony audio, and even renders them in canonical
form. So the ids are fully present in the signal and mode A is **inside our model**. Two
consequences:

* **Retract the telephony explanation.** §5.1 and the FDB-v3 comparison listed "8 kHz µ-law
  vs FDB-v3's 24 kHz clean" as a factor compounding mode A. It is not one for these four
  receipts. Bandwidth augmentation is not the SFT fix; the information was there.
* This does **not** yet separate our speech encoder from our LLM — Parakeet is not our
  encoder, so this only proves recoverability. Readouts 1–3 in the probe's docstring
  (reproduce / perceive / copy) are what localise it, and they need the 11B.

**Homophones are a genuine exception and should be split out of this mode.** On `airline__40`
the model sent `first_name: "May"` for "Mei"; Parakeet independently produced *"from May Lee
to May Garcia"* on the same audio. That error is inherent ambiguity in the signal, not our
defect, and unlike the spelled ids no amount of copy-fidelity training fixes it — only
context ("Mei" is the name already on the reservation) does.

### Mode B — the value goes into the wrong argument slot (3/14)

Separate from mis-hearing. In `airline__3` and `airline__34` the user spells a **user id**
(`anya_garcia_5901`, `yara_garcia_1905`) and the model puts its garbled version into
`reservation_id`. In `airline__28` the model takes the reservation id `SIN-555` it had
already invented and passes it as `user_id` to `get_user_details` **eight times**.

### Mode C — fabricates a format-plausible identifier instead of asking (5/14)

The most dangerous mode, because the output is well-formed and would pass any check that
only looks at shape.

| episode | what the user actually gave | model invented |
|---|---|---|
| `retail__35` | never mentioned any id; gold path is `find_user_id_by_email` | `product_id: "6086499569"` — a **correctly formatted 10-digit** product id |
| `retail__49` | name + city only | `order_1234`, then `product_1234` … `product_1241`, incrementing |
| `retail__7` | *"I don't have that with me right now. I don't remember it"* | `John Doe`, `john@example.com`, zip `12345` |
| `retail__64` | *"my name is James Sanchez… I live in Chicago… I don't remember my email"* | `James Smith`, zip `60622` (gold `60623`) |
| `airline__34` | spelled the user id `yara_garcia_1905` | `reservation_id: "8JX2WO"` — invented from nothing, in valid reservation-id format |

`retail__35` and `airline__34` are the sharpest: the model has learned each id slot's
*surface format* (10 digits; six alphanumerics) and emits a well-formed member of that
format rather than either using what was said or asking. `retail__64` is the same defect at
low amplitude — it heard "Chicago" and produced a real-looking Chicago zip that is wrong by
one digit.

### Mode D — violates the tool's argument schema (1/14)

`airline__40`. `update_reservation_passengers` takes `passengers: [{first_name, last_name,
dob}, …]`. The model tried, in order:

```
{"reservation_id": "3RGRK2T", "first_name": "May", "last_name": "Garcia", "dob": "1990-05-01"}
   -> Error: ... got an unexpected keyword argument 'first_name'
{"reservation_id": "3RGRK2T", "last_name": "Garcia", "dob": "1990-05-01"}
   -> Error: ... got an unexpected keyword argument 'last_name'
{"reservation_id": "3RGRK2T", "dob": "1990-05-01"}
   -> Error: ... got an unexpected keyword argument 'dob'
{"reservation_id": "3RGRK2T"}
   -> Error: ... missing 1 required positional argument
{"reservation_id": "3RGRK2T", "passengers": "May"}       # a string, not a list of objects
```

It flattens a nested schema, and when told the keyword is wrong it *deletes* the offending
key rather than restructuring. It never produces a list of objects. Only one episode in this
subset needed a nested argument, so treat the 1/14 as "untested", not "rare".

### Mode E — never calls a tool at all (5/14)

`airline__15`, `airline__9`, `airline__21`, `retail__21`, `retail__92`. Agent speech
64–1479 chars, zero tool calls, and the episode ends on `user_stop` or `max_steps`. Two distinguishable causes:

* **Unintelligible speech.** `airline__15`: the user simulator says *"Sorry, what was that?
  I didn't catch the end there"*, then *"I'm having trouble understanding you. Can you repeat
  that?"*, then *"Okay, I'm really confused. Let me start over."* The agent produced 698
  characters and none of it landed.
* **Loops without acting.** `retail__92`: *"You keep saying the same thing over and over."*
  The user also became confused about who they were talking to — *"Aren't you customer
  s[ervice]… who am I talking to then?"* — so the agent's speech misrepresented its own role.
  `airline__9`: the user spelled their id three times and the agent never used it.

This is the mode that mode 2.1 silently rewards.

### Mode F — invents tool names (2/14)

| episode | called | actually exists |
|---|---|---|
| `retail__7` | `get_user_id_by_name_zip`, `get_user_id_by_email`, `get_user_id_by_zip`, `get_user_id_by_name` | `find_user_id_by_name_zip`, `find_user_id_by_email` |
| `retail__64` | `find_user_id_by_zip`, `find_user_id_by_name` | `find_user_id_by_name_zip`, `find_user_id_by_email` |

`retail__7` is systematic: the model has the right *concept* for all four and the wrong verb
for all four (`get_` for `find_`). `retail__64` gets the verb right and invents plausible
argument-count variants that don't exist. Error text is explicit —
`Error: Tool 'get_user_id_by_zip' not found.` — and it retried `get_user_id_by_zip` seven
times regardless.

### Mode G (amplifier) — no error recovery, and the user is told nothing

Present in **all 9** error-capped episodes and the reason each cost only 46–99 s.

* Total agent speech in those episodes is **64–231 characters** — a greeting and nothing
  else. The model never tells the user that a lookup failed or asks them to repeat the id.
* After the first error it reissues the **byte-identical call**: `airline__3`, `airline__34`,
  `retail__35`, `retail__78` each fire the same call 10×; `airline__28` repeats one 8×;
  `retail__7` repeats one 7×. Where the calls do vary (`retail__49`, `retail__64`,
  `airline__40`) they vary along a mechanical axis — incrementing a counter, deleting one
  argument — never by going back to the user.

Without this mode every other mode would be recoverable in conversation. With it, one
mis-heard character ends the episode.

---

## 4. What is ruled out

Do not re-open these without new evidence.

* **The model does see the tool errors.** `push_tool_result`
  (`nemo/collections/speechlm2/models/streaming_fc_session.py:666`) injects
  `<TOOL_RESPONSE>[{"content": …, "is_error": true}]</TOOL_RESPONSE>` on the function
  channel. Mode G is a behavioural failure, not a plumbing one.
* **Not context truncation.** FC budget is 12000 tokens; no truncation warning in any log.
* **Not a parse failure.** Zero unparseable calls across all 14 episodes. Every call is
  well-formed JSON with a real-looking value — which is exactly what makes mode C dangerous.
* **Not the fixed-`MASTER_PORT` collision** and not the `name=""` parser bug; both were
  found and fixed earlier (see `logs/stage2_subset_0821_n1403/` and the τ-voice harness notes).
* **Not the episode cap.** Every error-capped episode used less than half its 200 s.

---

## 5. The control: are these episodes winnable?

### 5.0 Is the control's absolute level believable?

Checked against the τ-Voice paper (arXiv 2603.13686, Sierra.ai + Princeton) before using the
control for anything, because 1 real pass in 8 *looks* too low to be a valid reference.
It is not. Table 6 pass@1:

| provider | All Clean | All Realistic | retail Clean | airline Clean |
|---|---|---|---|---|
| **Google** `gemini-live-2.5-flash-native-audio` | **31%** | 26% | 45% | 28% |
| OpenAI `gpt-realtime-1.5` | 49% | 35% | — | — |
| xAI `grok-voice-agent` | 51% | 38% | — | — |

Google is the **lowest** of the three providers in the paper. Our control at n=8 scores
**3/8 reward > 0 = 37.5%** (avg reward 0.375; retail 3/5, airline 0/3), i.e. above the
published 31%, at a sample size whose 95% CI is roughly ±34 points. Nothing to explain.

The public leaderboard's much higher figures are **newer models, not this one**: Sierra's
blog puts xAI `grok-voice-think-fast-1.0` (Apr 2026) at 67% and states voice has gone from
"roughly 45% of text capability when the paper was written, to ~79% today". No Gemini voice
number is published there. Do not compare our Dec-2025-era Gemini Live against it.

Note also that the paper's pass@1 **is** the DB/COMMUNICATE reward whose vacuous-pass
artifact §2.1 documents, so the published numbers carry that artifact too.

Config audit vs the paper. Matched exactly: 200 ms tick, 1200 s conversation cap, ElevenLabs
TTS, fixed seed per task, and `--speech-complexity control` — verified equal to the paper's
"Clean" condition at `user_simulation_voice_presets.py:96`, whose `CONTROL_CONFIG` is
American personas, no background/burst noise, `frame_drop_rate 0.0`, no muffling, no vocal
tics, `telephony_enabled: True` (G.711 µ-law 8 kHz). (`regular` is the paper's *Realistic*
column and is what `prepare_submission.py:496` requires for leaderboard submission — a
different measurement, not a fix.) Remaining deviations, by likely impact:

1. Model `gemini-3.1-flash-live-preview` (the repo default) vs the paper's
   `gemini-live-2.5-flash-native-audio`, still present as `_LEGACY_GEMINI_MODEL`
   (`config.py:192`). 3.1 lacks proactive audio, input-audio transcription and context
   compression. Sign unknown.
2. User simulator `bedrock/…sonnet-4-5` vs the paper's `gpt-4.1`. No OpenAI key here.
   Symmetric across arms; a stronger simulator is plausibly a *harder* customer.
3. Stock ElevenLabs voices vs the 7 Sierra persona ids, which 404 on our key. This changes
   exactly the acoustics the ASR front-end sees, so it is the deviation most likely to
   matter for modes A and E specifically.
4. `--hallucination-retries 0` vs the CLI default 3 (`cli.py:444`). Matched across arms
   (`tau2_stage2_subset.sh:180`), so not a confound; mildly pessimistic for both.
5. n=8 on a subset chosen for arm A, not a random sample.

**Decision: not re-running the control on 2.5-native-audio.** Its job is to answer "were
these episodes winnable by some voice agent", and 37.5% vs arm A's 0 answers that with room
to spare. Re-running would move an absolute level that already agrees with publication and
would not touch any mode A/B receipt below, since those are per-episode tool-call
comparisons rather than aggregate rates.

### 5.1 Head to head

`gemini_baseline_0821b`, in flight. On the **4 episodes both arms have now
run**, which is the only comparison the numbers support so far:

| episode | arm A (NeMo 11B) | arm B (Gemini Live) |
|---|---|---|
| `airline__3` | `JFK59001` in `reservation_id`, same call 10× | `JMO1MG` + `anya_garcia_5901`, **2/2 actions**, 0 errors |
| `retail__49` | `order_1234`, `product_1234…1241` | reward **1.0**, 8/10 actions, **1/1 write exact** |
| `retail__7` | `John Doe`/`12345` + 4 invented tool names | `mei_kovacs_8020` exact, 2/6 actions, 0 errors |
| `retail__35` | `product_id 6086499569` fabricated, same call 10× | `arav.santos8321@example.com`, 1 error, then escalated |
| **totals** | 0 real passes, **0/4** expected actions, 40 tool errors | 1 real pass, **12/25** expected actions, 2 tool errors |

Arm A's `0/4` is over its 3 scored episodes only — the other 11 were never evaluated (§2.2),
which is itself the finding: they died before τ-voice could check anything.

The single most direct receipt, on the exact episode where arm A sent `SIN-555`:

```
airline__28   Gemini Live
  get_reservation_details {"reservation_id": "SI5UKW"}          -> ok
  get_user_details {"user_id": "amelia_rossi_1297"}             -> ok
  transfer_to_human_agents {...basic economy, no refund...}     -> Transfer successful
  reward 1.0, agent_stop, 193 s, 0 errors
```

Same audio pipeline, same user LLM, same voices. Gemini heard `SI5UKW` exactly and used the
right slot. **Modes A and B are model defects, not unwinnable episodes.** `airline__3` says
the same thing about the hardest sub-case, a spelled id containing the spoken word
"underscore": Gemini wrote `anya_garcia_5901`, arm A wrote `JFK59001`.

Two caveats so this is not over-claimed:

* `airline__28` records `action_checks: null`, so its 1.0 has nothing to verify against and
  an unchanged DB passes it. The *tool calls* are the evidence there, not the reward.
* Gemini is not clean on mode A either — on `retail__35` it heard
  `arav.santos8321@example.com` for `aarav.santos8321@example.com`, dropping one of the
  doubled `a`s in "Aarav". **The difference is mode G, not mode A.** Gemini got `User not
  found`, told the user, and escalated; arm A fires the identical failing call ten times in
  silence. One mis-heard character costs Gemini a detour and costs arm A the episode.

And an honest counter-signal: Gemini scored 0.0 on `retail__35` and `retail__7` as well.
Both were fully evaluated and failed on their merits, so those two episodes are hard for both
arms — the gap is not "every episode is winnable", it is "arm A cannot get far enough to
fail on the merits".

---

## 6. What SFT would need to fix, in priority order

Ranked by (episodes blocked) × (confidence the mode is real).

1. **Spelled alphanumeric grounding**, including the spoken tokens `underscore`, `dash`,
   `hash`, and digit-vs-letter homophones (`five`/`V`, `eight`/`A`). Must cover the id
   *formats* each domain uses, including retail's `#W` prefix. Blocks modes A and B.
2. **Ask, don't invent.** When a slot's value was not supplied, the target behaviour is a
   spoken question, never a well-formed guess. Blocks mode C, the most dangerous mode.
3. **An error-recovery speech act.** On `is_error: true`, say something to the user and
   change the call; never reissue verbatim. Blocks mode G, which gates everything else.
4. **Nested argument construction**, e.g. `passengers: [{...}]`. Blocks mode D. Undertested.
5. **Tool-name fidelity** — copy the name from the schema. Blocks mode F.
6. **Act at all, and intelligibly.** Blocks mode E, which the reward function currently
   rewards; needs §2.1 handled or improvement here will look like a regression.

---

## 7. Open questions

* **Is mode A an encoder error or an LLM error? Half answered.** Readout 0 is done
  (`scripts/mode_a_probe.py`, §3 mode A): an independent ASR recovers 4/4 ids from the same
  audio, so the failure is inside our model and the audio/codec explanation is retracted. What
  is still open is **encoder vs copy**, which needs the 11B and the remaining three readouts:

  | readout | condition | localises |
  |---|---|---|
  | 1 | real domain prompt, unchanged | does the failure reproduce? (**the gate** — greedy sampling, so it must) |
  | 2 | "repeat the id back, call no tools" | did the encoder+LLM **perceive** it? |
  | 3 | id supplied as text, no audio | can the LLM **copy** it into the slot at all? |

  If 2 succeeds where 1 fails, perception is intact and the value is lost on the way into the
  argument slot — which merges mode A into mode B and makes copy-fidelity supervision the fix,
  not more spelled-id audio. Caveat on readout 2: it changes the prompt, and this checkpoint is
  demonstrably prompt-sensitive, so it is evidence about perception rather than proof about
  that episode.

  The stimulus problem is solved and costs nothing: τ-voice persisted `both.wav` plus label
  tracks per episode, so all readouts use the exact waveform that produced `SIN-555`.
* **Does mode C survive when the id *is* supplied cleanly?** i.e. is it a grounding failure
  or a policy failure ("always fill the slot")?
* **Telecom is entirely unmeasured** — 0 of the subset's 8 telecom episodes have run on
  either arm. Mode proportions may not hold there.
* **Mode D is 1/14 because only one task needed a nested argument.** Its true rate is unknown.

---

## 8. How to reproduce

```bash
# arm A (needs a GPU slot in an existing allocation; never allocate a second node)
scripts/tau2_stage2_subset.sh <jobid>

# arm B (API only, no GPU)
cd ../tau-voice-2 && RUN_NAME=<new-name> TASK_IDS='15 28 3 34 40 9' \
  scripts/run_gemini_baseline.sh airline

# score either arm -- never quote raw reward
cd ../tau-voice-2 && python scripts/score_voice_run.py \
  data/simulations/stage2_subset_0821 data/simulations/<arm-b-run> \
  --label nemo_11b --label gemini_live --per-episode
```

Never reuse a run name: τ-voice writes per-episode directories and a collision silently
overwrites the arm you were going to compare against.

---

## Changelog

* **2026-08-21** — created. 14/24 arm-A episodes finished, 0 real completions. Six modes
  catalogued (A–F) plus amplifier G. Three benchmark scoring artifacts documented (§2), all
  three found while building the scorer rather than by reading the reward code. Gemini
  control in flight, 4 episodes matched: 1 real pass and 12/25 expected actions against arm
  A's 0 and 0/4. Confirms modes A and B are model defects, and that Gemini shares mode A at
  much lower amplitude but not mode G — which is what makes arm A's version fatal. Mode D
  and the encoder-vs-LLM question (§7) are open.
* **2026-08-21, later** — §5.0 added. The control's absolute level was audited against the
  τ-Voice paper because it looked too low to be a valid reference: it is not. Google's
  published Clean pass@1 is 31% All / 45% retail / 28% airline, the lowest of the three
  providers, and our control is at 37.5% (3/8) — above it. The public leaderboard's higher
  numbers belong to newer thinking-voice models (xAI at 67%), not to Gemini Live. Config
  audit recorded; `control` verified equal to the paper's "Clean". Decision: do not re-run
  the control on `gemini-live-2.5-flash-native-audio`.
