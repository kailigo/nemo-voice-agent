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

Homophone errors in ordinary words share the mode: `airline__40`, "change it from Mei Lee to
Mei Garcia" → the model sent `first_name: "May"`.

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

`gemini_baseline_0821b`, in flight. Head to head on the **4 episodes both arms have now
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

* **Is mode A an encoder error or an LLM error?** Unresolved and it decides what the
  training data looks like. The model is end-to-end duplex (speech encoder
  `nvidia/nemotron-speech-streaming-en-0.6b` feeding the LLM) and `user_transcript` is
  `None` in every tick, so this cannot be separated from outside τ². Proposed probe: synthesize
  the exact spelled ids from the failing episodes and drive the model directly, comparing what
  the encoder produces against what the function channel emits. ~1 GPU, minutes. **Needs a
  go-ahead — it changes the SFT recipe.**
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
