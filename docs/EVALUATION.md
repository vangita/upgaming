# Evaluation Report

How well the AI Engine finds evidence and answers questions, measured on 47 questions in five
categories, plus 500 HotpotQA questions for retrieval. All numbers come from the result files in
[`data/eval/results/`](../data/eval/results/) and can be reproduced with the commands at the end.

## Summary

- **Hybrid search finds the most evidence.** On the evaluation set it reaches Recall@5 0.949 and
  MRR 0.938, against 0.872 / 0.857 for semantic and 0.816 / 0.909 for keyword search. On 500
  HotpotQA questions it has the best Recall@10 (0.877 vs 0.831 and 0.805).
- **The agent answers correctly 95% of the time** (score 0.947 with gpt-oss-120b and with
  gpt-oss-20b), the simple pipeline 86% (0.862).
- **The whole difference comes from contradictions**: the agent reported the disagreement in 5 of 6
  contradictory questions; the simple pipeline read both versions every time and gave one, 0 of 6.
- **No configuration answered an unanswerable question** (8 of 8 abstentions each), and the agent
  abstained wrongly on only 1 of 39 answerable questions.
- **gpt-oss-120b and gpt-oss-20b are equal here**: the per-category differences are one question
  each, which is within the noise of a 47-question set.
- Of the agent's 3 wrong answers per model, 1 is a retrieval error (a needed chunk ranked 7th when
  the agent reads 6), 1 is over-caution, and 1 is a question with a broken label.
- Two fixes came after this evaluation (false premises, repeated searches, see 5.3). They were
  checked on targeted questions; a full re-run with them stopped at the daily token limit after 14
  of the 47 questions, all 14 correct. The numbers below describe the version before the fixes.

## 1. Evaluation set

`data/eval/questions.jsonl`, built by `scripts/build_eval_set.py` (the same file on every run):

| Category | Questions | Source | What it tests |
|---|---|---|---|
| simple | 10 | 5 about the INC-2041 documents, 5 about single HotpotQA paragraphs (written by hand) | One document holds the answer |
| multi_hop | 12 | 11 HotpotQA "bridge" questions, 1 INC-2041 question | A fact from one document is needed to use another |
| multi_doc | 11 | 8 HotpotQA "comparison" questions, 3 INC-2041 questions | Facts from several documents are combined |
| unanswerable | 8 | 4 INC-2041, 4 HotpotQA-style (written by hand) | No document holds the answer: the system must abstain |
| contradictory | 6 | INC-2041 | The documents disagree: both versions must be reported |

- **HotpotQA questions** are a random sample of the validation split (seed 2026). Of the 22
  sampled, 3 were dropped because their gold answer does not answer the question (e.g. "In what
  country…?" → "Iranian"); the reason for each is in `DROPPED`.
- **Unanswerable questions** were checked against the corpus by text search. Half are "near
  misses": the entity exists in the corpus but the fact does not (Carice van Houten is listed in
  the cast of *Black Book*, her birth date appears nowhere).
- **Contradictory questions** cover the four planted disagreements: cause (expired TLS
  certificate vs database failover), duration (47 vs about 65 minutes), end time (14:52 vs 15:10)
  and failed share (12% vs 18%). The task's own example question is one of them.
- **Gold documents** (titles) are given for every answerable question: the HotpotQA supporting
  paragraphs, or the INC-2041 files that hold the fact (all files with a version, for
  contradictions).

## 2. Metrics

| Metric | Definition |
|---|---|
| **Recall@K** | Share of the gold documents among the first K *documents*. The top 10 chunks are turned into a document ranking first (several chunks of one document count once, at the rank of the best one; a deduplicated paragraph also counts under its other titles). |
| **MRR** | Mean over questions of 1 / rank of the first gold document (0 if none is in the top 10). |
| **Answer correctness** | 1 / 0.5 / 0 for CORRECT / PARTIAL / INCORRECT, graded by an LLM judge against the reference answer (see 2.1). |
| **Citation accuracy** | Share of the answer's citations that point to a gold document. Strict: citing another document that states the same fact counts as wrong. |
| **Citation coverage** | Share of the gold documents cited at least once; a multi-hop answer must cite both links of the chain. |
| **Abstention** | Unanswerable: abstained correctly. Answerable: abstained wrongly (false abstention). |
| **Conflict detection** | Contradictory questions where the answer reports a conflict; "spurious" = a conflict reported anywhere else. |
| **Cost** | LLM calls, prompt + completion tokens and seconds per question. |

Unanswerable questions have no gold documents, so they are left out of Recall@K, MRR and the
citation metrics; they are measured by abstention.

### 2.1 The LLM judge

String matching cannot grade these answers: a comparison question ("Valley International or
Aberdeen?") names both candidates in every answer, right or wrong, and "Thirty-three" vs "33" or
"Eric Whitacre" vs "Eric Edward Whitacre" is still correct. The judge (`app/evaluation/judge.py`):

- is **qwen/qwen3.8-27b**, a different model family from the answering gpt-oss models (no
  preference for its own wording), with its own Groq rate limits (~340 tokens per grade);
- sees the question, the reference answer and the system answer **with the conflicts it
  reported**, because the UI shows them next to the answer;
- uses a **separate prompt for contradictory questions**: CORRECT = both versions, PARTIAL = says
  the sources disagree but gives one version, INCORRECT = one version without saying so;
- is called only when needed: an abstention on an unanswerable question is correct, an abstention
  on an answerable one is incorrect, without asking the judge.

**Calibration.** The judge was checked on 12 cases with known grades (wrong entity, name and
number formats, partial answers, every kind of contradiction answer), each run twice: **24 of 24**
as expected. An earlier version had one prompt for everything; it graded a one-version answer to a
contradiction PARTIAL ("one of two parts right"). A first fix looked right on the single example
it was tested on, and the developer's own test with another example showed the same error again;
the separate contradiction prompt fixed it.

**Known limitation**: the judge compares answers, not reasoning. "Carice van Houten, who starred in
Black Book, is Dutch" is graded correct for "Black Book starred the actress and writer of what
heritage?" (gold: Dutch, via Halina Reijn), although it names the wrong actress. Citation coverage
catches such cases: the gold paragraph is not cited.

## 3. Retrieval

`scripts/eval_retrieval.py`: no LLM, 6 configurations (3 search modes × 2 chunking strategies), the
39 answerable questions of the evaluation set and 500 random HotpotQA questions (seed 42), top 10
chunks. Results: `retrieval.json`.

### 3.1 Evaluation set (39 questions)

| Chunking / mode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | ms |
|---|---|---|---|---|---|---|
| structured / semantic | 0.500 | 0.808 | 0.872 | 0.897 | 0.857 | 66 |
| structured / keyword | 0.538 | 0.774 | 0.816 | 0.906 | 0.909 | 16 |
| **structured / hybrid** | **0.560** | **0.821** | **0.949** | **0.974** | **0.938** | 56 |
| fixed / semantic | 0.487 | 0.825 | 0.872 | 0.936 | 0.866 | 52 |
| fixed / keyword | 0.538 | 0.782 | 0.838 | 0.915 | 0.909 | 15 |
| fixed / hybrid | 0.560 | 0.846 | 0.974 | 0.987 | 0.942 | 53 |

(Recall@1 cannot exceed 0.5 for a question with two gold documents.)

### 3.2 HotpotQA, 500 random questions

| Chunking / mode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| structured / semantic | 0.428 | 0.714 | 0.789 | 0.831 | **0.903** |
| structured / keyword | 0.387 | 0.637 | 0.712 | 0.805 | 0.849 |
| **structured / hybrid** | 0.417 | 0.730 | 0.799 | **0.877** | 0.894 |
| fixed / semantic | 0.422 | 0.710 | 0.788 | 0.837 | 0.893 |
| fixed / keyword | 0.386 | 0.634 | 0.709 | 0.805 | 0.847 |
| fixed / hybrid | 0.413 | 0.728 | 0.809 | 0.879 | 0.887 |

### 3.3 By category (structured chunking, Recall@5 / MRR)

| Category | semantic | keyword | hybrid |
|---|---|---|---|
| simple (10) | 1.000 / 0.933 | 1.000 / 1.000 | 1.000 / 1.000 |
| multi_hop (12) | 0.750 / 0.883 | 0.667 / 0.844 | 0.833 / 0.896 |
| multi_doc (11) | 0.909 / 0.864 | 0.773 / 0.909 | 1.000 / 0.955 |
| contradictory (6) | 0.833 / 0.667 | 0.889 / 0.889 | 1.000 / 0.889 |

### 3.4 Findings

1. **Hybrid has the best recall everywhere.** The agent reads the top 6 chunks, so Recall@5–10
   matters more than the first position: hybrid is the right default.
2. **Semantic search has a slightly better MRR on HotpotQA** (0.903 vs 0.894): fusion sometimes
   lifts a keyword match above the semantic top hit. With 500 questions this 0.009 difference is
   about one standard error; the Recall@10 gain of hybrid (+0.046) is not.
3. **Keyword search carries the contradictions**: the status page never mentions "INC-2041" in
   its text and says "database failover"; semantic search ranks it low (MRR 0.667 on
   contradictory questions), BM25 finds it (0.889). Indexing the title with the BM25 text was
   needed for this: "INC-2041" is only in the title of that file (measured earlier: keyword
   Recall@10 "both gold" on 200 HotpotQA questions rose from 53.5% to 60.0% with titles indexed).
4. **The two chunking strategies are equal on this data** (differences of one question). HotpotQA
   paragraphs are short and fit in one chunk with either strategy, and the INC-2041 documents are
   small. `structured` is kept for its other benefit: a section with its heading is its own chunk,
   so citations point to the right part of a document.
5. **Speed**: keyword ~16 ms, semantic ~50 ms, hybrid ~53 ms on a laptop. Hybrid costs barely
   more than semantic: the query embedding (~23 ms on CPU) dominates, the two Qdrant searches are
   cheap.

**Caveat found in the error analysis**: Recall@K counts documents from the top 10 chunks. When one
document has several chunks in the top 10, a document ranked 5th can be the 7th chunk, outside
the 6 chunks the agent reads (see 5.1). Document-level recall therefore overstates slightly what
the LLM sees.

## 4. Answers

`scripts/eval_answers.py`: each method as the app runs it (hybrid search), all 47 questions, one
run per configuration. Results: `answers_*.jsonl` (every answer with its citations, conflicts,
searches, tokens and the judge's verdict) and `answers_summary.json`.

**Version note.** The three runs in this section used the code before two later changes (5.3): a
prompt rule against false premises and skipping repeated searches. The deployed configuration was
evaluated again with the new code (`answers_agent_gpt-oss-120b_v2.jsonl`), but that run stopped at
the free tier's daily token limit after 14 of the 47 questions (10 simple, 1 multi-hop, 3
multi-doc). All 14 were correct, as in the first run; the other 33 were not run before the deadline.

| Configuration | Method | Model |
|---|---|---|
| agent, gpt-oss-120b | Agent: top 6 chunks, up to 2 follow-up searches of 3 chunks, structured answer with `conflicts` | the deployed model |
| agent, gpt-oss-20b | the same | a 6× smaller model |
| simple, gpt-oss-20b | One search (top 10 chunks), one LLM call, plain-text answer | the same small model |

The simple pipeline was not run with gpt-oss-120b: the free tier gives each model 200K tokens per
day, and the deployed demo needs that model's quota.

### 4.1 Overall

| | agent, 120b | agent, 20b | simple, 20b |
|---|---|---|---|
| **Answer correctness** | **0.947** | **0.947** | 0.862 |
| Verdicts (correct / partial / incorrect) | 44 / 1 / 2 | 44 / 1 / 2 | 40 / 1 / 6 |
| Citation accuracy | 0.974 | 0.980 | 0.987 |
| Citation coverage | 0.868 | 0.860 | 0.786 |
| Abstained on unanswerable | 8 / 8 | 8 / 8 | 8 / 8 |
| False abstentions (of 39) | 1 | 1 | 0 |
| Conflicts found (of 6) | 5 | 5 | 0 |
| Spurious conflicts | 0 | 0 | 0 |
| LLM calls / question | 1.36 | 1.43 | 1.00 |
| Tokens / question | 2,284 | 2,360 | 1,639 |
| Seconds / question | 1.6 | 1.2 | 0.9 |
| Tokens for the run | 107,339 | 110,941 | 77,013 |

### 4.2 Answer correctness by category

| Category | agent, 120b | agent, 20b | simple, 20b |
|---|---|---|---|
| simple (10) | 1.00 | 1.00 | 1.00 |
| multi_hop (12) | 0.88 | 0.96 | 0.96 |
| multi_doc (11) | 1.00 | 0.91 | 1.00 |
| unanswerable (8) | 1.00 | 1.00 | 1.00 |
| contradictory (6) | 0.83 | 0.83 | **0.00** |

### 4.3 Findings

1. **Agent vs simple pipeline**: +0.085 correctness, all of it from contradictory questions. The
   agent's `conflicts` field makes the model look for disagreements; the simple prompt has no such
   step, and the model picks one version even though both are in its 10 sources (verified in 5.1).
2. **The agent rarely needed its follow-up searches on answerable questions**: of 47 questions,
   37 were answered without one (120b). In the HotpotQA distractor corpus both paragraphs of a
   bridge question are usually found by the first hybrid search (multi-hop Recall@5 0.833), so the
   second hop is already in the sources. The searches were used mostly on unanswerable questions,
   before abstaining.
3. **Abstaining costs the agent ~3× more** than the simple pipeline (4,373 vs 1,521 tokens on
   unanswerable questions): it searches twice before giving up, with the same result.
4. **gpt-oss-120b vs gpt-oss-20b**: equal overall; each wins one category by one question. On this
   set the smaller model is as accurate and faster (1.2 s vs 1.6 s per question), so the 20b model
   is a sound fallback when the 120b daily limit is used up, which is how the app uses it.
5. **Citations are precise but incomplete**: ~97–99% of citations point to a gold document, but
   multi-hop answers often cite only one link of the chain (coverage 0.63–0.71 on multi-hop).

## 5. Error analysis

`scripts/analyze_errors.py` rebuilds what the LLM read for each answer that is not correct (hybrid
search is deterministic on the same index: the first search plus the agent's saved follow-up
searches; the rebuilt number of sources matched the logged one for every answer) and sorts it:

- **retrieval**: a gold document never reached the LLM;
- **generation**: the LLM read every gold document and still answered wrong or abstained.

Results: `error_analysis.json`.

### 5.1 Wrong answers

| Question | Category | agent, 120b | agent, 20b | simple, 20b |
|---|---|---|---|---|
| inc-c5 "At what time was INC-2041 resolved?" | contradictory | **retrieval** | **retrieval** | generation |
| inc-c1, c2, c3, c4, c6 | contradictory | ✓ | ✓ | generation (5) |
| hp-multihop4 (garbled question) | multi_hop | label noise | label noise | label noise |
| hp-multihop6 "…first aired on what station?" | multi_hop | generation | ✓ | ✓ |
| hp-multidoc2 "Which film was released first…" | multi_doc | ✓ | generation | ✓ |
| **Total not correct** | | 3 (1 retrieval, 2 generation) | 3 (1 retrieval, 2 generation) | 7 (all generation) |

The patterns behind them:

1. **Chunk crowding (retrieval, inc-c5).** The hybrid top 10 for "At what time was INC-2041
   resolved?" is: support summary, Q1 report, post-mortem, post-mortem (timeline), Q1 report, an
   unrelated HotpotQA paragraph, **status page**, … The status page, the only source of "15:10",
   is the 7th chunk; the agent reads 6. Both models answered "14:52" with no conflict. The
   simple pipeline reads 10 chunks and saw it, but gave one version anyway. *Fix (not done)*: at
   most 2 chunks per document in the sources, or `top_k` 8.
2. **One version, silently (generation, simple pipeline, 6 of 6).** E.g. "The INC-2041 card
   payments outage lasted 47 minutes. [1][6]", with the status page's 65 minutes among its sources.
   Detecting contradictions needs an explicit step in the answer format, not only the evidence.
3. **Over-caution (generation, agent).** gpt-oss-120b abstained on hp-multihop6 without trying a
   follow-up search, although both gold paragraphs were in its sources; gpt-oss-20b searched twice
   for a Japanese release date on hp-multidoc2 and then abstained, although its sources held the
   answer. One false abstention in 39 answerable questions per model.
4. **Label noise (hp-multihop4).** "What are a Uruguayan professional footballer and Luz
   Pfannenstiel are best known as?" (gold: "the best known globetrotters"): the question is garbled
   in HotpotQA itself; every configuration answered half of it.

### 5.2 Correct answers with incomplete citations

9 (agent, 120b), 10 (agent, 20b) and 11 (simple) correct answers miss a gold document in their
citations or cite another one. Almost all are multi-hop or comparison answers that cite one link:
for "Jullian Allen … illustration appeared in Queen magazine… What is the name of the magazine
now?" the answer "Harper's Bazaar" cites the Queen magazine paragraph but not Julian Allen's. The
answer is supported, the chain is not fully shown.

### 5.3 Checks outside the question set

- **False premise (found by manual testing on the live demo, after this evaluation).** A user
  uploaded a document, asked about "the two candidate models evaluated on the same customers",
  got a correct cited answer, deleted the document and asked again. The agent then applied an
  unrelated HotpotQA paragraph (a generic method for ranking systems) to "the two candidate
  models" instead of abstaining. None of the 8 unanswerable questions had a false premise with
  matching words, so the evaluation could not catch it. A rule was added to both prompts: a source
  answers only if it is about what the question asks about (the same people, events, documents
  or numbers). With it, gpt-oss-120b abstained on all 5 false-premise questions tried (the user's
  and four similar ones, e.g. about an incident "INC-2052" that does not exist). Answerable
  questions checked afterwards: the 14 of the partial re-run (section 4) and two more were correct;
  *Black Book* was right 1 time in 3 (as unstable as before), and one comparison ("Which film was
  released first…") was answered wrong once, comparing US instead of Japanese release dates.
  Whether the rule makes the agent more cautious overall would need the full re-run.
- **Repeated searches (found on the live demo).** The agent once ran the identical search
  ("Carice van Houten heritage") twice and so spent both follow-up searches on the wrong actress. A
  repeated query is now not run: the model is told to try another candidate, and the repeat does
  not use up a search (at most one repeat, so the loop still ends). It does not help when the model
  rephrases the same search, which is what happened in the later *Black Book* failures.
- **Prompt injection (manual, one run).** A document with instructions addressed to the model was
  uploaded and a related question asked: the answer did not follow the instructions (the prompts
  say the sources are data, not instructions). The injected document did take one of the 6 source
  slots and pushed out a relevant one, which is the chunk-crowding problem again. An automated
  version, `scripts/eval_injection.py` (five planted attacks: ignore the instructions, a fake system
  message, reveal the prompt, call the search tool, state a false fact), is in the repository but
  was not run before the deadline because of the daily token limit.
- **Non-determinism.** At temperature 0 the agent still gave different answers to the same
  multi-hop question across runs (*Black Book*: right actress, wrong actress, abstention). Each
  configuration was run once, so a single question can flip between runs.
- **Deployed index.** The 47 questions searched on Qdrant Cloud and on the local index give the
  same hybrid top 10 for 29 of 47 questions and 9.7 of 10 shared chunks on average; hybrid Recall@5
  0.936 (cloud) vs 0.949 (local), about one gold document. The HNSW graph was rebuilt, and BM25's
  IDF statistics differ slightly (the local collection still counts files deleted during testing).

## 6. Corrections made during the evaluation

The measurement itself had errors; each was fixed and everything affected was measured again:

| Problem | Effect | Fix |
|---|---|---|
| The `fixed` collection did not contain the INC-2041 documents | `fixed` looked far worse on the evaluation set (Recall@10 0.60) | Indexed them (`ingest_files --chunking fixed`), re-ran the retrieval evaluation |
| One judge prompt for every category | One-version answers to contradictions graded PARTIAL | Separate prompt for contradictions, calibrated on 12 cases × 2 runs |
| The judge saw only the answer text | inc-c3: the agent's conflicts field had both versions (shown in the UI), the text only one → graded INCORRECT | The judge grades the answer with its reported conflicts; all answers re-graded |
| A reference answer with an extra detail (inc-s5: "99.94%, below the 99.95% target") | A correct "99.94%" graded PARTIAL | Reference fixed to "99.94%"; all answers re-graded |

## 7. Limits of this evaluation

- **Size**: 47 questions; one question moves answer correctness by about 0.02, a category score
  by 0.08–0.17. Differences of one question are not evidence.
- **One run per configuration**: the answering model is not deterministic (5.3).
- **The judge can be wrong**: it was calibrated on 12 cases, not validated on every answer; it
  does not check reasoning (2.1).
- **Who wrote what**: the INC-2041 documents and their questions were written by the same team,
  so they may be easier than real incident reports; the HotpotQA part is independent.
- **Not compared**: re-ranking, other embedding models, and the simple pipeline with
  gpt-oss-120b (free-tier token budget).

## 8. Reproducing

```bash
python -m scripts.build_eval_set                            # data/eval/questions.jsonl
python -m scripts.eval_retrieval                            # retrieval.json (~2.5 min, no LLM)
python -m scripts.eval_answers --methods agent              # with GROQ_MODEL (gpt-oss-120b)
python -m scripts.eval_answers --model openai/gpt-oss-20b   # agent and simple
python -m scripts.eval_answers --methods agent --tag v2     # a new code version, next to the old results
python -m scripts.analyze_errors                            # error_analysis.json
python -m scripts.eval_injection                            # injection.json (planted instructions)
```

`eval_answers` saves each answer as soon as it has it and continues where it stopped, so the
daily token limit only pauses it; `--regrade` grades the saved answers again without new LLM
answers, `--only ID…` runs a subset.
