# AI Usage Report

How I used AI to build this project, what I wrote and decided myself, how I checked what the AI
produced, and where it was wrong. The incidents below were logged while they happened
(26–29 October 2026), not reconstructed at the end.

## 1. Tools

| Tool | Used for |
|---|---|
| **Claude Code** (Anthropic's coding agent, in the terminal next to PyCharm) | Planning the stages, writing most of the code, tests and documentation, running checks and measurements, explaining every step |
| Groq models inside the product | `gpt-oss-120b` answers questions (`gpt-oss-20b` as fallback); `qwen3.8-27b` grades answers in the evaluation. These are part of the system, not development help |

## 2. How I worked with it

- **A fixed plan of 13 stages**, agreed at the start: skeleton, dataset, parsers, chunking and
  deduplication, embeddings and Qdrant, keyword and hybrid search, simple RAG, the agent
  (multi-step, tools, contradictions, abstention), API, UI, evaluation, tests and deployment,
  documentation.
- **Small steps**: each step was one to three files. Claude explained each file (in Georgian), I
  read it, ran it myself (PyCharm console, Swagger UI, the Streamlit UI) and only then asked for the
  next step.
- **I made every commit myself.** Claude never ran `git commit` or `git push`; it only suggested
  the commands. The git history is mine and shows the project growing step by step.
- **The interview is without AI**, so I asked for explanations I could repeat: why RRF needs no
  score calibration, why Recall@K raises an error without gold documents but MRR returns 0, how the
  ONNX embedder pools the [CLS] token, and so on.

## 3. What I wrote or decided myself

- **Reciprocal Rank Fusion** (`app/retrieval/fusion.py`): written by me. Claude wrote the tests
  and checked them first against a correct version and four typical bugs (ranks from 0, changing
  the input, summing raw scores, ignoring the limit), so that a passing test meant something.
- **Recall@K and MRR** were planned as mine too. I asked Claude to write them in the end; the tests
  for them had been checked against a correct version and nine typical bugs before. I reviewed the
  code and can explain both functions line by line.
- **Deployment platform**: the plan was Hugging Face Spaces (Docker). When I created the account, I
  found that Docker Spaces are no longer free for new accounts and asked for an alternative. That
  led to Render + Streamlit Community Cloud, and to replacing torch with ONNX Runtime so the API
  fits in 512 MB.
- **Product decisions**: the name "AI Engine", running the evaluation on gpt-oss-20b when the 120b
  daily quota was used up, which fixes to deploy before the deadline.
- **Testing by hand**: I ran every step's checks and the full requirement checklist, used the UI
  like a user would, and found several of the problems in the next section myself.

## 4. Problems I found by testing

| When | What I found | What it led to |
|---|---|---|
| Stage 4 | The deduplication results were not saved anywhere | `scripts/build_chunks.py`, which writes the chunk files |
| Stage 7–8 | "Was Cars 2 released by the same company…" was answered "yes" correctly, but cited the video game's paragraph instead of the film's | The difference between strict citation accuracy (gold documents) and "a source that also states it" in the Evaluation Report |
| Stage 8.3 | I changed the agent code on purpose to see the tests fail: 2 tests failed, Claude had predicted 1 | Both code paths reach the same final line; Claude also split assertions joined with `and` |
| Stage 8.4 | An abstention ("not enough information") came back with citations [1][6][7] | Abstentions now keep no citations and their markers are removed |
| Stage 8.4 | The root-cause contradiction of the task example was not detected; I checked that the status page is found when the question has no incident ID | Confirmed a retrieval problem: the incident ID was only in the title → titles are indexed in BM25 |
| Stage 8.4 | Full requirement checklist: formats, chunking, dedup, add/update/delete, hybrid search for "GW-503", the task example, LLM errors, secrets | Everything as required; the *Black Book* question picked the wrong actress (later an error-analysis example) |
| Stage 10 | The UI said "add or update" but only showed adding | The UI now recognizes an indexed file name and offers "Update document" |
| Stage 10 | The multi-hop example (*Black Book*) gave different answers, sometimes "not enough information" | A prompt rule for questions that describe someone by several attributes; the remaining randomness is documented |
| Stage 11 | The judge graded "About 12% of card transactions failed" PARTIAL for a contradiction question, after Claude had reported its judge prompt as fixed | A separate judge prompt for contradictions, calibrated on 12 cases × 2 runs |
| Stage 12 | On the live demo: after I deleted my uploaded document, the same question was answered from an unrelated paragraph ("false premise") instead of "not enough information" | A prompt rule that a source must be about what the question asks about (see the Evaluation Report, 5.3) |

## 5. Where the AI was wrong, and how it was caught

| Stage | AI mistake | Caught by | Fix |
|---|---|---|---|
| 3 | PDF text extracted one word per line (default pypdf mode) | Running the parser on a sample | `extraction_mode="layout"` |
| 3 | Parsers crashed on a non-object JSONL line and on a fake `.docx`; sentences duplicated in metadata | Code review of its own code and tests on broken files | Per-line errors, clear `IngestionError`s |
| 5 | Embedding time estimated at 15–30 minutes | A benchmark: 17 chunks/s on my CPU, about 2.5 hours for both strategies | Embeddings cached in `.npz`, keyed by model name and chunk-file hash |
| 5 | A check assumed "Ed Wood (film)" ranks first | The scores: a tie with the "Ed Wood" biography (0.7914 vs 0.7907) | The check was too strict, the code was right |
| 7 | Planned LLM (`llama-3.3-70b-versatile`) had already been retired by Groq | 404 on the first call | Listed and probed the available models, chose `gpt-oss-120b` |
| 7 | Concluded from one run that *Black Book* "abstains with 5 sources and guesses with 10" | My own run: it abstained both times | Lesson: one run of an LLM proves nothing; the cause was randomness |
| 8 | Groq's gpt-oss ignores a forced `tool_choice` and calls tools that were not offered | 400 `tool_use_failed` in a live run | The last step offers only `submit_answer` and says so; `ToolCallError` is handled |
| 8 | Agent answers had no inline `[n]` markers; an unanswerable question cost 3 rephrased searches (7k tokens) | Live runs | Required `sources` field, stricter answer description, `max_searches=2` (the cap, not the prompt, controls cost) |
| 8 | Broken tool-call JSON produced "not enough information" with `answered = true` | Writing the agent's unit tests | An empty or no-answer text is always an abstention |
| 8 | Conflict descriptions kept look-alike characters (narrow no-break space, non-breaking hyphen) | Live runs | Normalized like answers |
| 8 | A code comment said fixed dates keep the sample files identical | Regenerating them: the `.docx` changed (zip timestamps) | Comment corrected |
| 11 | Retrieval evaluation made `fixed` chunking look far worse (Recall@10 0.60 vs 0.97) | Reading the results: the `fixed` collection had no INC-2041 documents | Indexed them, measured again: the strategies are equal |
| 11 | Judge prompt "fixed" and reported as 9/9 after testing one contradiction example | My test with another example | Separate contradiction prompt; lesson: several examples per rule, repeated runs |
| 11 | The judge saw only the answer text, not the conflicts the UI shows next to it | Reading the failed answers | The judge grades the answer with its conflicts; everything re-graded |
| 11 | A reference answer included a detail the question did not ask for | A correct answer graded PARTIAL | Reference fixed, re-graded |
| 11 | The daily token limit (200K per model) was not checked when the evaluation was planned; Claude's own experiments that day had used much of it | The evaluation stopped with "tokens per day" | Resumable evaluation, per-minute vs daily limit errors, the main comparison run on gpt-oss-20b, a fallback model in the app |
| 12 | The deployment plan assumed free Docker Spaces | My account (see section 3) | Render + Streamlit Cloud; ONNX Runtime instead of torch (643 → 287 MB), vectors checked identical |

Claude also caught problems that were not its own: a `git add` I typed over two lines left one
file out of a commit (seen in `git status`), and HotpotQA questions whose gold answer does not
answer the question were found while building the evaluation set.

## 6. How AI output was checked

- **Tests**: 142 unit tests with fakes for the LLM, Qdrant and the embedder. For the parts where a
  wrong test would hide a wrong implementation (RRF, Recall@K, MRR), the tests were first run
  against deliberately broken versions to see that they fail.
- **Equivalence checks**: when torch was replaced by ONNX Runtime, the new vectors were compared
  with the stored ones (largest difference 2·10⁻⁷) and the retrieval evaluation was repeated: all 36
  result rows and 234 rankings were unchanged.
- **Calibration**: the LLM judge was checked on cases with known grades before it graded anything.
- **Reproducibility**: I re-ran the evaluation scripts myself; the evaluation set and the retrieval
  and error-analysis results came out identical to Claude's runs.
- **Live checks**: every endpoint after deployment (health, the task example, protected files,
  upload → question → delete), and the UI in a private window.
- **Reading the output, not only the scores**: several of the problems above were found by
  reading individual answers that a metric graded wrong.

## 7. Secrets and safety rules

- The API keys exist only in my `.env` (git-ignored) and in the hosts' environment settings.
  Claude did not read `.env` (I declined the one command that would have looked at it); checks
  against another database used environment overrides instead of editing the file. Command output
  was filtered so that a key could not be printed.
- A key I pasted into the chat at the very start was revoked and replaced.
- The Docker image contains only `requirements.txt` and `app/` (an allow-list `.dockerignore`).
- The public demo protects the sample documents from deletion by visitors.

## 8. What I learned

- **An LLM's single run is an anecdote.** Several conclusions changed after a second run or a
  second example (the *Black Book* answers, the judge prompt).
- **Test the tests.** A test suite that also passes a broken implementation proves nothing;
  checking it against typical bugs is cheap.
- **Measure before trusting an estimate**: the embedding time, the token cost per question (2.3k,
  not 4k), the memory of torch (643 MB) were all different from the first guess.
- **Read the failures.** The measurement errors in the evaluation (missing documents, the judge not
  seeing conflicts, a wrong reference) were found by reading wrong answers, not by the averages.
- **Free tiers shape the design**: the daily token limit led to the resumable evaluation and the
  fallback model, the 512 MB server to the ONNX embedder.
- **AI speeds up writing code, not understanding it.** The parts I can explain best are the ones
  I wrote or broke on purpose myself.
