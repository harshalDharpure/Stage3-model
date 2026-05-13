# Legal POSCO 3-Stage Project — Dataset & Research Details

This document answers the checklist questions for the **Legal_posco-3stages** repository (`q1_3stage_pipeline`). Counts and aggregates below were **recomputed from the dialogue JSONL files** in this repo (`train_70` / `val_10` / `test_20`) unless stated otherwise.

---

## 1. Source of legal documents

- **Primary packaged source (this repo):** dialogue-level JSONL splits used for training and evaluation:
  - `q1_3stage_pipeline/data/train_70_dialogues.jsonl`
  - `q1_3stage_pipeline/data/val_10_dialogues.jsonl`
  - `q1_3stage_pipeline/data/test_20_dialogues.jsonl`
- **Upstream merge (documented in code):** `create_70_10_20_split_dialogue_level.py` loads three **raw** dialogue corpora (expected at repo root when you run the script):
  - `hindi_complete_posco_data.jsonl`
  - `english_posco_dataset.jsonl`
  - `code_mixed_posco_dataset.jsonl`
- **Optional sync path (README):** `q1_3stage_pipeline/README.md` references copying earlier experiment data via  
  `experiments/exp3_pretraining_finetuning/finetuning` → `q1_3stage_pipeline/data/raw` using `sync_dataset.py` (that script path is documented; the parent `experiments/` tree may live outside this standalone repo clone).

**Domain:** Indian child-protection / criminal-law style Q&A (POCSO, IPC, CrPC, JJ Act, etc.), in **Hindi**, **English**, and **Hinglish / code-mixed** text.

---

## 2. Number of raw legal documents (dialogues)

- **Merged dialogue corpus used for the 70/10/20 split:** **1200 dialogues** total  
  (840 train + 120 validation + 240 test — see `create_70_10_20_split_dialogue_level.py` comments: exact global totals 840/120/240).

*Note:* “Raw legal documents” in the sense of **court judgments / FIR PDFs** are **not** stored in this repository; the unit of data here is **synthetic or curated legal dialogues** in JSONL form (see §3).

---

## 3. How were dialogues generated (model)?

- **This repository does not include** the original prompt templates, model name, or generation logs used to build `hindi_complete_posco_data.jsonl`, `english_posco_dataset.jsonl`, and `code_mixed_posco_dataset.jsonl`.
- **What is in-repo:** the **merge + stratified split** logic in `create_70_10_20_split_dialogue_level.py`, and the **training** stack that consumes the splits (e.g. **LLaMA-3.1-8B-Instruct** for SFT / multi-objective / DPO — see `q1_3stage_pipeline/configs/pipeline_default.yaml` and `REPORT_3STAGE_PIPELINE.md`).
- **Practical answer for reports:** if you need the exact **generating model / pipeline** for the gold dialogues, recover it from the **parent experiment repository** (Exp3 / finetuning project referenced in README) or from internal lab notes; cite this file as “downstream split only” if those artifacts are unavailable.

---

## 4. Prompting strategy

- **Global template (all stages):** strict delimiter format  
  `[USER]: …`  
  `[ASSISTANT]:`  
  (see `q1_3stage_pipeline/utils/prompt_format.py` and README “Global formatting contract”).
- **Supervision:** causal LM loss masks **prompt** tokens; only the **assistant** continuation is trained (standard SFT masking).
- **Optional Stage 2 prefix:** `--lang-tag-prefix` (e.g. `[HI_EN_LEGAL]`) can be prepended to every user prompt without editing JSONL files.

---

## 5. Dataset evaluation strategy

- **Split protocol:** **Dialogue-level** 70/10/20 stratified split by `language`, `complexity`, and `bucket` keys so that train/val/test stay balanced and **turns from the same dialogue do not leak across splits** (`create_70_10_20_split_dialogue_level.py`).
- **Strict protocol (README / REPORT):** tune on **train** + **val** only; **test** is reserved for **final** evaluation after choosing hyperparameters (e.g. DPO \(\beta\)); optional **final_train = train + val** for the last M2/M3 retrain.
- **Automatic metrics (`evaluation/run_eval.py`):**
  - Text overlap: **ROUGE / BLEU / METEOR** (`evaluation/metrics.py`).
  - **NLI-style** consistency score between reference and candidate (`calculate_nli_score`).
  - **Statute correctness proxy:** match extracted section numbers in the candidate vs `statutes_cited` (`evaluation/legal_metrics.py` — regex-based, not a full legal validator).
  - **Safety proxies:** simple regex flags for harmful phrases and refusal-like patterns (`evaluation/safety_metrics.py`).
- **Operational requirement:** predictions JSONL must align **in order** with the test JSONL (`run_eval.py`).

---

## 6. How multi-turn dialogues were created (structure → training pairs)

- Each JSONL row is one **dialogue** with a `turns` array: alternating `user` / `assistant` messages.
- **Flattening for SFT / Stage 2 / DPO preference building** uses a **rolling window** (`q1_3stage_pipeline/utils/dataset_builder.py`):
  - After each **assistant** turn, emit one training example:  
    **prompt** = full history so far + trailing `\n[ASSISTANT]:`  
    **output** = that assistant’s reply.
  - So a dialogue with \(T\) assistant turns yields **\(T\)** supervised pairs (in-memory; not stored as separate rows in the JSONL).

---

## 7. Dataset structure (JSONL schema)

Each dialogue line is a JSON object; fields observed in shipped data include:

| Field | Description |
|--------|-------------|
| `dialogue_id` | Unique ID (often encodes language / bucket / case). |
| `language` | e.g. `hindi`, `english`, `code_mixed`. |
| `complexity` | `layman`, `intermediate`, or `professional`. |
| `turn_count` | Integer count of turns (informational). |
| `turns` | List of `{ "role": "user"\|"assistant", "text": "..." }`. |
| `statutes_cited` | List of strings naming statutes/sections cited for that dialogue. |
| `bucket` | Stratification bucket label (`A`, `B`, `C`, `D`). |
| `case_id` | Numeric case identifier. |

**Derived (not stored in JSONL):** flattened `prompt` / `output` pairs are built in memory by `DatasetBuilder`.

**Scale (computed):**

| Scope | Dialogues | Flattened prompt→answer pairs |
|--------|------------|----------------------------------|
| Train (70%) | 840 | 3256 |
| Val (10%) | 120 | 471 |
| Test (20%) | 240 | 950 |
| **Train + Val** | 960 | 3727 |
| **All (train+val+test)** | **1200** | **4677** |

---

## 8. Dialogue categories: layman, intermediate, professional

- The **`complexity`** field labels each dialogue as **`layman`**, **`intermediate`**, or **`professional`** (legal depth / assumed audience).

---

## 9. Number of samples in each category

**Overall (1200 dialogues):**

| Complexity | Count |
|------------|------:|
| layman | 399 |
| intermediate | 399 |
| professional | 402 |

**By split:**

| Split | Total | layman | intermediate | professional |
|-------|------:|-------:|---------------:|---------------:|
| Train | 840 | 279 | 279 | 282 |
| Val | 120 | 39 | 39 | 42 |
| Test | 240 | 81 | 81 | 78 |

**By language (1200 dialogues, balanced):** `hindi` 400, `english` 400, `code_mixed` 400.

**By `bucket` (stratification key):** `A` 300, `B` 313, `C` 321, `D` 266.

---

## 10. Which statutes appear in the dataset (e.g. POCSO, IPC)

- **Explicitly named** in dialogue text and/or `statutes_cited` strings (non-exhaustive): **POCSO**, **IPC**, **CrPC**, **JJ Act (Juvenile Justice)**, **SC/ST Act**, **MTP Act**, **Dowry Prohibition Act**, **Legal Services Authorities Act**, and many **section-level** references (e.g. “POCSO Section 19”, “IPC धारा 376”).
- **Repository tooling** for evaluation extracts numeric section IDs via regex patterns for **IPC / Section / S.** style citations (`evaluation/legal_metrics.py`) — it does **not** ship a full statute ontology.

**Coarse frequency (string-level, all list entries across 1200 dialogues):**  
Counts below count **how many times** a family keyword appears in any `statutes_cited` string (one dialogue can contribute multiple hits):

| Family (keyword scan) | ~Mentions in `statutes_cited` lists |
|------------------------|--------------------------------------:|
| POCSO | 4215 |
| IPC | 2236 |
| CrPC | 1837 |
| JJ Act | 614 |
| SC/ST Act | 29 |
| MTP Act | 23 |
| Dowry Prohibition | 20 |
| Legal Services Authorities | 17 |

*Interpretation:* **POCSO + IPC + CrPC** dominate; other acts appear as needed by scenarios.

---

## 11. Average statutes per dialogue

- **Mean** number of entries in `statutes_cited` per dialogue: **≈ 8.45**
- **Min / max** entries per dialogue: **0** / **26**
- **Distinct `statutes_cited` strings** (exact string match, truncated uniqueness not applied): **2417** unique strings across the corpus

---

## 12. Ethical and privacy measures (if any)

**In the dataset text**

- Placeholders such as **`[Victim]`**, **`[Accused]`** appear instead of real names in many turns (pseudonymization style).
- Content is framed as **legal information / process explanation**, not as instructions to commit harm; models are still evaluated with lightweight **safety/refusal proxies** (`evaluation/safety_metrics.py`).

**In the research / code protocol**

- **No test leakage:** dialogue-level splits + README rule: test only for final reporting after tuning.
- **Secrets:** Hugging Face tokens must be supplied via **environment variables**, not committed to git (documented in README).
- **Large artifacts:** checkpoints and logs are typically **gitignored**; avoids accidental publication of bulky or run-specific outputs.

**Limitations (be transparent in write-ups)**

- Regex **statute “correctness”** is only a **proxy**; it is **not** a substitute for lawyer review or official legal databases.
- **Safety regexes** are narrow; they catch only a small class of harmful patterns.

---

## Additional notes (useful for reports)

- **Base LM for training (not necessarily the dialogue author):** `meta-llama/Meta-Llama-3.1-8B-Instruct` (`pipeline_default.yaml`).
- **Stage 2 teachers:** frozen **DeBERTa-large-MNLI** (entailment teacher) and **sentence-transformers/all-mpnet-base-v2** (embedding / triplet mining) — see `REPORT_3STAGE_PIPELINE.md`.
- **Stage 3:** DPO with **chosen = gold**, **rejected = dynamic hard negatives**; reference = M2.
- **Orchestration:** `run_full_pipeline.py` / `run_full_pipeline.sh` for sequential Stage 1 → 2 → 3.
- **Where to look in code:** `create_70_10_20_split_dialogue_level.py`, `q1_3stage_pipeline/utils/dataset_builder.py`, `q1_3stage_pipeline/utils/prompt_format.py`, `q1_3stage_pipeline/evaluation/`, `q1_3stage_pipeline/REPORT_3STAGE_PIPELINE.md`.

---

*Generated for project documentation. Re-run the aggregation script in §7 if the JSONL files change.*
