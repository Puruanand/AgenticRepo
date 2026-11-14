# Instructor README — Agent Memory in 2 Hours

This edition is intentionally designed for **typing live**. It uses **a real PR**, **three actual skill files**, **PostgreSQL + pgvector**, and the actual **OpenAI LLM**. It never inserts invented historical reviews or silently returns all records for a failed search.

## Timeline: only six demonstrations

| Minutes | What you type | What to explain | Evidence/conclusion |
|---|---|---|---|
| 00–10 | Notebook introduction | Short-term, semantic, procedural, episodic | Understand the four terms |
| 10–25 | Cell: short-term graph | `thread_id` isolates chats | A recalls Python; B does not |
| 25–40 | Docker commands + store setup | PostgreSQL, embeddings, pgvector | DB ready; 1536-dim real embedding |
| 40–65 | GitHub fetch + skill ingestion + search | Real sources; vector similarity ≠ key lookup | URLs, names, similarity scores |
| 65–85 | Real first PR review | Skill rules + `+` lines vs diff context | Review JSON; episode saved |
| 85–110 | Re-review + vector retrieval | Observe retrieval/citation; no fabricated findings | Prior review found by vector search |
| 110–120 | SQL proof + Q&A | Stored vs retrieved vs cited vs causal | SQL row + trace + conclusion |

## A. Preparation (do this BEFORE class)

1. In a terminal in the extracted project folder:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   cp .env.example .env
   ```

2. Edit `.env` with your real `OPENAI_API_KEY`. `GITHUB_TOKEN` is optional for a public repo, but helps avoid unauthenticated rate limits. Never put keys in notebook output or Git.
3. Open `Agent_Memory_2Hour_Live.ipynb`; select the `.venv` Python kernel. Run once before the lesson against a **test** database or project to check API credentials and PR availability.
4. PR #13 is the known example. You can change `PR_NUMBER` in `.env`; results depend on real current PR content.

## B. Create PostgreSQL live (students watch)

**First explain:** host port 5434 avoids the 5433 conflict you previously encountered. Inside the container, PostgreSQL still listens on 5432.

```bash
# Make sure Docker Desktop is running.
PG_HOST_PORT=5434 docker compose -p codesentinel-2hour up -d

docker compose -p codesentinel-2hour ps

docker compose -p codesentinel-2hour exec postgres \
  pg_isready -U demo -d agent_memory

# pgvector must be available; image already contains it.
docker compose -p codesentinel-2hour exec postgres \
  psql -U demo -d agent_memory -c 'CREATE EXTENSION IF NOT EXISTS vector;'

# Look at the empty database BEFORE store.setup().
docker compose -p codesentinel-2hour exec postgres \
  psql -U demo -d agent_memory -c '\dt'
```

**Expected:** readiness says `accepting connections`; `CREATE EXTENSION` succeeds. Before notebook initializes the store, LangGraph tables are absent. This is expected.

**IMPORTANT:** Don't delete Docker volumes from your earlier test project. `-p codesentinel-2hour` creates a separate project volume and avoids destroying prior data. If 5434 is occupied, pick 5435 and change `PG_HOST_PORT` **and** `.env` URI port.

## C. Teach the notebook in exact order

### 1 — Concepts (00–10)

Say: “An agent's memory isn't magic. Information is stored in a database; embeddings help it find similar information; the model gets retrieved text as context.”

### 2 — Short-term (10–25)

Type and run the short-term cell. **Show:** student-A remembers Python; student-B doesn't know it. **Conclusion:** thread isolation. This example is **not durable** across kernel restart.

### 3 — PostgreSQL store (25–40)

After Docker setup, execute store creation. **Show:** `Connected; real embedding dimension: 1536`. Explain why `store.setup()` is called. Run SQL inspection after first actual insert, not before.

### 4 — Semantic and procedural (40–65)

Fetch live PR #13 from GitHub. Explain `html_url`, `head.sha`, and the file patches. Store the **source-backed PR facts** as semantic memory. Read and store all three skills. Search with natural-language query, e.g. “rules for secure payment code”. **Show:** vector scores, record IDs and real paths/URL. **Conclusion:** IDs are storage identities; semantic retrieval is vector similarity.

### 5 — Episodic (65–85)

Review the actual PR using the policy. Print the model's structured findings and deterministic verdict. Only then save the resulting review with its URL and SHA as an episode. **Show:** episode ID, not invented prior findings. *Important*: PR #13 may only add a comment, so APPROVED may be correct under your policy.

### 6 — Recall and proof (85–110)

Build a two-node LangGraph (retrieve_memory → review_pr), then execute real vector search for a prior PR review. It should retrieve the record saved in Step 5. Run the second review, passing retrieved records. **Show:** retrieval IDs and scores, and the model's used_memory_ids (which are checked against retrieved IDs).

**Never claim:** “This changed the verdict” unless a controlled comparison demonstrates that. If `used_memory_ids` is empty, say retrieval occurred but the model did not cite the memory.

### 7 — SQL proof (110–120)

```bash
docker compose -p codesentinel-2hour exec -T postgres \
  psql -U demo -d agent_memory -f /sql/inspect.sql
```

**Expected:** namespaces with at least one semantic, three procedural, and one episodic record. SQL also displays PR URL, preview, pgvector extension, and vector table count. The exact counts depend on repeated runs.

**Conclusion:** `store` table proves stored data. Python retrieval trace proves query returned it. A validated memory ID in review suggests the LLM referenced it. No stronger causal claim without A/B comparison.

## D. What students should write at the end

1. **Short-term:** state persists only for a thread and only while the in-process checkpointer lives.
2. **Semantic:** facts sourced from real repository data are indexed.
3. **Procedural:** supplied skill documents provide review instructions.
4. **Episodic:** an actual previous review becomes a saved experience.
5. **Vector retrieval:** pgvector searches by meaning, not by memorized key.
6. **Evidence:** storage, retrieval, and usage are different proof levels.

## E. Troubleshooting

| Error | Explanation / fix |
|---|---|
| `Bind ... 5434 failed: port is already allocated` | Use 5435 in Docker port AND `.env` |
| `No module named langgraph.store.postgres` | Install requirements in current notebook kernel; restart kernel |
| `401` from OpenAI | Correct `OPENAI_API_KEY` in `.env`; restart kernel |
| `403` from GitHub | Token/rate-limit; add `GITHUB_TOKEN` or try later |
| `UndefinedTable: store` | Run `store.setup()` first |
| pgvector extension missing | Ensure `pgvector/pgvector:pg16` image, enable extension |
| Retrieval empty | Do not return every record; inspect namespace, rows and embeddings |
| Repeated runs contain records | Use separate Docker project and volume for a fresh classroom DB |
| Review cites no memory | Report retrieval-only evidence; don't claim model used it |

## F. Reconnect after kernel restart (persistence proof)

Run setup imports and store initialization, then **only** this code (do not rerun the first review-save cell):

```python
hits = store.search(
    ("class-memory", REPO, "episodic"),
    query=f"previous review of PR {PR_NUMBER}",
    limit=5,
)
for h in hits:
    print(h.key, h.score, h.value.get("source_url"))
```

If the prior episode appears, PostgreSQL data survived the kernel restart.

## G. Instructor safety notes

- Read-only project: **does not post GitHub comments**.
- Do not demonstrate by editing production code, or show API keys on screen.
- Same PR re-review is intentional to demonstrate recall, not broad knowledge transfer.
- Keep semantic and episodic labels accurate: PR source knowledge is semantic/source-derived; prior review is episodic.
- Actual outputs will vary with PR content, model version and embeddings.
