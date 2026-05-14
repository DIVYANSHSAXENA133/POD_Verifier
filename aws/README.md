# POD scoring pipeline — AWS

Single container Lambda merges Metabase ingestion, POD image downloads (staged under `/tmp`), EfficientNet inference, PostgreSQL inserts, and batch CloudWatch summaries. Daily trigger via EventBridge Scheduler.

## Layout

- `lambda_scorer/` — Docker image (`handler.py`): full pipeline (`STAGING_TMP_PATH` defaults to `/tmp`).
- `infra/` — SAM template (`template.yaml`), PostgreSQL schema (`schema.sql`), `samconfig.toml`, and `ephemeral_peak_mb.py` for rough `/tmp` sizing.
- `config.env.example` — Environment variable reference (copy values into SAM parameters / Secrets Manager; do not commit secrets).

## Model artifact

Copy your trained checkpoint into the image build context (not committed to Git):

```bash
# From this repo's aws/ directory (paths relative to aws/)
cp /path/to/your/checkpoints/best.pt lambda_scorer/model/best.pt
```

`.pt` files are ignored by this repository's `.gitignore`.

## Ephemeral `/tmp` and `FETCH_BATCH_SIZE`

Lambda `/tmp` is capped by **`TmpEphemeralMB`** on the pipeline function (`5120` MiB default). Oversizing batches relative to typical image payloads can exhaust `/tmp`.

Run a quick sanity check locally (no dependencies):

```bash
python3 infra/ephemeral_peak_mb.py --fetch-batch-size 500 --avg-kb-per-image 200
```

Tune **`FetchBatchSize`** and **`TmpEphemeralMB`** in SAM accordingly.

## Deploy (outline)

1. Apply schema: `psql -f infra/schema.sql` against your RDS database.
2. Build and push the pipeline image to ECR (same URI you pass to SAM as **`ScorerImageUri`** parameter).
3. From `infra/`: `sam build` then `sam deploy --guided` (supply VPC, subnets, Metabase, RDS, ECR URI, optional `TmpEphemeralMB` overrides).

Schedule in the template runs at **18:25 UTC** (`cron(25 18 * * ? *)`), aligned with **23:55 IST**.

## GitHub Projects

After pushing, you can add a **Project** in GitHub (repository → Projects → New project) to track deployment tasks and credentials rotation.
