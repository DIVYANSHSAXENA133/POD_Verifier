# POD scoring pipeline — AWS

Two Lambda functions (Fetcher + Scorer), shared EFS for downloaded POD images, and PostgreSQL for per-image scores. Daily trigger via EventBridge Scheduler.

## Layout

- `lambda_fetcher/` — Metabase fetch, expand POD links, download to EFS, async-invoke Scorer per batch.
- `lambda_scorer/` — Container image: EfficientNet inference, PostgreSQL inserts, CloudWatch batch summaries.
- `infra/` — SAM template (`template.yaml`), PostgreSQL schema (`schema.sql`), `samconfig.toml`.
- `config.env.example` — Environment variable reference (copy values into SAM parameters / Secrets Manager; do not commit secrets).

## Model artifact

Copy your trained checkpoint into the image build context (not committed to Git):

```bash
# From this repo's aws/ directory (paths relative to aws/)
cp /path/to/your/checkpoints/best.pt lambda_scorer/model/best.pt
```

`.pt` files are ignored by this repository's `.gitignore`.

## Deploy (outline)

1. Apply schema: `psql -f infra/schema.sql` against your RDS database.
2. Build and push the Scorer image to ECR (same URI you pass to SAM as `ScorerImageUri`).
3. From `infra/`: `sam build` then `sam deploy --guided` (supply VPC, subnets, Metabase, RDS, ECR URI).

Schedule in the template runs at **18:25 UTC** (`cron(25 18 * * ? *)`), aligned with **23:55 IST**.

## GitHub Projects

After pushing, you can add a **Project** in GitHub (repository → Projects → New project) to track deployment tasks and credentials rotation.
