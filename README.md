# POD_Verifier

AWS Lambda pipeline for POD quality scoring: Metabase → single container Lambda (download to `/tmp` → EfficientNet inference) → PostgreSQL.

See **[aws/README.md](aws/README.md)** for architecture and deployment steps.

## Repository layout

- `aws/` — Combined pipeline source, Dockerfile, SAM template, PostgreSQL schema, env template.

Model weights (`*.pt`) are not committed; copy `best.pt` into `aws/lambda_scorer/model/` before building the container image.
