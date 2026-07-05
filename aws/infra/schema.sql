-- POD Scoring Pipeline — PostgreSQL Schema
-- Run once to create the table. Idempotent (IF NOT EXISTS).
--
-- v2 (single-invocation redesign): adds status/failure_reason so EVERY input row
-- gets an outcome, and a UNIQUE key so the pipeline can upsert (idempotent retries
-- + resume). See aws/SINGLE_INVOCATION_DESIGN.md §6.

CREATE TABLE IF NOT EXISTS pod_scores (
    id                   SERIAL PRIMARY KEY,
    awb                  VARCHAR(50) NOT NULL,
    trip_id              VARCHAR(50),
    pod_score            DOUBLE PRECISION,          -- NULL when status <> 'scored'
    pod_link             TEXT NOT NULL,
    context_valid_prob   DOUBLE PRECISION,
    package_visible_prob DOUBLE PRECISION,
    label_readable_prob  DOUBLE PRECISION,
    image_clarity_prob   DOUBLE PRECISION,
    status               VARCHAR(20) NOT NULL DEFAULT 'scored',  -- 'scored' | 'download_failed'
    failure_reason       TEXT,
    run_date             DATE NOT NULL,
    scored_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Idempotency / resume key: one row per image per day. Enables ON CONFLICT upsert.
CREATE UNIQUE INDEX IF NOT EXISTS uq_pod_scores_awb_link_date
    ON pod_scores (awb, pod_link, run_date);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_pod_scores_awb ON pod_scores (awb);
CREATE INDEX IF NOT EXISTS idx_pod_scores_run_date ON pod_scores (run_date);
CREATE INDEX IF NOT EXISTS idx_pod_scores_awb_run_date ON pod_scores (awb, run_date);
CREATE INDEX IF NOT EXISTS idx_pod_scores_trip_id ON pod_scores (trip_id);
CREATE INDEX IF NOT EXISTS idx_pod_scores_status ON pod_scores (status);

-- View for AWB-level flagging (parameterize threshold by replacing 0.7).
-- Only 'scored' rows drive the pass/flag decision; download failures are excluded
-- so operations never penalise an AWB on a failed fetch.
CREATE OR REPLACE VIEW pod_scores_flagged AS
WITH ranked AS (
    SELECT
        awb,
        trip_id,
        pod_score,
        pod_link,
        run_date,
        ROW_NUMBER() OVER (PARTITION BY awb, run_date ORDER BY pod_score DESC) AS img_num,
        MAX(pod_score) OVER (PARTITION BY awb, run_date) AS max_score
    FROM pod_scores
    WHERE status = 'scored' AND pod_score IS NOT NULL
)
SELECT
    awb,
    trip_id,
    run_date,
    max_score,
    CASE WHEN max_score >= 0.7 THEN 'PASS' ELSE 'FLAG' END AS flag,
    MAX(CASE WHEN img_num = 1 THEN pod_score END) AS pod_score_img_1,
    MAX(CASE WHEN img_num = 2 THEN pod_score END) AS pod_score_img_2,
    MAX(CASE WHEN img_num = 3 THEN pod_score END) AS pod_score_img_3,
    MAX(CASE WHEN img_num = 1 THEN pod_link END) AS pod_1,
    MAX(CASE WHEN img_num = 2 THEN pod_link END) AS pod_2,
    MAX(CASE WHEN img_num = 3 THEN pod_link END) AS pod_3
FROM ranked
GROUP BY awb, trip_id, run_date, max_score;
