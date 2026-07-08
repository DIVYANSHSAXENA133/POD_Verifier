"""
Test suite for the single-invocation POD pipeline handler.

Design goals covered:
  * expand/dedup correctness
  * concurrent download partitions success vs failure, never drops an input
  * idempotent upsert SQL shape (ON CONFLICT) + failure rows recorded
  * resume-from-checkpoint skips already-scored rows
  * WHOLE-DATASET COVERAGE in one invocation with ample time (no timeout)
  * clock-aware continuation fires (and only fires) near the wall
  * preprocessing shape + normalization

Pipeline-logic tests mock the model and DB, so they run without torch or a
database. A separate torch-gated test exercises real inference wiring.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import handler as H  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers / fakes
# --------------------------------------------------------------------------- #

def _png_bytes(w=32, h=32):
    import cv2
    img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


class FakeResp:
    def __init__(self, content=b"", status=200):
        self.content = content
        self.status_code = status


class FakeSession:
    """Returns a valid PNG for most URLs; configurable failures."""
    def __init__(self, fail_urls=None, http_status=None):
        self.fail_urls = fail_urls or set()
        self.http_status = http_status or {}

    def get(self, url, timeout=0):
        if url in self.fail_urls:
            raise ConnectionError("boom")
        if url in self.http_status:
            return FakeResp(_png_bytes(), self.http_status[url])
        return FakeResp(_png_bytes(), 200)


class FakeCursor:
    def __init__(self, conn):
        self.conn = conn
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def execute(self, sql, params=None):
        self.conn._last_sql = sql
    def fetchall(self):
        return list(self.conn.done_rows)


class FakeConn:
    def __init__(self, done_rows=None):
        self.done_rows = done_rows or []
        self.upserted = []
        self.committed = 0
    def cursor(self):
        return FakeCursor(self)
    def commit(self):
        self.committed += 1
    def close(self):
        pass


class FakeContext:
    def __init__(self, remaining_ms):
        self._r = remaining_ms
    def get_remaining_time_in_millis(self):
        return self._r


def _fake_score_prepared(model, device, successes):
    out = []
    for s in successes:
        out.append({
            "awb": s["awb"], "trip_id": s["trip_id"], "pod_link": s["pod_link"],
            "status": "scored", "failure_reason": None, "pod_score": 0.9,
            "context_valid_prob": 0.9, "package_visible_prob": 0.9,
            "label_readable_prob": 0.9, "image_clarity_prob": 0.9,
        })
    return out


@pytest.fixture
def wire(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(H, "get_model", lambda: (object(), "cpu"))
    monkeypatch.setattr(H, "get_db_connection", lambda: conn)
    monkeypatch.setattr(H, "score_prepared", _fake_score_prepared)
    monkeypatch.setattr(H, "emit_coverage", lambda *a, **k: None)

    def _capture_upsert(c, rows, run_date):
        c.upserted.extend(rows)
        return len(rows)
    monkeypatch.setattr(H, "upsert_results", _capture_upsert)
    monkeypatch.setattr(H, "SOURCE_QUERY", "SELECT 1", raising=False)
    monkeypatch.setattr(H, "PG_HOST", "db", raising=False)
    monkeypatch.setattr(H, "PG_PASSWORD", "pw", raising=False)
    return conn


def _mb_df(n):
    return pd.DataFrame([
        {"AWB": f"AWB{i}", "Trip Id": f"T{i}", "POD": f"http://img/{i}.png"}
        for i in range(n)
    ])


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #

def test_expand_dedup_and_filter():
    df = pd.DataFrame([
        {"AWB": "A1", "Trip Id": "T1", "POD": "http://a/1.png, http://a/2.png"},
        {"AWB": "A1", "Trip Id": "T1", "POD": "http://a/1.png"},
        {"AWB": "A2", "Trip Id": "T2", "POD": "not_a_url, http://a/3.png"},
        {"AWB": "A3", "Trip Id": "T3", "POD": ""},
    ])
    out = H.expand_pod_links(df)
    assert set(out["pod_link"]) == {"http://a/1.png", "http://a/2.png", "http://a/3.png"}
    assert len(out) == 3


def test_preprocess_shape_and_normalization():
    img = (np.random.rand(50, 40, 3) * 255).astype(np.uint8)
    chw = H.preprocess_image(img, size=224, normalize=True)
    assert chw.shape == (3, 224, 224)
    assert chw.dtype == np.float32
    raw = H.preprocess_image(img, size=224, normalize=False)
    assert raw.min() >= 0.0 and raw.max() <= 1.0
    assert chw.min() < 0.0


def test_download_and_prepare_outcomes():
    s = FakeSession(fail_urls={"http://img/x.png"}, http_status={"http://img/y.png": 404})
    ok = H.download_and_prepare(s, {"awb": "a", "trip_id": "t", "pod_link": "http://img/ok.png"})
    assert ok["status"] == "scored" and "chw" in ok
    exc = H.download_and_prepare(s, {"awb": "a", "trip_id": "t", "pod_link": "http://img/x.png"})
    assert exc["status"] == "download_failed" and exc["failure_reason"] == "ConnectionError"
    http = H.download_and_prepare(s, {"awb": "a", "trip_id": "t", "pod_link": "http://img/y.png"})
    assert http["status"] == "download_failed" and http["failure_reason"] == "http_404"


def test_download_window_partitions_all_inputs():
    rows = [{"awb": f"a{i}", "trip_id": "t", "pod_link": f"http://img/{i}.png"} for i in range(20)]
    s = FakeSession(fail_urls={"http://img/3.png", "http://img/9.png"})
    succ, fail = H.download_window(s, rows)
    assert len(succ) + len(fail) == 20
    assert {f["pod_link"] for f in fail} == {"http://img/3.png", "http://img/9.png"}


def test_upsert_sql_is_idempotent(monkeypatch):
    captured = {}
    def fake_execute_values(cur, sql, tuples, page_size=100):
        captured["sql"] = sql
        captured["tuples"] = tuples
    monkeypatch.setattr(H.psycopg2.extras, "execute_values", fake_execute_values)
    conn = FakeConn()
    rows = [{"awb": "a", "trip_id": "t", "pod_link": "u", "status": "scored",
             "pod_score": 0.8, "context_valid_prob": 0.1, "package_visible_prob": 0.2,
             "label_readable_prob": 0.3, "image_clarity_prob": 0.4}]
    n = H.upsert_results(conn, rows, "2026-07-05")
    assert n == 1
    assert "ON CONFLICT (awb, pod_link, run_date) DO UPDATE" in captured["sql"]
    assert conn.committed == 1


def test_load_done_keys():
    conn = FakeConn(done_rows=[("A1", "u1"), ("A2", "u2")])
    assert H.load_done_keys(conn, "2026-07-05") == {("A1", "u1"), ("A2", "u2")}


# --------------------------------------------------------------------------- #
# Handler-level: coverage, resume, continuation
# --------------------------------------------------------------------------- #

def test_handler_covers_whole_dataset_one_invocation(wire, monkeypatch):
    N = 2500
    monkeypatch.setattr(H, "fetch_pod_data", lambda: _mb_df(N))
    monkeypatch.setattr(H, "build_session", lambda: FakeSession())
    monkeypatch.setattr(H, "WINDOW_SIZE", 800, raising=False)

    resp = H.handler({}, FakeContext(600_000))
    body = json.loads(resp["body"])
    assert body["status"] == "complete"
    assert body["scored_this_invocation"] == N
    assert body["failed_this_invocation"] == 0
    assert len(wire.upserted) == N
    assert body["continuation"] == 0


def test_handler_records_failures_as_outcomes(wire, monkeypatch):
    N = 100
    fail = {f"http://img/{i}.png" for i in (5, 10, 42)}
    monkeypatch.setattr(H, "fetch_pod_data", lambda: _mb_df(N))
    monkeypatch.setattr(H, "build_session", lambda: FakeSession(fail_urls=fail))
    resp = H.handler({}, FakeContext(600_000))
    body = json.loads(resp["body"])
    assert body["scored_this_invocation"] == 97
    assert body["failed_this_invocation"] == 3
    assert body["scored_this_invocation"] + body["failed_this_invocation"] == N
    assert len([r for r in wire.upserted if r["status"] == "download_failed"]) == 3


def test_handler_resume_skips_done(wire, monkeypatch):
    N = 50
    monkeypatch.setattr(H, "fetch_pod_data", lambda: _mb_df(N))
    monkeypatch.setattr(H, "build_session", lambda: FakeSession())
    done = {(f"AWB{i}", f"http://img/{i}.png") for i in range(20)}
    monkeypatch.setattr(H, "load_done_keys", lambda c, d: done)
    resp = H.handler({}, FakeContext(600_000))
    body = json.loads(resp["body"])
    assert body["already_done_at_start"] == 20
    assert body["scored_this_invocation"] == 30
    assert len(wire.upserted) == 30


def test_handler_continuation_near_wall(wire, monkeypatch):
    N = 3000
    monkeypatch.setattr(H, "fetch_pod_data", lambda: _mb_df(N))
    monkeypatch.setattr(H, "build_session", lambda: FakeSession())
    monkeypatch.setattr(H, "WINDOW_SIZE", 500, raising=False)
    monkeypatch.setattr(H, "CONTINUATION_SAFETY_MS", 90_000, raising=False)
    called = {}
    monkeypatch.setattr(H, "invoke_continuation",
                        lambda rid, rd, c: called.update(run_id=rid, cont=c))

    class Ctx:
        def __init__(self):
            self.calls = 0
        def get_remaining_time_in_millis(self):
            self.calls += 1
            return 600_000 if self.calls == 1 else 10_000

    resp = H.handler({}, Ctx())
    body = json.loads(resp["body"])
    assert body["status"] == "continuing"
    assert called["cont"] == 1
    assert body["scored_this_invocation"] == 500


# --------------------------------------------------------------------------- #
# Torch-gated: real inference wiring
# --------------------------------------------------------------------------- #

torch = pytest.importorskip("torch", reason="torch not installed")

def test_score_prepared_real_math():
    import torch as T
    from src.model import ATTRIBUTE_NAMES

    class FakeModel:
        def __call__(self, batch):
            b = batch.shape[0]
            return {n: T.zeros(b) for n in ATTRIBUTE_NAMES}

    successes = [{"awb": "a", "trip_id": "t", "pod_link": "u",
                  "chw": np.zeros((3, 224, 224), dtype=np.float32)} for _ in range(3)]
    out = H.score_prepared(FakeModel(), T.device("cpu"), successes)
    assert len(out) == 3
    assert abs(out[0]["pod_score"] - 0.5) < 1e-6
    assert abs(out[0]["context_valid_prob"] - 0.5) < 1e-6
