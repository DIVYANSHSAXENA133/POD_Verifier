"""
Tests for aws/lambda_scorer/handler.py (merged POD pipeline).

Run from `aws/lambda_scorer/`:
  pip install -r requirements.txt -r requirements-dev.txt
  pytest -v
"""

from __future__ import annotations

import importlib
import json
import os
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

import handler
from src.model import ATTRIBUTE_NAMES


def reload_handler():
    """Reload handler after env tweaks."""
    return importlib.reload(handler)


@pytest.fixture(autouse=True)
def reset_handler_model_cache():
    handler._model = None
    handler._device = None
    yield
    handler._model = None
    handler._device = None


class _FakeLogitModel(nn.Module):
    """Constant logits → sigmoid = 1/(1+e^-logit)."""

    def __init__(self, logit: float = 0.0):
        super().__init__()
        self.logit = logit

    def forward(self, x: torch.Tensor) -> dict:
        b = x.shape[0]
        dev = x.device
        return {name: torch.full((b,), self.logit, device=dev) for name in ATTRIBUTE_NAMES}


def test_expand_pod_links_uppercase_pod_column():
    df = pd.DataFrame(
        {
            "AWB": ["A1"],
            "Trip Id": ["T1"],
            "POD": [" https://cdn/a.jpg , https://cdn/b.jpg "],
        }
    )
    out = reload_handler().expand_pod_links(df)
    assert len(out) == 2
    assert list(out["pod_link"]) == ["https://cdn/a.jpg", "https://cdn/b.jpg"]
    assert out.iloc[0]["AWB"] == "A1"


def test_expand_pod_links_lowercase_pod_column():
    df = pd.DataFrame({"awb": ["x"], "pod": ["https://one.png"], "trip_id": ["1"]})
    out = reload_handler().expand_pod_links(df)
    assert len(out) == 1
    assert out.iloc[0]["pod_link"] == "https://one.png"


def test_expand_pod_links_skips_non_http_and_empty():
    df = pd.DataFrame({"POD": ["ftp://x, https://ok.jpg", "", None]})
    out = reload_handler().expand_pod_links(df)
    assert len(out) == 1


def test_expand_pod_links_raises_without_pod_columns():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="POD column not found"):
        reload_handler().expand_pod_links(df)


def test_fetch_pod_data_builds_frame():
    h = reload_handler()
    h.METABASE_URL = "https://mb.example.com"
    h.METABASE_API_KEY = "secret"
    h.METABASE_CARD_ID = 42

    mock_session = MagicMock()
    resp = MagicMock()
    resp.json.return_value = [{"col": "v"}]
    resp.raise_for_status = MagicMock()
    mock_session.post.return_value = resp

    out = h.fetch_pod_data(mock_session)
    mock_session.post.assert_called_once()
    assert len(out) == 1
    url = mock_session.post.call_args[0][0]
    assert "/api/card/42/query/json" in url


def test_download_batch_skips_when_body_too_small(tmp_path):
    h = reload_handler()
    batch = pd.DataFrame({"pod_link": ["https://a.jpg"], "AWB": ["1"], "Trip Id": ["t"]})
    session = MagicMock()
    session.get.return_value = MagicMock(status_code=200, content=b"x" * 100)

    dest = tmp_path / "imgs"
    manifest = h.download_batch_to_dir(session, batch, str(dest), start_idx=0)
    assert manifest == []
    if dest.exists():
        assert os.listdir(dest) == []


def test_download_batch_writes_file_and_manifest(tmp_path):
    h = reload_handler()
    img_dir = tmp_path / "d"
    url = "https://host/pod.JPG?q=1"
    batch = pd.DataFrame({"pod_link": [url], "AWB": ["AWB9"], "Trip Id": ["TR123"]})

    blob = os.urandom(600)
    session = MagicMock()
    session.get.return_value = MagicMock(status_code=200, content=blob)

    manifest = h.download_batch_to_dir(session, batch, str(img_dir), start_idx=7)

    assert len(manifest) == 1
    assert manifest[0]["filename"] == "pod_7.JPG"
    assert manifest[0]["awb"] == "AWB9"
    assert manifest[0]["trip_id"] == "TR123"

    filepath = img_dir / "pod_7.JPG"
    assert filepath.is_file()
    assert filepath.read_bytes() == blob


def test_download_batch_long_fake_extension_maps_to_png(tmp_path):
    h = reload_handler()
    awful = "https://x.invalid/" + "a" * 10 + "?x=1"
    batch = pd.DataFrame({"pod_link": [awful], "AWB": ["a"], "Trip Id": ["t"]})
    session = MagicMock()
    session.get.return_value = MagicMock(status_code=200, content=b"y" * 600)

    manifest = h.download_batch_to_dir(session, batch, str(tmp_path), start_idx=0)
    assert manifest[0]["filename"] == "pod_0.png"


def test_tmp_image_dataset_zeros_for_missing_file(tmp_path):
    h = reload_handler()
    ghost = str(tmp_path / "ghost_missing.png")
    ds = h.TmpImageDataset([ghost], input_size=64)
    tensor, out_path = ds[0]
    assert tensor.shape == (3, 64, 64)
    assert out_path == ghost


def test_tmp_image_dataset_reads_real_image(tmp_path, monkeypatch):
    h = reload_handler()
    monkeypatch.setattr(h, "INPUT_SIZE", 32)

    png = tmp_path / "real.png"
    img = np.zeros((40, 50, 3), dtype=np.uint8)
    img[:] = (10, 20, 30)
    cv2.imwrite(str(png), img)

    ds = h.TmpImageDataset([str(png)], input_size=32)
    tensor, out_path = ds[0]
    assert tensor.dtype == torch.float32
    assert tensor.shape == (3, 32, 32)


def test_score_batch_outputs_expected_columns(monkeypatch, tmp_path):
    h = reload_handler()
    monkeypatch.setattr(h, "BATCH_SIZE", 2)
    monkeypatch.setattr(h, "INPUT_SIZE", 32)

    img_path = tmp_path / "sc.png"
    cv2.imwrite(str(img_path), np.zeros((8, 8, 3), dtype=np.uint8))

    device = torch.device("cpu")
    model = _FakeLogitModel(0.0)
    df = h.score_batch(model, device, [str(img_path)])

    assert len(df) == 1
    expected_cols = {
        "image_path",
        "pod_score",
        "context_valid_prob",
        "package_visible_prob",
        "label_readable_prob",
        "image_clarity_prob",
    }
    assert set(df.columns) == expected_cols

    expected_prob = torch.sigmoid(torch.tensor(0.0)).item()
    np.testing.assert_allclose(df.iloc[0]["context_valid_prob"], expected_prob, rtol=1e-5)


def test_write_to_postgres_invokes_execute_values(monkeypatch):
    h = reload_handler()

    conn = MagicMock()
    fake_cur = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=fake_cur)
    ctx.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = ctx

    results = [{
        "awb": "1",
        "trip_id": "2",
        "pod_score": 0.9,
        "pod_link": "https://z",
        "context_valid_prob": 0.1,
        "package_visible_prob": 0.2,
        "label_readable_prob": 0.3,
        "image_clarity_prob": 0.4,
    }]

    with patch.object(h.psycopg2.extras, "execute_values") as ev:
        h.write_to_postgres(conn, results, "2026-05-01")

    ev.assert_called_once()
    assert ev.call_args[1].get("page_size") == 200
    conn.commit.assert_called_once()


def test_write_to_postgres_empty(monkeypatch):
    h = reload_handler()

    conn = MagicMock()
    fake_cur = MagicMock()
    ctx = MagicMock()
    ctx.__enter__ = MagicMock(return_value=fake_cur)
    ctx.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = ctx

    with patch.object(h.psycopg2.extras, "execute_values") as ev:
        h.write_to_postgres(conn, [], "2099-01-01")

    ev.assert_called_once()


def test_handler_requires_metabase_env(monkeypatch):
    monkeypatch.delenv("METABASE_URL", raising=False)
    monkeypatch.delenv("METABASE_API_KEY", raising=False)
    monkeypatch.setenv("METABASE_CARD_ID", "1")
    h = reload_handler()
    rsp = h.handler({}, MagicMock(aws_request_id="req-env"))
    assert rsp["statusCode"] == 500


def test_handler_empty_metabase_response(monkeypatch, tmp_path):
    monkeypatch.setenv("METABASE_URL", "https://mb.fake")
    monkeypatch.setenv("METABASE_API_KEY", "secret")
    monkeypatch.setenv("STAGING_TMP_PATH", str(tmp_path))
    monkeypatch.setenv("METABASE_CARD_ID", "10989")

    h = reload_handler()
    fake_session = MagicMock()

    with patch.object(h, "build_session", return_value=fake_session), patch.object(h, "fetch_pod_data", return_value=pd.DataFrame()):
        rsp = h.handler({"i": 0}, MagicMock(aws_request_id="xyz"))

    body = json.loads(rsp["body"])
    assert rsp["statusCode"] == 200
    assert body.get("message") == "No data"


def test_handler_no_pod_links_after_expand(monkeypatch, tmp_path):
    monkeypatch.setenv("METABASE_URL", "https://mb.fake")
    monkeypatch.setenv("METABASE_API_KEY", "secret")
    monkeypatch.setenv("STAGING_TMP_PATH", str(tmp_path))

    h = reload_handler()

    with patch.object(h, "fetch_pod_data", return_value=pd.DataFrame({"POD": ["nope-not-url"]})):
        rsp = h.handler({"i": 0}, MagicMock(aws_request_id="q"))

    loaded = json.loads(rsp["body"])
    assert rsp["statusCode"] == 200
    assert loaded.get("message") == "No POD links"


def test_handler_loops_batches_in_memory_and_flushes_postgres_once(monkeypatch):
    monkeypatch.setenv("METABASE_URL", "https://mb.fake")
    monkeypatch.setenv("METABASE_API_KEY", "secret")
    monkeypatch.setenv("FETCH_BATCH_SIZE", "2")

    h = reload_handler()
    rows = [{"AWB": f"A{n}", "Trip Id": f"T{n}", "POD": f"https://u{n}.jpg"} for n in range(5)]

    dummy_model = MagicMock()
    dummy_device = torch.device("cpu")
    chunk_result = [{
        "awb": "x", "trip_id": "y", "pod_score": 0.5, "pod_link": "https://",
        "context_valid_prob": 0.1, "package_visible_prob": 0.2,
        "label_readable_prob": 0.3, "image_clarity_prob": 0.4,
    }]

    with ExitStack() as stack:
        stack.enter_context(patch.object(h, "build_session", return_value=MagicMock()))
        stack.enter_context(patch.object(h, "fetch_pod_data", return_value=pd.DataFrame(rows)))
        stack.enter_context(
            patch.object(
                h,
                "download_batch_to_memory",
                return_value=[{"awb": "x", "trip_id": "y", "pod_link": "https://", "image": object()}],
            ),
        )
        stack.enter_context(patch.object(h, "get_model", return_value=(dummy_model, dummy_device)))
        score_mock = stack.enter_context(
            patch.object(h, "score_samples_in_memory", return_value=chunk_result),
        )
        flush = stack.enter_context(patch.object(h, "_flush_results_to_postgres", return_value=0.01))

        rsp = h.handler({"i": 0}, MagicMock(aws_request_id="full"))

    assert score_mock.call_count == 3
    flush.assert_called_once()
    assert len(flush.call_args[0][0]) == 3

    summary = json.loads(rsp["body"])
    assert summary["total_images"] == 5
    assert summary["batches_processed"] == 3
    assert summary["final_i"] == 5
    assert summary["postgres_flushed"] is True
    assert summary["total_scored_rows"] == 3


def test_process_one_batch_no_files_returns_zero(monkeypatch, tmp_path):
    h = reload_handler()
    monkeypatch.setattr(h, "get_db_connection", MagicMock())

    stats = h.process_one_batch(
        None,
        torch.device("cpu"),
        [{"filename": "gone.jpg", "awb": "", "trip_id": "", "pod_link": ""}],
        str(tmp_path / "nosuch"),
        0,
        0.5,
        "2099-01-01",
    )
    assert stats["scored"] == 0
    h.get_db_connection.assert_not_called()
