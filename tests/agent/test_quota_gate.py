"""Tests for the dynamic quota gate (nanobot.agent.quota_gate).

Covers:
- QuotaLimit / QuotaSnapshot math (requests-only, tokens-only, both).
- Week-edge resume scheduling (window + weekly both tight).
- Stale window-reset roll-forward.
- fit_iterations shrinking.
- Fail-open fetch orchestration (unknown provider, missing key, fetch error).
- YuanyuAI fetcher parsing (live-shaped payload, offline via mock).
- Session-metadata round-trip (snapshot, suspension, resume job).
- User-facing status/notice; ensures quota is never injected into model context.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from nanobot.agent import quota_gate as qg


def _snap(
    *,
    remain_w=500,
    limit_w=500,
    reset_w=1_800_000_000,
    remain_t=6000,
    limit_t=6000,
    reset_t=1_803_600_000,
    provider="custom_yuanyuai",
    model="glm-5.3",
) -> qg.QuotaSnapshot:
    return qg.QuotaSnapshot(
        provider=provider,
        model=model,
        fetched_at=time.time(),
        requests=qg.QuotaLimit(remain=remain_w, limit=limit_w, next_reset=reset_w, label="5h window"),
        tokens=qg.QuotaLimit(remain=remain_t, limit=limit_t, next_reset=reset_t, label="weekly"),
    )


def test_quota_limit_has_limit_and_ratio() -> None:
    limit = qg.QuotaLimit(remain=400, limit=500)
    assert limit.has_limit is True
    assert limit.used_ratio == pytest.approx(0.2)
    assert qg.QuotaLimit(remain=0, limit=0).has_limit is False


def test_snapshot_blocking_requests_only() -> None:
    s = _snap(remain_w=1, limit_w=500)
    blocking = s.blocking()
    assert blocking is not None
    assert blocking.label == "5h window"
    assert s.exhausted() is True


def test_snapshot_blocking_tokens_only() -> None:
    s = _snap(remain_w=500, remain_t=500, limit_t=6000)
    blocking = s.blocking()
    assert blocking is not None
    assert blocking.label == "weekly"


def test_snapshot_not_blocking_when_room() -> None:
    s = _snap()
    assert s.blocking() is None
    assert s.exhausted() is False


def test_snapshot_no_limits_never_blocks() -> None:
    s = qg.QuotaSnapshot(provider="p", model="m")
    assert s.blocking() is None
    assert s.exhausted() is False


def test_next_resume_requests_only() -> None:
    s = _snap(remain_w=0, reset_w=1_800_000_000, remain_t=6000)
    assert s.next_resume_ts(now=1_799_000_000) == 1_800_000_000


def test_next_resume_weekly_only() -> None:
    s = _snap(remain_w=500, remain_t=1, reset_t=1_803_600_000)
    assert s.next_resume_ts(now=1_799_000_000) == 1_803_600_000


def test_next_resume_both_tight_week_edge() -> None:
    # Window resets every 5h anchored at 1_800_000_000; weekly resets 1 week
    # later at 1_800_604_800. The first window slot at/after the weekly reset
    # is 1_800_000_000 + 34*5h = 1_800_612_000.
    window_reset = 1_800_000_000
    weekly_reset = window_reset + 7 * 24 * 3600
    expected = window_reset + 34 * 5 * 3600
    s = _snap(remain_w=0, reset_w=window_reset, remain_t=1, reset_t=weekly_reset)
    assert s.next_resume_ts(now=1_799_000_000) == expected
    # Weekly already reset but window still tight -> next window slot.
    assert s.next_resume_ts(now=weekly_reset + 1) == expected


def test_next_resume_stale_window_rolls_forward() -> None:
    # Window reset is already in the past; the next fixed slot must be used.
    now = 1_800_000_000 + 3 * 3600  # 3h after the last reset
    s = _snap(remain_w=0, reset_w=1_800_000_000, remain_t=6000)
    assert s.next_resume_ts(now=now) == 1_800_000_000 + 5 * 3600


def test_next_resume_none_when_no_future_reset() -> None:
    s = _snap(remain_w=0, reset_w=0, remain_t=0, reset_t=0)
    assert s.next_resume_ts(now=1_799_000_000) is None


def test_fit_iterations_shrinks_to_room() -> None:
    s = _snap(remain_w=10)
    assert qg.fit_iterations(60, s) == 10 - 2  # minus default buffer


def test_fit_iterations_no_limits_keeps_requested() -> None:
    s = qg.QuotaSnapshot(provider="p", model="m")
    assert qg.fit_iterations(60, s) == 60


def test_fit_iterations_never_below_one() -> None:
    s = _snap(remain_w=1)
    assert qg.fit_iterations(60, s) == 1


# ---------------------------------------------------------------------------
# fetch orchestration (fail-open)
# ---------------------------------------------------------------------------

def test_fetch_for_model_unknown_provider_returns_none() -> None:
    assert qg.fetch_for_model(provider_name="not-a-provider", model="m", api_key="k") is None


def test_fetch_for_model_missing_key_returns_none() -> None:
    assert qg.fetch_for_model(provider_name="custom_yuanyuai", model="m", api_key=None) is None


def test_fetch_for_model_error_returns_none() -> None:
    def boom(api_key: str, model: str) -> qg.QuotaSnapshot:
        raise qg.QuotaFetchError("down")

    qg.register_quota_fetcher("boom_provider", boom)
    try:
        assert qg.fetch_for_model(provider_name="boom_provider", model="m", api_key="k") is None
    finally:
        qg._FETCHERS.pop("boom_provider", None)


def test_fetch_for_model_success() -> None:
    s = _snap()
    qg.register_quota_fetcher("ok_provider", lambda api_key, model: s)
    try:
        assert qg.fetch_for_model(provider_name="ok_provider", model="m", api_key="k") is s
    finally:
        qg._FETCHERS.pop("ok_provider", None)


# ---------------------------------------------------------------------------
# YuanyuAI fetcher
# ---------------------------------------------------------------------------

def _quota_payload(**overrides) -> dict:
    payload = {
        "success": True,
        "data": {
            "key": "sk-xxx",
            "name": "glm5.3-500次-sdfjyb",
            "group": "周限月卡500次",
            "status": 1,
            "remark": "",
            "unlimited": True,
            "rateLimit": 500,
            "windowCalls": 100,
            "windowRemain": 400,
            "billedCalls": 100,
            "realCalls": 100,
            "weeklyLimit": 6000,
            "weeklyBilled": 1200,
            "weeklyReal": 1200,
            "weeklyRemain": 4800,
            "weeklyPercent": 20.0,
            "nextWindowReset": 1791064800,
            "nextWeeklyReset": 1791129600,
            "expiredTime": -1,
            "currentMultiplier": 1.2,
            "multiplierLabel": "×1.2 (19:00-12:59)",
        },
    }
    data = payload["data"]
    data.update(overrides)
    return payload


def test_fetch_yuanyuai_quota_parses_live_shape() -> None:
    captured: dict = {}

    def fake_get_json(url, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        return _quota_payload()

    with patch.object(qg, "_http_get_json", side_effect=fake_get_json):
        s = qg.fetch_yuanyuai_quota("sk-xxx", "glm-5.3")

    assert s.provider == "custom_yuanyuai"
    assert s.model == "glm-5.3"
    assert s.requests.remain == 400
    assert s.requests.limit == 500
    assert s.requests.next_reset == 1791064800
    assert s.tokens.remain == 4800
    assert s.tokens.limit == 6000
    assert s.tokens.next_reset == 1791129600
    assert s.status == "active"
    assert captured["url"] == "https://yuanyuaicloud.cn/api/query-quota"
    assert captured["headers"]["Authorization"] == "Bearer sk-xxx"


def test_fetch_yuanyuai_quota_missing_key_raises() -> None:
    with pytest.raises(qg.QuotaFetchError):
        qg.fetch_yuanyuai_quota("", "glm-5.3")


def test_fetch_yuanyuai_quota_http_error_fails_open() -> None:
    with patch.object(qg, "_http_get_json", side_effect=qg.QuotaFetchError("HTTP 500")):
        with pytest.raises(qg.QuotaFetchError):
            qg.fetch_yuanyuai_quota("sk-xxx", "glm-5.3")


# ---------------------------------------------------------------------------
# session metadata round-trip
# ---------------------------------------------------------------------------

def test_snapshot_metadata_round_trip() -> None:
    s = _snap()
    meta: dict = {}
    qg.write_snapshot(meta, s)
    restored = qg.read_snapshot(meta)
    assert restored is not None
    assert restored.requests.remain == s.requests.remain
    assert restored.tokens.next_reset == s.tokens.next_reset
    assert restored.provider == s.provider


def test_suspension_metadata_round_trip() -> None:
    meta: dict = {}
    qg.write_suspension(
        meta,
        reason="window",
        resume_at=1_800_000_000,
        provider="custom_yuanyuai",
        model="glm-5.3",
        job_id="abc123",
    )
    susp = qg.read_suspension(meta)
    assert susp is not None
    assert susp["reason"] == "window"
    assert susp["resume_at"] == 1_800_000_000
    assert qg.resume_job_id(meta) == "abc123"

    qg.clear_suspension(meta)
    assert qg.read_suspension(meta) is None
    assert qg.resume_job_id(meta) is None


# ---------------------------------------------------------------------------
# presentation (user-facing only; never model context)
# ---------------------------------------------------------------------------

def test_status_lines_user_facing() -> None:
    s = _snap(remain_w=400, remain_t=4800)
    lines = qg.status_lines(s)
    assert any("5h window" in line and "400/500" in line for line in lines)
    assert any("weekly" in line and "4800/6000" in line for line in lines)


def test_status_payload_shape() -> None:
    s = _snap()
    susp = {"reason": "window", "resume_at": 1_800_000_000}
    payload = qg.status_payload(s, susp)
    assert payload["suspended"] is True
    assert payload["requests"]["remain"] == 500
    assert payload["suspension"]["resume_display"]
    assert "lines" in payload


def test_suspension_notice_includes_reset() -> None:
    s = _snap(remain_w=0, reset_w=1_791_064_800)  # 2026-10-04 06:00 CST
    notice = qg.suspension_notice(s, "rate limit")
    assert "paused" in notice
    assert "resume automatically" in notice
    assert "2026-10-04" in notice  # reset time rendered


def test_no_quota_context_injection_helpers() -> None:
    """Quota must be decision + user display, not model prompt context."""
    assert not hasattr(qg, "format_context_block")
    assert not hasattr(qg, "QUOTA_CONTEXT_SOURCE")
    assert not hasattr(qg, "RuntimeContextBlock")
