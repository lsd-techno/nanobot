"""Dynamic quota/limit fit and respect for agent sessions.

When a session starts (new or continued) and nanobot does not know the current
limits for the chosen model, this module fetches them from the model's quota
endpoint and uses them to bound the turn: the tool-iteration budget is fitted
to what is left so a turn cannot blow past the remaining allowance.

When limits run out mid-turn (billing / rate-limit provider errors, or a
pre-flight check that shows no room for even one more model call), the session
is suspended: a user-visible notice is delivered, the suspension is recorded in
session metadata, and a one-shot cron job is scheduled at the server's next
reset timestamp. The cron turn re-checks the quota first; if quota is back it
continues the session, otherwise it re-suspends with the new reset time.

Quota numbers are **never** written into the model prompt: they are internal
decision input plus user-facing status only. This keeps turns cheap (no extra
context tokens) and avoids the model treating a stale snapshot as fact.

Design constraints:
- Provider-agnostic core (``QuotaSnapshot`` in requests and/or tokens).
- Per-provider fetchers registered by provider name; unknown providers are a
  no-op (fail-open: the turn runs exactly as before).
- All network/parse failures are fail-open: a missing quota endpoint must
  never block a turn.
- No new third-party dependencies: stdlib ``urllib`` only.
- Suspension state lives in ``session.metadata`` under reserved keys so it
  survives restarts and is visible to SDK callers.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, TypedDict, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from loguru import logger

from nanobot.config_base import Base
from nanobot.cron.types import CronSchedule


class QuotaGateConfig(Base):
    """Dynamic quota/limit fit and respect for agent sessions."""

    enabled: bool = True
    suspend_on_exhaustion: bool = True
    shrink_iterations: bool = True
    min_request_buffer: int = 2
    min_token_buffer: int = 1000
    snapshot_ttl_seconds: float = 300.0
    fetch_timeout_seconds: float = 8.0


CST = timezone(timedelta(hours=8))

# Reserved session.metadata keys. The ``_nanobot_`` prefix keeps them out of
# the user-visible goal/blob namespace used by SDK callers.
QUOTA_SNAPSHOT_META_KEY = "_nanobot_quota_snapshot"
QUOTA_SUSPENDED_META_KEY = "_nanobot_quota_suspended"
QUOTA_RESUME_JOB_META_KEY = "_nanobot_quota_resume_job"

# YuanyuAI 5h windows reset at fixed times (01/06/11/16/21 Asia/Taipei).
WINDOW_RESET_PERIOD_S = 5 * 3600

# Fallback buffers used when a caller does not pass config-derived values.
DEFAULT_MIN_REQUEST_BUFFER = 2
DEFAULT_MIN_TOKEN_BUFFER = 1_000


class QuotaFetchError(RuntimeError):
    """Raised when a quota endpoint cannot be reached or parsed."""


@dataclass
class QuotaLimit:
    """One quota dimension: remaining budget plus the next reset timestamp."""

    remain: int = 0
    limit: int = 0
    next_reset: int = 0  # unix seconds; 0 = unknown / no reset
    label: str = ""  # e.g. "5h window", "weekly"

    @property
    def has_limit(self) -> bool:
        return self.limit > 0

    @property
    def used_ratio(self) -> float:
        return (self.limit - self.remain) / self.limit if self.limit else 0.0


@dataclass
class _QuotaLimitPayload(TypedDict, total=False):
    remain: int
    limit: int
    next_reset: int
    label: str


class _QuotaSnapshotPayload(TypedDict, total=False):
    provider: str
    model: str
    fetched_at: float
    status: str
    note: str
    requests: _QuotaLimitPayload
    tokens: _QuotaLimitPayload


@dataclass
class QuotaSnapshot:
    """Provider-agnostic quota state for the model in use.

    A model may be limited in requests, in tokens, or in both — whichever
    dimensions the provider reports are populated, the rest stay at
    ``has_limit == False`` and are ignored by the gate.
    """

    provider: str = ""
    model: str = ""
    fetched_at: float = 0.0
    requests: QuotaLimit = field(default_factory=lambda: QuotaLimit(label="requests"))
    tokens: QuotaLimit = field(default_factory=lambda: QuotaLimit(label="tokens"))
    status: str = "active"  # "active" | "suspended" | "unknown"
    note: str = ""

    @property
    def fresh(self) -> bool:
        return bool(self.fetched_at)

    def blocking(
        self,
        *,
        min_request_buffer: int = DEFAULT_MIN_REQUEST_BUFFER,
        min_token_buffer: int = DEFAULT_MIN_TOKEN_BUFFER,
    ) -> QuotaLimit | None:
        """Return the exhausted dimension, or None when there is room to run."""
        for dim, buffer in (
            (self.requests, min_request_buffer),
            (self.tokens, min_token_buffer),
        ):
            if dim.has_limit and dim.remain < buffer:
                return dim
        return None

    def exhausted(
        self,
        *,
        min_request_buffer: int = DEFAULT_MIN_REQUEST_BUFFER,
        min_token_buffer: int = DEFAULT_MIN_TOKEN_BUFFER,
    ) -> bool:
        return self.blocking(
            min_request_buffer=min_request_buffer,
            min_token_buffer=min_token_buffer,
        ) is not None

    def next_resume_ts(
        self,
        now: float | None = None,
        *,
        min_request_buffer: int = DEFAULT_MIN_REQUEST_BUFFER,
        min_token_buffer: int = DEFAULT_MIN_TOKEN_BUFFER,
    ) -> int | None:
        """Earliest future reset across exhausted dimensions.

        Handles the week-edge case: when both dimensions are tight (e.g. the
        weekly budget resets while the current 5h window is still exhausted),
        the resume lands on the first window reset at/after the weekly reset,
        because the window must be fresh again before we can run.
        """
        now = int(now or time.time())
        window_reset = self.requests.next_reset if (
            self.requests.has_limit and self.requests.remain < min_request_buffer
        ) else 0
        weekly_reset = self.tokens.next_reset if (
            self.tokens.has_limit and self.tokens.remain < min_token_buffer
        ) else 0
        if window_reset and weekly_reset:
            base = max(weekly_reset, now)
            if window_reset >= base:
                return window_reset if window_reset > now else None
            periods = (base - window_reset + WINDOW_RESET_PERIOD_S - 1) // WINDOW_RESET_PERIOD_S
            candidate = window_reset + periods * WINDOW_RESET_PERIOD_S
            return candidate if candidate > now else None
        if window_reset:
            if window_reset > now:
                return window_reset
            # Stale past value: roll forward to the next fixed window slot.
            periods = (now - window_reset) // WINDOW_RESET_PERIOD_S + 1
            return window_reset + periods * WINDOW_RESET_PERIOD_S
        if weekly_reset and weekly_reset > now:
            return weekly_reset
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "fetched_at": self.fetched_at,
            "status": self.status,
            "note": self.note,
            "requests": {
                "remain": self.requests.remain,
                "limit": self.requests.limit,
                "next_reset": self.requests.next_reset,
                "label": self.requests.label,
            },
            "tokens": {
                "remain": self.tokens.remain,
                "limit": self.tokens.limit,
                "next_reset": self.tokens.next_reset,
                "label": self.tokens.label,
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QuotaSnapshot":
        def _limit(raw: object, label: str) -> QuotaLimit:
            raw_map: dict[str, object] = (
                dict(cast(Mapping[str, object], raw).items()) if isinstance(raw, Mapping) else {}
            )
            return QuotaLimit(
                remain=_to_int(raw_map.get("remain", 0), 0),
                limit=_to_int(raw_map.get("limit", 0), 0),
                next_reset=_to_int(raw_map.get("next_reset", 0), 0),
                label=_to_str(raw_map.get("label", label), label),
            )

        payload: _QuotaSnapshotPayload = cast(_QuotaSnapshotPayload, data)
        return cls(
            provider=_to_str(payload.get("provider", ""), ""),
            model=_to_str(payload.get("model", ""), ""),
            fetched_at=_to_float(payload.get("fetched_at", 0.0), 0.0),
            requests=_limit(payload.get("requests"), "requests"),
            tokens=_limit(payload.get("tokens"), "tokens"),
            status=_to_str(payload.get("status", "active"), "active"),
            note=_to_str(payload.get("note", ""), ""),
        )


def _to_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _to_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _to_str(value: object, default: str = "") -> str:
    if value is None:
        return default
    string_value = str(value)
    return string_value if string_value else default


# ---------------------------------------------------------------------------
# provider fetchers
# ---------------------------------------------------------------------------

QuotaFetcher = Callable[[str, str], QuotaSnapshot]
"""Fetch live quota for (api_key, model). Raises QuotaFetchError on failure."""

_FETCHERS: dict[str, QuotaFetcher] = {}


def register_quota_fetcher(provider_name: str, fetcher: QuotaFetcher) -> None:
    """Register a quota fetcher for a normalized provider name."""
    _FETCHERS[provider_name.strip().lower()] = fetcher


def quota_fetcher_for(provider_name: str) -> QuotaFetcher | None:
    """Return the fetcher for a provider, or None when unsupported."""
    return _FETCHERS.get((provider_name or "").strip().lower())


def _http_get_json(url: str, headers: dict[str, str], timeout: float) -> dict[str, object]:
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            payload_obj: object = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise QuotaFetchError(f"quota endpoint HTTP {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise QuotaFetchError(f"quota endpoint unreachable: {exc.reason}") from exc
    except (TimeoutError, OSError) as exc:
        raise QuotaFetchError(f"quota endpoint error: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise QuotaFetchError(f"quota endpoint returned invalid JSON: {exc}") from exc
    if not isinstance(payload_obj, dict):
        raise QuotaFetchError("quota endpoint returned a non-object payload")
    return cast(dict[str, object], payload_obj)


def fetch_yuanyuai_quota(api_key: str, model: str, *, timeout: float = 8.0) -> QuotaSnapshot:
    """Fetch the YuanyuAI (Zhipu GLM) quota for one API key.

    Mirrors the ``query-quota`` skill contract: GET
    https://yuanyuaicloud.cn/api/query-quota with a Bearer key. The dashboard
    reports the 5h request window (rateLimit/windowRemain/nextWindowReset) and
    the weekly billed-call budget (weeklyLimit/weeklyRemain/nextWeeklyReset).
    """
    if not api_key or not api_key.strip():
        raise QuotaFetchError("missing YuanyuAI API key")
    payload = _http_get_json(
        "https://yuanyuaicloud.cn/api/query-quota",
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "nanobot-quota-gate/1.0",
        },
        timeout=timeout,
    )
    if not payload.get("success", True):
        raise QuotaFetchError(f"quota endpoint returned an error payload: {payload!r}")
    raw_data = payload.get("data")
    if not isinstance(raw_data, dict):
        raise QuotaFetchError("quota endpoint missing data object")
    data = cast(dict[str, object], raw_data)

    def _num(key: str, default: float = 0.0) -> float:
        value = data.get(key, default)
        return _to_float(value, default)

    return QuotaSnapshot(
        provider="custom_yuanyuai",
        model=model,
        fetched_at=time.time(),
        requests=QuotaLimit(
            remain=int(_num("windowRemain", 0)),
            limit=int(_num("rateLimit", 0)),
            next_reset=int(_num("nextWindowReset", 0)),
            label="5h window",
        ),
        tokens=QuotaLimit(
            remain=int(_num("weeklyRemain", 0)),
            limit=int(_num("weeklyLimit", 0)),
            next_reset=int(_num("nextWeeklyReset", 0)),
            label="weekly",
        ),
        status="active" if int(_num("status", 0)) == 1 else "suspended",
        note=str(data.get("remark", "") or data.get("name", "")),
    )


register_quota_fetcher("custom_yuanyuai", fetch_yuanyuai_quota)
register_quota_fetcher("custom-yuanyuai", fetch_yuanyuai_quota)


# ---------------------------------------------------------------------------
# fetch orchestration
# ---------------------------------------------------------------------------

def fetch_for_model(
    *,
    provider_name: str,
    model: str,
    api_key: str | None,
    timeout: float = 8.0,
) -> QuotaSnapshot | None:
    """Fetch quota for one provider/model, or None when unsupported/unavailable.

    Fail-open: any missing fetcher, key, or fetch error returns None so the
    caller runs the turn exactly as before.
    """
    fetcher = quota_fetcher_for(provider_name)
    if fetcher is None or not api_key:
        return None
    try:
        return fetcher(api_key, model)
    except QuotaFetchError as exc:
        logger.debug("quota fetch skipped for {}: {}", provider_name, exc)
        return None
    except Exception as exc:  # noqa: BLE001 - fail open on any fetcher bug
        logger.debug("quota fetch failed for {}: {!r}", provider_name, exc)
        return None


# ---------------------------------------------------------------------------
# session metadata helpers
# ---------------------------------------------------------------------------

def read_snapshot(metadata: Mapping[str, Any] | None) -> QuotaSnapshot | None:
    """Read the persisted quota snapshot, or None when absent/invalid."""
    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get(QUOTA_SNAPSHOT_META_KEY)
    if not isinstance(raw, Mapping):
        return None
    try:
        return QuotaSnapshot.from_dict(cast(Mapping[str, object], raw))
    except (TypeError, ValueError):
        return None


def write_snapshot(metadata: dict[str, Any], snapshot: QuotaSnapshot) -> None:
    metadata[QUOTA_SNAPSHOT_META_KEY] = snapshot.to_dict()


def read_suspension(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Read the suspension record, or None when the session is not suspended."""
    if not isinstance(metadata, Mapping):
        return None
    raw = metadata.get(QUOTA_SUSPENDED_META_KEY)
    if not isinstance(raw, Mapping):
        return None
    raw_map: dict[str, object] = dict(cast(Mapping[str, object], raw).items())
    return {str(key): value for key, value in raw_map.items()}


def write_suspension(
    metadata: dict[str, Any],
    *,
    reason: str,
    resume_at: int | None,
    provider: str,
    model: str,
    job_id: str | None = None,
) -> None:
    metadata[QUOTA_SUSPENDED_META_KEY] = {
        "reason": reason,
        "resume_at": resume_at,
        "provider": provider,
        "model": model,
        "job_id": job_id,
        "suspended_at": time.time(),
    }
    if job_id:
        metadata[QUOTA_RESUME_JOB_META_KEY] = job_id


def clear_suspension(metadata: dict[str, Any]) -> None:
    metadata.pop(QUOTA_SUSPENDED_META_KEY, None)
    metadata.pop(QUOTA_RESUME_JOB_META_KEY, None)


def resume_job_id(metadata: Mapping[str, Any] | None) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get(QUOTA_RESUME_JOB_META_KEY)
    return value if isinstance(value, str) and value else None


# ---------------------------------------------------------------------------
# presentation
# ---------------------------------------------------------------------------

def format_reset(ts: int) -> str:
    if not ts:
        return "unknown"
    return datetime.fromtimestamp(ts, CST).strftime("%Y-%m-%d %H:%M %Z")


QUOTA_STATUS_HEADER = "Model quota"

# The gate never writes quota into the model prompt: limits are internal
# decision input plus user-facing status, not instructions for the LLM.
# Keeping them out of context avoids burning tokens on every turn and avoids
# the model treating stale numbers as fact.


def status_lines(snapshot: QuotaSnapshot) -> list[str]:
    """User-facing one-line-per-dimension status for the WebUI/CLI surface."""
    lines: list[str] = []
    if snapshot.requests.has_limit:
        lines.append(
            f"{snapshot.requests.label}: {snapshot.requests.remain}/{snapshot.requests.limit} left "
            f"— resets {format_reset(snapshot.requests.next_reset)}"
        )
    if snapshot.tokens.has_limit:
        lines.append(
            f"{snapshot.tokens.label}: {snapshot.tokens.remain}/{snapshot.tokens.limit} left "
            f"— resets {format_reset(snapshot.tokens.next_reset)}"
        )
    if not lines:
        lines.append("No quota limits reported for this model")
    return lines


def status_payload(
    snapshot: QuotaSnapshot | None,
    suspension: Mapping[str, Any] | None,
    *,
    min_request_buffer: int = DEFAULT_MIN_REQUEST_BUFFER,
    min_token_buffer: int = DEFAULT_MIN_TOKEN_BUFFER,
) -> dict[str, Any]:
    """Structured quota status for display (no model-facing text)."""
    payload: dict[str, Any] = {
        "provider": snapshot.provider if snapshot else "",
        "model": snapshot.model if snapshot else "",
        "suspended": bool(suspension),
        "lines": status_lines(snapshot) if snapshot else [],
    }
    if snapshot is not None:
        payload["requests"] = {
            "remain": snapshot.requests.remain,
            "limit": snapshot.requests.limit,
            "next_reset": snapshot.requests.next_reset,
            "reset_display": format_reset(snapshot.requests.next_reset),
        }
        payload["tokens"] = {
            "remain": snapshot.tokens.remain,
            "limit": snapshot.tokens.limit,
            "next_reset": snapshot.tokens.next_reset,
            "reset_display": format_reset(snapshot.tokens.next_reset),
        }
        payload["exhausted"] = snapshot.exhausted(
            min_request_buffer=min_request_buffer,
            min_token_buffer=min_token_buffer,
        )
    if suspension:
        resume_at = suspension.get("resume_at") or 0
        payload["suspension"] = {
            "reason": suspension.get("reason", ""),
            "resume_at": resume_at,
            "resume_display": format_reset(int(resume_at)),
        }
    return payload


def suspension_notice(
    snapshot: QuotaSnapshot | None,
    reason: str,
    *,
    min_request_buffer: int = DEFAULT_MIN_REQUEST_BUFFER,
    min_token_buffer: int = DEFAULT_MIN_TOKEN_BUFFER,
) -> str:
    """User-visible notice explaining a quota suspension."""
    blocking = None
    if snapshot is not None:
        blocking = snapshot.blocking(
            min_request_buffer=min_request_buffer,
            min_token_buffer=min_token_buffer,
        )
    if snapshot is not None and blocking is not None:
        reset = format_reset(blocking.next_reset)
        model_name = snapshot.model or "the current model"
        detail = (
            f"The {blocking.label} budget for {model_name} is exhausted "
            f"({blocking.remain}/{blocking.limit} left)."
        )
    elif snapshot is not None and snapshot.tokens.next_reset:
        reset = format_reset(snapshot.tokens.next_reset)
        model_name = snapshot.model or "the current model"
        detail = f"The provider rejected the request for {model_name}: {reason}."
    else:
        reset = "unknown"
        detail = f"The provider rejected the request: {reason}."
    return (
        f"{detail} This session is paused and will resume automatically after "
        f"the next reset ({reset})."
    )


def fit_iterations(
    requested: int,
    snapshot: QuotaSnapshot,
    *,
    min_request_buffer: int = DEFAULT_MIN_REQUEST_BUFFER,
) -> int:
    """Shrink a turn's iteration budget to fit the remaining request quota.

    Token budgets cannot be mapped to iterations without a usage model, so
    only the request dimension shrinks the budget. Never returns less than 1:
    a turn with zero room is a suspension decision, not a zero-iteration run.
    """
    if not snapshot.requests.has_limit:
        return requested
    room = max(0, snapshot.requests.remain - min_request_buffer)
    if room <= 0:
        return 1
    return max(1, min(requested, room))


# ---------------------------------------------------------------------------
# resume scheduling
# ---------------------------------------------------------------------------

def resume_message(provider: str, model: str, reason: str) -> str:
    """Prompt for the one-shot cron turn that resumes a suspended session."""
    return (
        f"Quota resume check for model {model} (provider {provider}). "
        f"The session was suspended ({reason}) because the model quota ran out. "
        "First re-check the current quota: if there is room for at least one "
        "model call, continue the suspended work from the saved session context "
        "without asking the user. If the quota is still exhausted, stay suspended "
        "and schedule the next check at the newly reported reset time."
    )


def build_resume_schedule(resume_at: int | None) -> CronSchedule:
    """Return a cron schedule for a one-shot resume at ``resume_at``."""
    if resume_at:
        return CronSchedule(kind="at", at_ms=int(resume_at) * 1000)
    return CronSchedule(kind="every", every_ms=5 * 60_000)
