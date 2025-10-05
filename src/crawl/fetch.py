"""HTTP and headless fetching utilities for the logo similarity pipeline."""

from __future__ import annotations

import atexit
import logging
from threading import Lock
from urllib.parse import urljoin, urlparse

import requests
from requests import Session
from tenacity import (  # type: ignore[import-untyped]
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

try:
    from playwright.sync_api import (  # type: ignore[import-untyped]
        TimeoutError as PlaywrightTimeoutError,
        sync_playwright,
    )
except ImportError:  # pragma: no cover - graceful degradation when playwright missing
    sync_playwright = None  # type: ignore[assignment]
    PlaywrightTimeoutError = Exception  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10.0
_PLAYWRIGHT_TIMEOUT_MS = int(_DEFAULT_TIMEOUT * 1000)
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

_session_lock = Lock()
_session: Session | None = None


class RetryableHTTPStatusError(Exception):
    """Raised for HTTP status codes that should trigger a retry."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"Server returned status {status_code}")
        self.status_code = status_code


def _get_session() -> Session:
    """Return a shared requests session configured with default headers."""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                session = requests.Session()
                session.headers.update(
                    {
                        "User-Agent": _USER_AGENT,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                    }
                )
                _session = session
    return _session


_retryer = Retrying(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type(
        (requests.Timeout, requests.ConnectionError, RetryableHTTPStatusError)
    ),
    before_sleep=before_sleep_log(logger, logging.DEBUG),
    reraise=True,
)


def _ensure_http_scheme(url: str) -> str:
    """Ensure *url* is qualified with an HTTP scheme, defaulting to https."""
    cleaned = url.strip()
    if not cleaned:
        return cleaned
    if cleaned.startswith(("http://", "https://")):
        return cleaned
    if cleaned.startswith("//"):
        return f"https:{cleaned}"
    parsed = urlparse(cleaned)
    if parsed.scheme:
        return cleaned
    return f"https://{cleaned}"


_playwright_lock = Lock()
_playwright = None
_browser = None
_browser_context = None


def _shutdown_playwright() -> None:
    """Close Playwright resources on interpreter shutdown."""
    global _playwright, _browser, _browser_context
    with _playwright_lock:
        if _browser_context is not None:
            try:
                _browser_context.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.debug("Failed to close browser context", exc_info=True)
            _browser_context = None
        if _browser is not None:
            try:
                _browser.close()
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.debug("Failed to close browser", exc_info=True)
            _browser = None
        if _playwright is not None:
            try:
                _playwright.stop()
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.debug("Failed to stop Playwright", exc_info=True)
            _playwright = None


def _ensure_browser_context():
    """Start Playwright and return a shared browser context."""
    global _playwright, _browser, _browser_context
    if sync_playwright is None:
        logger.error("Playwright is not installed; cannot render pages.")
        return None
    with _playwright_lock:
        if _browser_context is not None:
            return _browser_context
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        context.set_default_navigation_timeout(_PLAYWRIGHT_TIMEOUT_MS)
        context.set_default_timeout(_PLAYWRIGHT_TIMEOUT_MS)
        _playwright = playwright
        _browser = browser
        _browser_context = context
        atexit.register(_shutdown_playwright)
        return _browser_context


def _fetch_once(url: str, timeout: float) -> tuple[str, str]:
    """Issue a single HTTP GET request and return the resolved URL and HTML."""
    target_url = _ensure_http_scheme(url)
    session = _get_session()
    response = session.get(target_url, timeout=timeout, allow_redirects=True)
    if 500 <= response.status_code < 600:
        raise RetryableHTTPStatusError(response.status_code)
    if not response.encoding:
        response.encoding = response.apparent_encoding or "utf-8"
    html = response.text
    return response.url, html


def fetch_html(url: str) -> tuple[str | None, str | None]:
    """Fetch *url* via HTTP, returning the final URL and HTML content.

    Retries are attempted for transient failures such as server errors or timeouts.
    On error the function returns ``(None, None)`` and logs the failure.
    """
    try:
        final_url, html = _retryer(lambda: _fetch_once(url, _DEFAULT_TIMEOUT))
        return final_url, html
    except RetryableHTTPStatusError as exc:
        logger.warning("Server error fetching %s: %s", url, exc)
    except requests.RequestException as exc:
        logger.warning("Request error fetching %s: %s", url, exc)
    except Exception:  # noqa: BLE001 - avoid leaking unexpected exceptions
        logger.exception("Unexpected error fetching %s", url)
    return None, None


def render_page(url: str, wait_selector: str | None = None) -> tuple[str | None, str | None]:
    """Render *url* in a headless browser and return the final URL and HTML.

    If *wait_selector* is provided, the renderer waits for the selector to appear
    before capturing the page content. Returns ``(None, None)`` when rendering
    fails; errors are logged but not raised.
    """
    context = _ensure_browser_context()
    if context is None:
        return None, None
    page = context.new_page()
    try:
        target_url = _ensure_http_scheme(url)
        page.goto(target_url, wait_until="domcontentloaded", timeout=_PLAYWRIGHT_TIMEOUT_MS)
        if wait_selector:
            page.wait_for_selector(wait_selector, timeout=_PLAYWRIGHT_TIMEOUT_MS)
        final_url = page.url
        html = page.content()
        return final_url, html
    except PlaywrightTimeoutError as exc:  # type: ignore[call-arg]
        logger.warning("Render timeout for %s: %s", url, exc)
    except Exception:  # noqa: BLE001 - avoid leaking Playwright exceptions
        logger.exception("Render failed for %s", url)
    finally:
        try:
            page.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.debug("Failed to close page for %s", url, exc_info=True)
    return None, None


def normalize_url(url: str, base: str) -> str:
    """Return an absolute URL by resolving *url* against *base*."""
    return urljoin(base, url)
