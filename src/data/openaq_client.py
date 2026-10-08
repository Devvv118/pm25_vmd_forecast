"""Thin client for the OpenAQ v3 API (https://docs.openaq.org/api).

Endpoints used:
  - GET /locations                      (find stations near a point)
  - GET /locations/{id}/sensors         (fallback: sensors at a station)
  - GET /sensors/{id}/hours             (hourly-aggregated measurements)

Requires a free API key: https://explore.openaq.org/register. Put it in a
`.env` file in the project root (OPENAQ_API_KEY=...); it is loaded
automatically. An already-set environment variable or api_key= also works.

Robustness features (vs. the first version):
  * retries on 429 *and* 5xx / network errors, honouring Retry-After and the
    x-ratelimit-* headers instead of guessing a sleep time
  * proactive throttling: pauses when the rate-limit window is exhausted
    rather than burning retries on guaranteed 429s
  * every list endpoint is paginated (the old /locations call silently
    truncated at one page)
  * clear errors for 401/403 (bad key) and other 4xx (bad parameters), with
    the API's own message included
"""
import os
import time

import requests
from dotenv import load_dotenv

# Project root = two levels above src/data/. Loading from there (and from the
# current directory) means `.env` is found no matter where you run from.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(_ROOT, ".env"))
load_dotenv()

BASE_URL = "https://api.openaq.org/v3"
PM25_PARAMETER_ID = 2  # OpenAQ parameters list: id=2 -> "pm25"
MAX_RADIUS_M = 25000   # OpenAQ's per-request radius cap


class OpenAQError(RuntimeError):
    pass


class OpenAQClient:
    def __init__(self, api_key: str = None, timeout: int = 30,
                 max_retries: int = 6, backoff_seconds: float = 2.0,
                 verbose: bool = False):
        self.api_key = api_key or os.environ.get("OPENAQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No OpenAQ API key found. Add OPENAQ_API_KEY=your_key to a .env "
                "file in the project root (copy .env.example). "
                "Get a free key at https://explore.openaq.org/register"
            )
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": self.api_key})
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    # low-level
    # ------------------------------------------------------------------ #
    @staticmethod
    def _header_float(resp, name):
        try:
            return float(resp.headers.get(name))
        except (TypeError, ValueError):
            return None

    def _sleep(self, seconds: float, why: str):
        seconds = max(0.0, min(seconds, 120.0))
        if self.verbose or seconds >= 5:
            print(f"    [openaq] {why}; sleeping {seconds:.0f}s")
        time.sleep(seconds)

    def _get(self, path: str, params: dict = None) -> dict:
        url = f"{BASE_URL}{path}"
        last_problem = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                last_problem = f"network error: {exc}"
                self._sleep(self.backoff_seconds * attempt, last_problem)
                continue

            if resp.status_code == 429:
                wait = (self._header_float(resp, "Retry-After")
                        or self._header_float(resp, "x-ratelimit-reset")
                        or self.backoff_seconds * attempt)
                last_problem = "HTTP 429 (rate limited)"
                self._sleep(wait + 1, last_problem)
                continue

            if resp.status_code >= 500:
                last_problem = f"HTTP {resp.status_code} (server error)"
                self._sleep(self.backoff_seconds * attempt, last_problem)
                continue

            if resp.status_code in (401, 403):
                raise OpenAQError(
                    f"HTTP {resp.status_code} from OpenAQ - API key missing/invalid. "
                    f"Check OPENAQ_API_KEY in your .env file. ({resp.text[:200]})")

            if resp.status_code >= 400:
                raise OpenAQError(
                    f"HTTP {resp.status_code} for {resp.url}: {resp.text[:300]}")

            # Success: if we've used up the rate-limit window, wait it out now.
            remaining = self._header_float(resp, "x-ratelimit-remaining")
            reset = self._header_float(resp, "x-ratelimit-reset")
            if remaining is not None and remaining <= 1 and reset:
                self._sleep(reset + 1, "rate-limit window exhausted")
            return resp.json()

        raise OpenAQError(
            f"Failed to GET {url} after {self.max_retries} attempts "
            f"(last problem: {last_problem})")

    def _paginate(self, path: str, params: dict, page_limit: int = 1000,
                  max_pages: int = 200) -> list:
        """Collect every page of a list endpoint."""
        results = []
        for page in range(1, max_pages + 1):
            data = self._get(path, {**params, "limit": page_limit, "page": page})
            batch = data.get("results", [])
            results.extend(batch)
            if len(batch) < page_limit:
                break
        return results

    # ------------------------------------------------------------------ #
    # endpoints
    # ------------------------------------------------------------------ #
    def find_locations(self, lat: float, lon: float, radius_m: int = MAX_RADIUS_M,
                       parameters_id: int = PM25_PARAMETER_ID) -> list:
        """Stations within radius_m of (lat, lon) that measure the parameter.

        Each result already includes a `sensors` list (id + parameter), plus
        `datetimeFirst` / `datetimeLast`, so no extra per-station call is
        normally needed.
        """
        params = {
            "coordinates": f"{lat},{lon}",
            "radius": min(radius_m, MAX_RADIUS_M),
            "parameters_id": parameters_id,
        }
        return self._paginate("/locations", params, page_limit=100)

    def get_location_sensors(self, location_id: int) -> list:
        return self._get(f"/locations/{location_id}/sensors").get("results", [])

    def get_sensor_hours(self, sensor_id: int, datetime_from: str, datetime_to: str,
                         page_limit: int = 1000) -> list:
        """Hourly-aggregated measurements for one sensor, all pages.

        datetime_from / datetime_to are ISO-8601 strings; include a UTC offset
        (e.g. "2026-09-01T00:00:00+05:30") so the window is unambiguous.
        """
        params = {"datetime_from": datetime_from, "datetime_to": datetime_to}
        return self._paginate(f"/sensors/{sensor_id}/hours", params, page_limit=page_limit)
