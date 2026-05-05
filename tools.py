"""Mock tools for DealScout agents.

Every tool is a regular Python function with a clear input/output contract.
In v1 these return realistic fake data sourced from fixture files. At M7,
one tool gets swapped for a real API behind the same interface — the agent
code does not change.

Two pieces of metadata travel with the functions:
  - TOOL_SCHEMAS    : OpenAI-format JSON schemas (what the LLM sees)
  - TOOL_FUNCTIONS  : name -> Python function (how the agent loop dispatches)
"""

import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Mock URLs map to fixture files. Real fetch would hit the network instead.
URL_TO_FIXTURE = {
    "mock://swift_clean": "clean_listing.txt",
    "mock://honda_sparse": "sparse_listing.txt",
    "mock://i20_noisy": "noisy_listing.txt",
    "mock://innova_dealer": "dealer_listing.txt",
}


_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _extract_title(html: str) -> str | None:
    m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None


def _extract_meta(html: str, key: str) -> str | None:
    """Pull content from <meta property='key' ...> or <meta name='key' ...>."""
    esc = re.escape(key)
    patterns = [
        rf'<meta[^>]+(?:property|name)=["\']{esc}["\'][^>]*content=["\']([^"\']+)["\']',
        rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]*(?:property|name)=["\']{esc}["\']',
    ]
    for p in patterns:
        m = re.search(p, html, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def _fetch_real_url(url: str) -> dict:
    """Fetch an http(s) URL and pull listing text from HTML <title> + meta tags.

    Uses Playwright + Firefox, not raw httpx — major listing sites (OLX, etc.)
    do TLS/HTTP2 fingerprinting that rejects non-browser clients at the
    transport layer. A real browser engine renders JS too, so SPAs work as
    well. Cost: ~3-5s per fetch (Firefox cold start).

    Most listing sites populate og:title/og:description with the
    make/model/year/location/km — more than enough for the Listing Analyst
    to extract a FactSheet.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"status": "error", "reason": "playwright not installed"}

    title = description = None
    try:
        with sync_playwright() as p:
            browser = p.firefox.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, timeout=30000, wait_until="domcontentloaded")

                def _meta(selector: str) -> str | None:
                    loc = page.locator(selector).first
                    if loc.count():
                        return loc.get_attribute("content")
                    return None

                title = _meta('meta[property="og:title"]') or page.title()
                description = (
                    _meta('meta[property="og:description"]')
                    or _meta('meta[name="description"]')
                )
                # Schema.org JSON-LD blocks carry structured fields (price,
                # mileage, year) that aren't in og:* meta tags. Most major
                # listing sites embed at least one Product/Car block.
                ld_blocks = []
                ld_loc = page.locator('script[type="application/ld+json"]')
                for i in range(ld_loc.count()):
                    txt = ld_loc.nth(i).text_content()
                    if txt and ("price" in txt.lower() or '"product"' in txt.lower() or '"car"' in txt.lower()):
                        ld_blocks.append(txt.strip())
            finally:
                browser.close()
    except Exception as e:
        return {"status": "error", "reason": f"Browser fetch failed: {e}"}

    pieces = []
    if title:
        pieces.append(f"Title: {title}")
    if description:
        pieces.append(f"Description: {description}")
    if ld_blocks:
        pieces.append("Structured data (JSON-LD):\n" + "\n".join(ld_blocks))

    if not pieces:
        return {"status": "error", "reason": "No title/description in HTML"}

    return {"status": "ok", "text": "\n\n".join(pieces)}


def fetch_listing(url: str) -> dict:
    """Fetch a listing from a URL and return cleaned text + structured hints.

    Returns one of:
      {
        "status": "ok",
        "text":             <cleaned listing text>,
        "structured_hints": {<key>: <value>, ...},
        "fetched_at":       <iso8601>,
      }
      {"status": "error", "reason": <str>, "fetched_at": <iso8601>}

    Dispatch:
      mock://...     → fixture file (deterministic test data)
      http(s)://...  → real HTTP fetch + HTML metadata extraction

    Note: parse_listing is called internally so the LLM only sees one tool.
    Exposing both fetch and parse as separate tools tempted GPT-4o-mini into
    chunking the listing and calling parse_listing repeatedly. One tool, one
    call, no loop.
    """
    # Real fetches take 200ms-1s. Match the order of magnitude so parallelism
    # at M3 actually feels like a win.
    time.sleep(random.uniform(0.2, 0.6))

    fetched_at = datetime.now(timezone.utc).isoformat()

    if url in URL_TO_FIXTURE:
        fixture_path = FIXTURES_DIR / URL_TO_FIXTURE[url]
        raw_text = fixture_path.read_text()
        parsed = parse_listing(raw_text)
        return {
            "status": "ok",
            "text": parsed["text"],
            "structured_hints": parsed["structured_hints"],
            "fetched_at": fetched_at,
        }

    if url.startswith(("http://", "https://")):
        result = _fetch_real_url(url)
        if result["status"] == "error":
            result["fetched_at"] = fetched_at
            return result
        parsed = parse_listing(result["text"])
        return {
            "status": "ok",
            "text": parsed["text"],
            "structured_hints": parsed["structured_hints"],
            "fetched_at": fetched_at,
        }

    return {
        "status": "error",
        "reason": f"URL not found: {url}",
        "fetched_at": fetched_at,
    }


def parse_listing(text: str) -> dict:
    """Clean raw listing text and extract obvious 'Label: value' hints.

    The LLM does the heavy extraction. This tool does the easy mechanical
    bits — strip HTML if present, normalise whitespace, surface any
    structured key/value pairs as hints.

    Returns:
      {"text": <cleaned text>, "structured_hints": {<key>: <value>, ...}}
    """
    time.sleep(random.uniform(0.05, 0.15))

    cleaned = re.sub(r"<[^>]+>", "", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    hints: dict[str, str] = {}
    for line in cleaned.splitlines():
        match = re.match(r"^([A-Za-z][A-Za-z /.\-]{1,30})\s*[:=]\s*(.+)$", line.strip())
        if match:
            key = match.group(1).strip().lower().replace(" ", "_").replace(".", "")
            value = match.group(2).strip()
            if key and value and len(value) < 200:
                hints[key] = value

    return {"text": cleaned, "structured_hints": hints}


# =============================================================================
# Visual Inspector tool — mock photo analysis (M3+).
# =============================================================================
# Maps photo URL prefixes to a "scenario" key, then a scenario to predetermined
# findings. At M7 this would be swapped for real gpt-4o-mini vision calls.

PHOTO_URL_TO_SCENARIO = {
    "https://example.com/listings/swift_clean/": "swift_clean",
    "https://example.com/i20_pics/": "i20_noisy",
    "https://premiumcars.example.com/listings/inv-2240/": "innova_dealer",
}

PHOTO_SCENARIO_FINDINGS = {
    "swift_clean": {
        "concerns": [],
        "missing_angles": ["engine bay", "odometer close-up"],
        "overall_condition_hint": "good",
        "notes": "Consistent paint across panels. Clean exterior, no visible damage. "
                 "Interior shots not provided.",
    },
    "i20_noisy": {
        "concerns": [
            {
                "type": "scratches_or_dents",
                "severity": "low",
                "confidence": "medium",
                "location": "rear bumper",
                "description": "Two visible scratches on rear bumper, "
                               "consistent with the parking incident the seller mentions.",
            }
        ],
        "missing_angles": ["engine bay", "odometer close-up", "interior dashboard", "front view"],
        "overall_condition_hint": "fair",
        "notes": "Only 2 photos provided. Cannot assess interior or engine condition.",
    },
    "innova_dealer": {
        "concerns": [
            {
                "type": "interior_wear",
                "severity": "medium",
                "confidence": "high",
                "location": "driver seat and steering wheel",
                "description": "Driver seat bolster and steering wheel show wear "
                               "typical of fleet-vehicle high-mileage use.",
            }
        ],
        "missing_angles": ["under-bonnet", "odometer close-up"],
        "overall_condition_hint": "good",
        "notes": "Five photos, well-detailed exterior + cabin. "
                 "Fleet usage visible in driver-side wear.",
    },
}


def analyze_photos(photo_urls: list[str]) -> dict:
    """Mock photo analysis. Returns concerns + missing angles + condition hint.

    For honda_sparse (zero photos), surfaces an absence-of-data finding.
    For unknown URL patterns, returns a low-confidence empty result.
    """
    time.sleep(random.uniform(0.3, 0.8))  # vision-API-ish latency

    if not photo_urls:
        return {
            "scenario": "no_photos",
            "concerns": [],
            "missing_angles": [
                "front", "rear", "side profile", "interior", "odometer", "engine bay"
            ],
            "overall_condition_hint": "fair",
            "photo_count": 0,
            "notes": "No photos provided. Cannot inspect the vehicle visually.",
        }

    for url in photo_urls:
        for prefix, scenario in PHOTO_URL_TO_SCENARIO.items():
            if url.startswith(prefix):
                findings = dict(PHOTO_SCENARIO_FINDINGS[scenario])
                findings["scenario"] = scenario
                findings["photo_count"] = len(photo_urls)
                return findings

    return {
        "scenario": "unknown",
        "concerns": [],
        "missing_angles": [],
        "overall_condition_hint": "fair",
        "photo_count": len(photo_urls),
        "notes": f"Could not match {len(photo_urls)} photo URL(s) to a known scenario.",
    }


# =============================================================================
# Price Auditor tool — mock market price lookup (M3+).
# =============================================================================
# Hand-picked market bands keyed to our fixtures. Real implementation at M7
# would query a market-price API or scrape comparable listings.


def market_price_lookup(
    make: str, model: str, year: int, km_driven: int, city: str
) -> dict:
    """Mock market price lookup. Returns p25/p50/p75 + comparable count + notes."""
    time.sleep(random.uniform(0.2, 0.5))

    m = (model or "").lower()
    y = year or 0

    if "swift" in m and y == 2018:
        return {
            "p25": 420000,
            "p50": 460000,
            "p75": 510000,
            "comparable_count": 47,
            "notes": "Strong used-market presence. Pricing fairly stable for VXi trim.",
        }
    if "city" in m and y == 2015:
        return {
            "p25": 400000,
            "p50": 470000,
            "p75": 560000,
            "comparable_count": 32,
            "notes": "Older Honda Citys vary widely on condition. Verify service history.",
        }
    if "i20" in m and y == 2019:
        return {
            "p25": 580000,
            "p50": 640000,
            "p75": 700000,
            "comparable_count": 41,
            "notes": "Hyundai i20 Sportz is mid-trim. Holds value well in metros.",
        }
    if "innova" in m and y == 2020:
        return {
            "p25": 1500000,
            "p50": 1700000,
            "p75": 1900000,
            "comparable_count": 23,
            "notes": "Innova Crysta retains value. Fleet variants typically 5-10% lower.",
        }

    return {
        "p25": 0,
        "p50": 0,
        "p75": 0,
        "comparable_count": 0,
        "notes": (
            f"No comparable listings indexed for {make} {model} {year} "
            f"in {city}. Pricing assessment unavailable."
        ),
    }


# =============================================================================
# History Checker tools — mock RC / insurance / challan / IIB-hint (M4+).
# =============================================================================
# Mock data keyed by registration number. Engineered disagreements baked in
# for swift_clean (TN10AB1234) and innova_dealer (TN22XY9988) per the M4 plan.

RC_DATA = {
    "TN10AB1234": {  # swift_clean — engineered disagreement vs. seller + Visual
        "owner_count": 3,            # seller's listing said "1 owner" — RC contradicts
        "registration_year": 2018,
        "rto": "TN10",
        "fitness_valid": True,
        "hypothecation": False,
        "blacklist": False,
        "notes": "Three ownership transfers in the last 18 months. Atypical "
                 "pattern often associated with problem vehicles.",
    },
    "KA03MN5678": {  # i20_noisy — confirms seller's claims (no disagreement)
        "owner_count": 1,
        "registration_year": 2019,
        "rto": "KA03",
        "fitness_valid": True,
        "hypothecation": False,
        "blacklist": False,
        "notes": "",
    },
    "TN22XY9988": {  # innova_dealer — engineered disagreement
        "owner_count": 3,            # seller said 2 owners
        "registration_year": 2020,
        "rto": "TN22",
        "fitness_valid": True,
        "hypothecation": True,       # active lien from financier
        "blacklist": False,
        "notes": "Active hypothecation by SBI Auto Loans. Most recent transfer "
                 "approximately 8 weeks ago.",
    },
}

INSURANCE_DATA = {
    "TN10AB1234": {
        "insurer": "ICICI Lombard",
        "valid_until": "August 2026",
        "lapsed": False,
        "lapse_duration_days": None,
        "notes": "",
    },
    "KA03MN5678": {
        "insurer": "HDFC Ergo",
        "valid_until": "April 2026",
        "lapsed": True,
        "lapse_duration_days": 35,
        "notes": "Lapse confirms seller's claim ('insurance lapsed since last month').",
    },
    "TN22XY9988": {
        "insurer": "Bajaj Allianz",
        "valid_until": "March 2026",
        "lapsed": True,
        "lapse_duration_days": 60,
        "notes": "Lapsed 60 days ago. Not disclosed in seller's listing.",
    },
}

CHALLAN_DATA = {
    "TN10AB1234": [],
    "KA03MN5678": [],
    "TN22XY9988": [
        {
            "challan_id": "TN22-CH-2025-1138",
            "offence": "Overspeeding (60 km/h in 30 km/h zone)",
            "issued_to_name": "RAVI KUMAR S",
            "fine_amount": 1500,
            "paid": False,
            "issue_date": "2025-12-14",
        },
        {
            "challan_id": "TN22-CH-2025-1457",
            "offence": "Signal jumping",
            "issued_to_name": None,
            "fine_amount": 500,
            "paid": False,
            "issue_date": "2026-01-22",
        },
    ],
}


def _rc_lookup_mock(reg_no: str) -> dict:
    """In-memory dict lookup — the M4 mock kept for offline runs."""
    time.sleep(random.uniform(0.8, 1.5))

    if not reg_no:
        return {"status": "error", "reason": "reg_no is empty or missing"}

    if reg_no not in RC_DATA:
        return {
            "status": "error",
            "reason": f"No RC record indexed for {reg_no}. State coverage may be incomplete.",
        }

    return {"status": "ok", "reg_no": reg_no, **RC_DATA[reg_no]}


def _rc_lookup_real(
    api_url: str, reg_no: str, *, max_retries: int = 2
) -> dict:
    """Real HTTP call to the RC lookup service with proper error handling.

    Catches the realistic failure modes and converts each to the standard
    `{"status": "error", "reason": ...}` shape so the agent layer doesn't
    have to know real HTTP is involved.

    Retries on transient failures (timeout, 5xx) with exponential backoff.
    Does NOT retry on auth failures (401), missing data (404), or
    connection refused (server down — fail fast).
    """
    if not reg_no:
        return {"status": "error", "reason": "reg_no is empty or missing"}

    api_key = os.getenv("RC_API_KEY", "dev-secret-key")
    url = f"{api_url.rstrip('/')}/rc-lookup/{reg_no}"
    headers = {"X-API-Key": api_key}

    last_error: str = "unknown"

    for attempt in range(max_retries + 1):
        try:
            response = httpx.get(url, headers=headers, timeout=5.0)

            # Non-retryable client errors.
            if response.status_code == 401:
                return {
                    "status": "error",
                    "reason": "Authentication failed (HTTP 401). Check RC_API_KEY.",
                }
            if response.status_code == 404:
                return {
                    "status": "error",
                    "reason": f"No RC record indexed for {reg_no} (HTTP 404).",
                }

            # Retryable server errors.
            if 500 <= response.status_code < 600:
                last_error = f"HTTP {response.status_code} from server"
                if attempt < max_retries:
                    time.sleep(0.5 * (2 ** attempt))  # 0.5s, 1s, 2s...
                    continue
                return {
                    "status": "error",
                    "reason": f"Server error after {max_retries + 1} attempts: {last_error}",
                }

            # Successful HTTP — parse and return.
            response.raise_for_status()
            try:
                return response.json()
            except (ValueError, KeyError) as exc:
                return {
                    "status": "error",
                    "reason": f"Malformed response: {exc}",
                }

        except httpx.TimeoutException:
            last_error = "request timed out (5s)"
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            return {
                "status": "error",
                "reason": f"Timeout after {max_retries + 1} attempts.",
            }

        except httpx.ConnectError:
            return {
                "status": "error",
                "reason": (
                    f"Cannot connect to {api_url}. Is the mock server running? "
                    f"Start it with: uvicorn mock_server:app --port 8080"
                ),
            }

        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(0.5 * (2 ** attempt))
                continue
            return {
                "status": "error",
                "reason": f"HTTP error after {max_retries + 1} attempts: {last_error}",
            }

    return {"status": "error", "reason": f"Exhausted retries: {last_error}"}


def rc_lookup(reg_no: str) -> dict:
    """RC lookup — dispatches to real HTTP if RC_API_URL is set, otherwise mock.

    Backward compatible: M1-M6 runs without RC_API_URL get the same in-memory
    behaviour as before. M7 sets RC_API_URL=http://127.0.0.1:8080 to use the
    mock server (real HTTP, real failures, same data).
    """
    api_url = os.getenv("RC_API_URL")
    if api_url:
        return _rc_lookup_real(api_url, reg_no)
    return _rc_lookup_mock(reg_no)


def insurance_status(reg_no: str) -> dict:
    """Mock insurance lookup. Returns insurer, validity, lapse status."""
    time.sleep(random.uniform(0.5, 1.0))

    if not reg_no:
        return {"status": "error", "reason": "reg_no is empty or missing"}

    if reg_no not in INSURANCE_DATA:
        return {
            "status": "error",
            "reason": f"No insurance record indexed for {reg_no}.",
        }

    return {"status": "ok", "reg_no": reg_no, **INSURANCE_DATA[reg_no]}


def challan_check(reg_no: str) -> dict:
    """Mock challan lookup. Returns list of challans (pending and disposed)."""
    time.sleep(random.uniform(0.4, 0.8))

    if not reg_no:
        return {"status": "error", "reason": "reg_no is empty or missing"}

    if reg_no not in CHALLAN_DATA:
        return {
            "status": "error",
            "reason": f"No challan data indexed for {reg_no}. State coverage may be incomplete.",
        }

    return {
        "status": "ok",
        "reg_no": reg_no,
        "challans": CHALLAN_DATA[reg_no],
        "total_unpaid": sum(
            1 for c in CHALLAN_DATA[reg_no] if not c["paid"]
        ),
    }


def accident_lookup_hint(reg_no: str) -> dict:
    """Returns IIB lookup guidance — NOT data.

    Per Appendix A: the only authoritative source for accident history in
    India is IIB V-Seva, which has no public API. This tool intentionally
    returns instructions rather than fabricated data.
    """
    time.sleep(random.uniform(0.05, 0.1))

    return {
        "status": "guidance",
        "instruction": (
            f"Run IIB V-Seva manually for {reg_no or '(missing reg_no)'}. The "
            "Insurance Information Bureau is the only authoritative source for "
            "accident history in India and has no public API. The data also "
            "has a ~2-month lag, so very recent claims may not yet be visible."
        ),
        "url": "https://nonlife.iib.gov.in/iib/",
        "data_lag_months": 2,
        "automation_status": "not_available",
    }


# OpenAI-format tool schemas for the History Checker's internal ReAct loop.
HISTORY_CHECKER_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "rc_lookup",
            "description": (
                "Look up registration certificate details for a vehicle. Returns "
                "owner_count, registration_year, RTO, fitness_valid, hypothecation, "
                "blacklist, notes. Call this FIRST — its results inform whether "
                "the other lookups are worthwhile."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "reg_no": {
                        "type": "string",
                        "description": "Vehicle registration number, e.g. TN10AB1234.",
                    }
                },
                "required": ["reg_no"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "insurance_status",
            "description": (
                "Look up the current insurance status. Returns insurer, "
                "valid_until, lapsed flag, lapse_duration_days, notes."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "reg_no": {"type": "string"},
                },
                "required": ["reg_no"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "challan_check",
            "description": (
                "Look up traffic challans (fines) issued against the vehicle. "
                "Returns a list of challans with offence, fine_amount, paid status, "
                "issued_to_name (mismatch vs. RC owner is a red flag), issue_date."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "reg_no": {"type": "string"},
                },
                "required": ["reg_no"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "accident_lookup_hint",
            "description": (
                "Returns IIB V-Seva lookup GUIDANCE — not data. Accident history "
                "in India is only available via IIB's web form, which has no "
                "public API. Call this tool to get the correct manual-lookup "
                "instructions to surface in your report. ALWAYS call this exactly "
                "once per investigation."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "reg_no": {"type": "string"},
                },
                "required": ["reg_no"],
                "additionalProperties": False,
            },
        },
    },
]

HISTORY_CHECKER_TOOL_FUNCTIONS = {
    "rc_lookup": rc_lookup,
    "insurance_status": insurance_status,
    "challan_check": challan_check,
    "accident_lookup_hint": accident_lookup_hint,
}


# =============================================================================
# OpenAI-format tool schemas — what the LLM sees.
# Only fetch_listing is exposed; it calls parse_listing internally.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_listing",
            "description": (
                "Fetch a used-car listing from a URL. Returns cleaned text and "
                "any obvious 'Label: value' structured hints. Call this exactly "
                "once per listing, then produce the FactSheet."
            ),
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The listing URL (e.g. mock://swift_clean during v1 testing).",
                    }
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
]

# Dispatch table — the agent loop looks up tool calls by name.
TOOL_FUNCTIONS = {
    "fetch_listing": fetch_listing,
}


if __name__ == "__main__":
    # Demo: fetch and parse each known fixture, then try an unknown URL.
    print(f"Available mock URLs: {list(URL_TO_FIXTURE.keys())}\n")

    for url in URL_TO_FIXTURE:
        print("=" * 60)
        print(f"URL: {url}")
        print("=" * 60)

        start = time.time()
        result = fetch_listing(url)
        elapsed = time.time() - start
        print(f"fetch_listing -> status={result['status']}, took {elapsed:.2f}s")

        if result["status"] == "ok":
            parsed = parse_listing(result["text"])
            print(f"parse_listing -> {len(parsed['text'])} chars cleaned, "
                  f"{len(parsed['structured_hints'])} hints found")
            print(f"  hint keys: {list(parsed['structured_hints'].keys())[:8]}")
        print()

    print("=" * 60)
    print("Unknown URL (should return an error)")
    print("=" * 60)
    print(fetch_listing("mock://does_not_exist"))
