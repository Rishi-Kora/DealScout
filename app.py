"""DealScout — Streamlit UI.

Run with:
    streamlit run app.py

Reads/writes the same output/<slug>/ folders as agents.py CLI does, so any
prior CLI runs show up in the "Recent runs" sidebar list.
"""

import json
import os
import re
from pathlib import Path

import streamlit as st

from agents import orchestrate
from observability import Scratchpad, TraceLogger

OUTPUT_DIR = Path(__file__).parent / "output"

# Real RC API on by default. The History Checker hits the local mock server
# at this URL — start it with `uvicorn mock_server:app --port 8080` in another
# terminal. To run pure-mock (no server needed), unset RC_API_URL.
os.environ.setdefault("RC_API_URL", "http://127.0.0.1:8080")
os.environ.setdefault("RC_API_KEY", "dev-secret-key")

# Recommendation -> (icon, Streamlit alert kind)
REC_STYLE = {
    "buy":         ("✅", "success"),
    "negotiate":   ("💬", "warning"),
    "investigate": ("🔍", "info"),
    "walk_away":   ("🚫", "error"),
}
SEVERITY_DOT = {"high": "🔴", "medium": "🟡", "low": "🟢"}


# =============================================================================
# Page setup
# =============================================================================

st.set_page_config(
    page_title="DealScout",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide the empty sidebar + its toggle arrow.
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] { display: none; }
        [data-testid="collapsedControl"] { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "current_slug" not in st.session_state:
    st.session_state.current_slug = None


# =============================================================================
# Helpers
# =============================================================================

def _load_json(folder: Path, name: str):
    """Load output/<slug>/<name> if it exists, else None."""
    p = folder / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _save_run(slug: str, verdict, reports, reviews, scratchpad):
    """Persist a fresh run to output/<slug>/."""
    folder = OUTPUT_DIR / slug
    folder.mkdir(parents=True, exist_ok=True)

    (folder / "verdict.json").write_text(
        json.dumps(verdict.model_dump(mode="json"), indent=2)
    )
    (folder / "scratchpad.json").write_text(
        json.dumps(scratchpad.to_dict(), indent=2)
    )
    (folder / "critic_notes.json").write_text(
        json.dumps([r.model_dump(mode="json") for r in reviews], indent=2)
    )

    file_map = {
        "fact_sheet": "fact_sheet.json",
        "visual_inspection": "visual_inspection.json",
        "price_audit": "price_audit.json",
        "history_report": "history_report.json",
    }
    for key, fname in file_map.items():
        if reports.get(key) is not None:
            (folder / fname).write_text(json.dumps(reports[key], indent=2))


# =============================================================================
# Main — input on top, results below
# =============================================================================

st.title("DealScout")
st.caption("Multi-agent used-car verdicts")

with st.container(border=True):
    st.subheader("Analyse a listing")
    custom = st.text_input(
        "Listing URL",
        placeholder="Paste the listing URL here....",
        help=(
            "Paste a mock URL for a built-in test fixture:\n"
            "• mock://swift_clean\n"
            "• mock://honda_sparse\n"
            "• mock://i20_noisy\n"
            "• mock://innova_dealer\n\n"
            "Real http(s) URLs are NOT scraped in v1."
        ),
        label_visibility="visible",
    )
    listing_input = custom.strip()

    if listing_input:
        if listing_input.startswith("mock://"):
            target_slug = listing_input.replace("mock://", "")
        else:
            target_slug = re.sub(
                r"[^a-zA-Z0-9_]+", "_", listing_input[:40]
            ).strip("_") or "custom_run"

        if st.session_state.current_slug != target_slug:
            with st.status("Running 5 agents…", expanded=True) as status:
                try:
                    listing_folder = OUTPUT_DIR / target_slug
                    listing_folder.mkdir(parents=True, exist_ok=True)

                    scratchpad = Scratchpad()
                    with TraceLogger(listing_folder / "trace.jsonl") as tracer:
                        status.update(label="Coordinator dispatching specialists…")
                        verdict, reports, reviews = orchestrate(
                            f"Analyse this listing: {listing_input}",
                            scratchpad=scratchpad,
                            tracer=tracer,
                            verbose=False,
                        )

                    _save_run(target_slug, verdict, reports, reviews, scratchpad)
                    st.session_state.current_slug = target_slug
                    status.update(label="Done", state="complete")
                except Exception as exc:
                    status.update(label=f"Failed: {exc}", state="error")
                    st.exception(exc)
                    st.stop()
            st.rerun()


# =============================================================================
# Results — verdict + tabs (only when a run is loaded)
# =============================================================================

slug = st.session_state.current_slug

if slug is None:
    st.stop()


run_folder = OUTPUT_DIR / slug
verdict_data = _load_json(run_folder, "verdict.json")
fact_sheet   = _load_json(run_folder, "fact_sheet.json")
visual       = _load_json(run_folder, "visual_inspection.json")
price        = _load_json(run_folder, "price_audit.json")
history      = _load_json(run_folder, "history_report.json")
scratchpad_d = _load_json(run_folder, "scratchpad.json")
critic_data  = _load_json(run_folder, "critic_notes.json")

if verdict_data is None:
    st.error(f"No verdict.json in `output/{slug}/`. Re-run analysis.")
    st.stop()


# ---- Header: car title + verdict banner ----

if fact_sheet:
    parts = [
        str(fact_sheet.get("year") or ""),
        fact_sheet.get("make") or "",
        fact_sheet.get("model") or "",
        fact_sheet.get("variant") or "",
    ]
    car_title = " ".join(p for p in parts if p).strip()
    if car_title:
        st.title(car_title)

rec = verdict_data["recommendation"]
icon, kind = REC_STYLE.get(rec, ("❓", "info"))
banner = (
    f"{icon} **{rec.replace('_', ' ').upper()}** · "
    f"{verdict_data['confidence']} confidence — {verdict_data['headline']}"
)
{
    "success": st.success,
    "warning": st.warning,
    "info":    st.info,
    "error":   st.error,
}[kind](banner)


# ---- Tabs ----

tab_verdict, tab_specialists, tab_critic = st.tabs([
    "📋 Verdict",
    "👥 Specialists",
    "✏️ Critic",
])

# --- Verdict tab ---
with tab_verdict:
    st.subheader("Reasoning")
    st.write(verdict_data.get("reasoning", ""))

    if verdict_data.get("key_concerns"):
        st.subheader("Key concerns")
        for c in verdict_data["key_concerns"]:
            st.markdown(f"- {c}")

    if verdict_data.get("disagreements"):
        st.subheader("⚡ Disagreements between specialists")
        for d in verdict_data["disagreements"]:
            with st.container(border=True):
                st.markdown(f"**Between** `{', '.join(d['between'])}`")
                st.markdown(f"**About:** {d['on']}")
                st.markdown(f"**Resolution:** {d['resolution']}")

    st.caption(
        "Specialists consulted: "
        + ", ".join(verdict_data.get("specialists_consulted", []))
    )


# --- Specialists tab ---
with tab_specialists:
    sub = st.tabs(["📑 Listing", "📸 Visual", "💰 Price", "📜 History"])

    with sub[0]:
        if fact_sheet:
            c1, c2, c3 = st.columns(3)
            c1.metric("Year", fact_sheet.get("year") or "?")
            km = fact_sheet.get("km_driven")
            c2.metric("KM driven", f"{km:,}" if isinstance(km, int) else "?")
            ap = fact_sheet.get("asking_price")
            c3.metric("Asking", f"₹{ap:,}" if isinstance(ap, int) else "?")
            if fact_sheet.get("missing_fields"):
                st.warning("Missing: " + ", ".join(fact_sheet["missing_fields"]))
        else:
            st.warning("No fact sheet")

    with sub[1]:
        if visual:
            c1, c2 = st.columns(2)
            c1.metric("Photos analysed", visual.get("photo_count", 0))
            c2.metric("Overall condition", visual.get("overall_condition", "?"))

            photos = (fact_sheet or {}).get("photos") or []
            if photos:
                st.subheader("Photos")
                cols = st.columns(3)
                for i, url in enumerate(photos):
                    cols[i % 3].image(url, use_container_width=True)

            if visual.get("concerns"):
                st.subheader("Concerns")
                for c in visual["concerns"]:
                    sev = c.get("severity", "")
                    st.markdown(
                        f"{SEVERITY_DOT.get(sev, '⚪')} "
                        f"**{c['type']}** ({sev}) — {c['description']}"
                    )
        else:
            st.warning("No visual report")

    with sub[2]:
        if price:
            c1, c2, c3 = st.columns(3)
            c1.metric("Asking", f"₹{price.get('asking_price', 0):,}")
            c2.metric("Market median", f"₹{price.get('market_band', {}).get('p50', 0):,}")
            c3.metric("Δ from median", f"{price.get('delta_pct', 0):+.1f}%")
            st.markdown(f"**Verdict:** `{price.get('verdict', '?')}`")
            if price.get("notes"):
                st.caption(price["notes"])
        else:
            st.warning("No price audit")

    with sub[3]:
        if history:
            c1, c2 = st.columns(2)
            c1.metric("Assessment", history.get("overall_assessment", "?"))
            c2.metric("Red flags", len(history.get("red_flags", [])))
            if history.get("red_flags"):
                st.subheader("Red flags")
                for rf in history["red_flags"]:
                    sev = rf.get("severity", "")
                    st.markdown(
                        f"{SEVERITY_DOT.get(sev, '⚪')} "
                        f"**{rf['type']}** ({sev}) — {rf['description']}"
                    )
            if history.get("iib_recommendation"):
                st.info("📝 " + history["iib_recommendation"])
        else:
            st.warning("No history report")


# --- Critic tab ---
with tab_critic:
    if not critic_data:
        st.info("No critic notes")
    else:
        for review in critic_data:
            round_n = review.get("revision_round", 0)
            approved = review.get("approved", False)
            issues = review.get("issues", [])
            label = (
                f"✅ Round {round_n} — APPROVED"
                if approved
                else f"✏️ Round {round_n} — {len(issues)} issue(s)"
            )
            st.subheader(label)
            if review.get("summary"):
                st.write(review["summary"])
            for issue in issues:
                sev = issue.get("severity", "")
                st.markdown(
                    f"{SEVERITY_DOT.get(sev, '⚪')} "
                    f"**{issue.get('type')}** ({sev})"
                )
                st.caption(f"Issue: {issue.get('description')}")


