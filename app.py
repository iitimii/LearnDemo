"""
Ribara — Job Analyzer & Learning Tutor (Streamlit)

Usage:
    streamlit run app.py

This UI wraps your existing core modules:
    - entry.py  (JobAnalyzer)
    - learning.py (LearningPlatform)

It keeps core logic headless and adds a clean, demo-ready interface with:
    • URL/Text job input
    • Structured report view (skills + gaps)
    • Skill selection and an assertive tutor chat
    • Lightweight progress tracking per skill
    • Export/Import of reports and transcripts
"""

import os
import io
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import streamlit as st

# --- Your core modules ---
from entry import JobAnalyzer
from learning import LearningPlatform

from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# -------------------- Config --------------------
st.set_page_config(
    page_title="Ribara — Job Analyzer & Tutor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_COLOR = "#111827"  # Tailwind slate-900 feel
ACCENT_COLOR = "#4F46E5"    # Indigo-600

# -------------------- Styling --------------------
CUSTOM_CSS = f"""
<style>
    .stApp {{
        background: #0b1020;
        color: #E5E7EB;
    }}
    header {{ visibility: hidden; }}
    .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}
    .ribara-title {{
        font-size: 2rem; font-weight: 800; letter-spacing: 0.2px;
        color: #E5E7EB; margin-bottom: 0.25rem;
    }}
    .ribara-sub {{ color: #9CA3AF; margin-bottom: 1.0rem; }}
    .metric-card {{
        border: 1px solid #1f2937; border-radius: 16px; padding: 14px 16px;
        background: #0f172a; box-shadow: 0 0 0 1px rgba(79,70,229,0.05) inset;
    }}
    .accent {{ color: {ACCENT_COLOR}; }}
    .chip {{
        display: inline-block; padding: 4px 10px; border-radius: 999px;
        background: #111827; border: 1px solid #1f2937; color: #D1D5DB;
        font-size: 12px; margin-right: 6px; margin-bottom: 4px;
    }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------- Helpers --------------------

def _load_user_data(default_path: str) -> Dict[str, Any]:
    """Load user profile JSON from disk or fallback."""
    try:
        with open(default_path, "r") as f:
            return json.load(f)
    except Exception:
        return {"role": "Unknown", "assessment_result": {}}


def _download_button(label: str, data: Any, file_name: str, mime: str = "application/json"):
    buf = io.BytesIO()
    if isinstance(data, (dict, list)):
        buf.write(json.dumps(data, indent=2).encode("utf-8"))
    elif isinstance(data, str):
        buf.write(data.encode("utf-8"))
    else:
        buf.write(str(data).encode("utf-8"))
    st.download_button(label, buf.getvalue(), file_name=file_name, mime=mime)


def _init_state():
    ss = st.session_state
    ss.setdefault("api_key", os.getenv("GEMINI_API_KEY", ""))
    ss.setdefault("user_data", _load_user_data("../users_data/full_stack_dev_user.json"))
    ss.setdefault("report", None)
    ss.setdefault("messages", [])           # Chat transcript
    ss.setdefault("current_skill", None)
    ss.setdefault("progress", {})           # {skill: {proficiency_estimate, notes, last_updated}}
    ss.setdefault("turn_count", 0)
    ss.setdefault("tutor_started", False)


_init_state()

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("<div class='ribara-title'>Ribara</div>", unsafe_allow_html=True)
    st.markdown("<div class='ribara-sub'>Job Analyzer & Learning Tutor</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("Configuration")

    st.session_state.api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.api_key,
        help="Needed to run analysis and tutor. Stored only in session.",
    )
    st.caption("Tip: Set GEMINI_API_KEY env var to skip this step.")

    st.markdown("**User Profile**")
    uploaded_user = st.file_uploader("Upload user_data JSON (optional)", type=["json"])
    if uploaded_user is not None:
        try:
            st.session_state.user_data = json.load(uploaded_user)
            st.success("User profile loaded.")
        except Exception as e:
            st.error(f"Failed to parse JSON: {e}")

    st.json(st.session_state.user_data, expanded=False)

    st.markdown("**Demo Presets**")
    if st.button("Load Demo Job Text"):
        st.session_state["demo_text"] = (
            "Founding Mid Level Front End Engineer | TypeScript, React, Y.js, TipTap | 3+ years | AI startup"
        )

# -------------------- Header --------------------
col1, col2 = st.columns([0.75, 0.25])
with col1:
    st.markdown("<div class='ribara-title'>Analyze a Job, Close the Gaps, Learn Fast.</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ribara-sub'>Paste a job description or link, get a structured report, then learn with an assertive tutor." \
        "</div>",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown("<div class='metric-card'>",
                unsafe_allow_html=True)
    st.metric("Session", value=datetime.now().strftime("%Y-%m-%d"))
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- Input Section --------------------
with st.container():
    tabs = st.tabs(["Analyze", "Skills", "Tutor", "Progress", "Settings"])  # Navigate core flows

# ========== ANALYZE TAB ==========
with tabs[0]:
    st.subheader("1) Job Input")
    job_text = st.text_area(
        "Paste Job Description (or paste a URL)",
        height=200,
        placeholder="Paste full job description text here, or a URL like https://...",
        value=st.session_state.get("demo_text", ""),
    )

    colA, colB = st.columns([0.2, 0.8])
    with colA:
        analyze_btn = st.button("Analyze", type="primary", use_container_width=True)
    with colB:
        st.caption("We strip noise, extract the role, skills, and compare with your profile.")

    if analyze_btn:
        if not st.session_state.api_key:
            st.error("API key is required.")
        elif not job_text.strip():
            st.error("Please paste a job description or a URL.")
        else:
            analyzer = JobAnalyzer(api_key=st.session_state.api_key, user_data=st.session_state.user_data)
            with st.spinner("Analyzing job... this usually takes a few seconds"):
                try:
                    if job_text.strip().lower().startswith("http"):
                        report = analyzer.run_analysis("url", url=job_text.strip())
                    else:
                        report = analyzer.run_analysis("manual", description=job_text)
                    st.session_state.report = report
                    st.success("Analysis complete.")
                except Exception as e:
                    st.exception(e)

    if st.session_state.report:
        st.subheader("2) Report")
        rep = st.session_state.report

        top = st.columns(4)
        with top[0]:
            st.metric("Job Role", rep.get("job_role", "-"))
        with top[1]:
            st.metric("Required Skills", rep.get("required_skills_count", 0))
        with top[2]:
            st.metric("Skill Gaps", rep.get("skill_gaps_count", 0))
        with top[3]:
            st.metric("User Role", rep.get("user_role", "-"))

        st.markdown("**Required Skills**")
        st.dataframe(rep.get("required_skills", []), use_container_width=True, hide_index=True)

        st.markdown("**Skill Gaps**")
        st.dataframe(rep.get("skill_gaps", []), use_container_width=True, hide_index=True)

        exp = st.expander("Raw JSON")
        with exp:
            st.json(rep)

        down_cols = st.columns([0.2,0.2,0.6])
        with down_cols[0]:
            _download_button("Download Report JSON", rep, file_name=f"ribara_report_{int(time.time())}.json")
        with down_cols[1]:
            if st.button("Use in Tutor", type="primary"):
                st.toast("Report loaded into Tutor tab.")
                st.experimental_rerun()

# ========== SKILLS TAB ==========
with tabs[1]:
    st.subheader("Choose a Skill to Focus On")
    rep = st.session_state.report
    if not rep:
        st.info("Run an analysis first in the Analyze tab.")
    else:
        gaps: List[Dict[str, Any]] = rep.get("skill_gaps", [])
        if not gaps:
            st.success("No gaps found. You're job-ready for this posting.")
        else:
            skill_names = [g.get("skill_name", "(unnamed)") for g in gaps]
            default_idx = skill_names.index(st.session_state.current_skill) if st.session_state.current_skill in skill_names else 0
            choice = st.selectbox("Focus Skill", options=skill_names, index=default_idx)

            # Persist selection to session
            st.session_state.current_skill = choice

            # Quick glance metrics
            sel_gap = next((g for g in gaps if g.get("skill_name") == choice), None)
            if sel_gap:
                c1, c2, c3 = st.columns(3)
                c1.metric("Your Level", sel_gap.get("current_level", "-"))
                c2.metric("Required Level", sel_gap.get("required_level", "-"))
                c3.metric("Severity", sel_gap.get("gap_severity", "-"))

            st.caption("Tip: Head to the Tutor tab to start an assertive lesson on this skill.")

# ========== TUTOR TAB ==========
with tabs[2]:
    st.subheader("Assertive Tutor")

    if not st.session_state.report:
        st.info("Run an analysis and pick a skill first.")
    else:
        # Ensure a current skill is chosen
        gaps = st.session_state.report.get("skill_gaps", [])
        if not gaps:
            st.success("No gaps to teach. Try another job or use Required Skills for review.")
        else:
            if st.session_state.current_skill is None:
                st.session_state.current_skill = gaps[0].get("skill_name")

            # Initialize platform per-session
            platform = LearningPlatform(api_key=st.session_state.api_key, report=st.session_state.report)
            platform.current_skill = st.session_state.current_skill
            platform.turn_count = st.session_state.turn_count

            # Control bar
            lcol, rcol = st.columns([0.6, 0.4])
            with lcol:
                st.markdown(f"**Current Skill:** <span class='accent'>{st.session_state.current_skill}</span>", unsafe_allow_html=True)
            with rcol:
                start_intro = st.checkbox("Tutor should introduce lesson first", value=not st.session_state.tutor_started)
                if st.button("Start / Reset Session", type="primary"):
                    st.session_state.messages = []
                    st.session_state.turn_count = 0
                    st.session_state.tutor_started = False
                    st.toast("Tutor session reset.")

            st.divider()

            # Chat area
            chat_container = st.container()

            # Tutor starts first when not started
            if not st.session_state.tutor_started and start_intro:
                with st.spinner("Tutor is preparing an intro..."):
                    intro = platform.tutor_reply("", intro=True)
                st.session_state.messages.append({"role": "assistant", "content": intro})
                st.session_state.tutor_started = True

            # Render transcript
            for m in st.session_state.messages:
                st.chat_message(m["role"]).markdown(m["content"])

            # Input box
            if user_msg := st.chat_input(placeholder="Ask a question or answer the tutor..."):
                st.session_state.messages.append({"role": "user", "content": user_msg})
                with st.spinner("Tutor thinking..."):
                    reply = platform.tutor_reply(user_msg)
                st.session_state.messages.append({"role": "assistant", "content": reply})

                # Update turn counter
                st.session_state.turn_count += 1
                platform.turn_count = st.session_state.turn_count

                # Periodic update notice
                if st.session_state.turn_count % 5 == 0:
                    st.info("Tutor refreshed with latest resources.")

            # Footer controls
            cdl, cdr = st.columns(2)
            with cdl:
                _download_button(
                    "Download Transcript",
                    st.session_state.messages,
                    file_name=f"ribara_transcript_{int(time.time())}.json",
                )
            with cdr:
                if st.button("Clear Transcript"):
                    st.session_state.messages = []
                    st.toast("Transcript cleared.")

# ========== PROGRESS TAB ==========
with tabs[3]:
    st.subheader("Progress & Notes")

    if not st.session_state.report:
        st.info("Analyze a job first to seed skills/gaps.")
    else:
        # Ensure entry for selected skill
        skill = st.session_state.current_skill
        if skill:
            st.markdown(f"### {skill}")
            prog = st.session_state.progress.setdefault(skill, {
                "proficiency_estimate": "beginner",
                "notes": "",
                "last_updated": datetime.now().isoformat(timespec="seconds"),
            })
            colp1, colp2 = st.columns([0.4, 0.6])
            with colp1:
                prog["proficiency_estimate"] = st.selectbox(
                    "Your current estimate",
                    ["none", "beginner", "intermediate", "advanced", "expert"],
                    index=["none","beginner","intermediate","advanced","expert"].index(prog["proficiency_estimate"]),
                    help="Self-estimate; use conservatively.",
                )
            with colp2:
                prog["notes"] = st.text_area("Notes / Key takeaways", value=prog.get("notes",""), height=140)

            if st.button("Save Progress", type="primary"):
                prog["last_updated"] = datetime.now().isoformat(timespec="seconds")
                st.session_state.progress[skill] = prog
                st.success("Progress saved.")

        st.markdown("---")
        st.markdown("#### All Progress JSON")
        st.json(st.session_state.progress, expanded=False)
        _download_button("Download Progress", st.session_state.progress, file_name="ribara_progress.json")

# ========== SETTINGS TAB ==========
with tabs[4]:
    st.subheader("Settings & Utilities")
    st.caption("Tune behavior or export/import artifacts.")

    # Import report
    imp_col1, imp_col2 = st.columns(2)
    with imp_col1:
        uploaded_report = st.file_uploader("Import a report JSON", type=["json"], key="rep_upl")
        if uploaded_report is not None:
            try:
                st.session_state.report = json.load(uploaded_report)
                st.success("Report imported.")
            except Exception as e:
                st.error(f"Failed to import: {e}")

    with imp_col2:
        uploaded_trans = st.file_uploader("Import a transcript JSON", type=["json"], key="trs_upl")
        if uploaded_trans is not None:
            try:
                st.session_state.messages = json.load(uploaded_trans)
                st.success("Transcript imported.")
            except Exception as e:
                st.error(f"Failed to import transcript: {e}")

    st.markdown("---")
    st.caption("About: Ribara converts job descriptions into actionable learning. Built for demos, designed for production.")
