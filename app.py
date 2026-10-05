"""Demo: study material -> question paper -> rubric -> evaluate handwritten answer sheet.

Run:  .venv\\Scripts\\streamlit run app.py     (needs OPENROUTER_API_KEY in .env)
"""

from __future__ import annotations

import base64
import hmac
import html
import io
import re
import os
import tempfile

import pymupdf
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

import qgen
from topper_maker.evaluation import EvaluationConfig, EvaluationPipeline
from topper_maker.evaluation.schemas.state import EvaluationRequest

load_dotenv()
st.set_page_config(page_title="Exam Demo", page_icon="📝", layout="centered")
st.title("📝 Question paper → Rubric → Evaluation")

# Optional access gate: set APP_PASSWORD (env or Streamlit secret) to require it. Unset = open (local use).
_pw = os.getenv("APP_PASSWORD")
if _pw and not st.session_state.get("authed"):
    entered = st.text_input("Access password", type="password")
    if entered and hmac.compare_digest(entered, _pw):
        st.session_state.authed = True
        st.rerun()
    elif entered:
        st.error("Wrong password.")
    st.stop()

api_key = os.getenv("OPENROUTER_API_KEY") or st.sidebar.text_input(
    "OpenRouter API key", type="password", help="Or put it in .env as OPENROUTER_API_KEY")
if not api_key:
    st.info("Enter your OpenRouter API key in the sidebar to begin.")
    st.stop()
os.environ["OPENROUTER_API_KEY"] = api_key  # the evaluation pipeline reads it from the env
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def m(x: float) -> str:
    """Marks to 1 decimal place."""
    return f"{x:.1f}"


def plain(text: str) -> str:
    """Make prose-in-LaTeX wrap.

    The pipeline returns answers as LaTeX (text{...} inside $...$), which the browser lays
    out as one unbreakable math line. If the answer is only prose, drop the LaTeX wrapper;
    real maths (frac, ^, _ ...) is left untouched.
    """
    bs = chr(92)
    t = re.sub(re.escape(bs) + r"text(?:bf|it)?\{([^{}]*)\}", r"\1", text)
    t = t.replace(bs * 2, "\n\n")  # LaTeX line break -> paragraph break
    if re.search(re.escape(bs) + r"[a-zA-Z]+|[\^_]", t):
        return text
    return t.replace("$", "").replace(bs + " ", " ")


def to_pdf(title: str, body_html: str) -> bytes:
    """Render simple HTML to PDF bytes."""
    doc = pymupdf.Story(html=f"<h2>{html.escape(title)}</h2>{body_html}")
    out = io.BytesIO()
    writer = pymupdf.DocumentWriter(out)
    more = 1
    while more:
        dev = writer.begin_page(pymupdf.paper_rect("a4"))
        more, _ = doc.place(pymupdf.paper_rect("a4") + (40, 40, -40, -40))
        doc.draw(dev)
        writer.end_page()
    writer.close()
    return out.getvalue()


def paper_html(qs, with_rubric: bool) -> str:
    out, section = [], None
    for q in qs:
        if q.section_id != section:
            section = q.section_id
            label = next(v[1] for v in qgen.QUESTION_TYPES.values() if v[0] == section)
            out.append(f"<h3>{label}</h3>")
        stmt = html.escape(q.question_statement).replace("\n", "<br>")
        out.append(f"<p><b>Q{q.question_number}.</b> {stmt} <i>[{m(q.max_score)}]</i></p>")
        if with_rubric:
            out.append(f"<p><i>Model answer:</i> {html.escape(q.expected_answer or '')}</p><ul>")
            for p in q.rubric_points:
                dep = f" (needs: {', '.join(p.depends_on)})" if p.depends_on else ""
                out.append(f"<li>{html.escape(p.description)} - <b>{m(p.marks)}</b>{dep}</li>")
            out.append("</ul>")
    return "".join(out)


tab1, tab2, tab3 = st.tabs(["1 · Question paper", "2 · Rubric", "3 · Evaluate"])

# ------------------------------------------------------------------ 1. generate
with tab1:
    subject = st.text_input("Subject", "Biology")
    files = st.file_uploader("Study material (PDF / txt / md)", type=["pdf", "txt", "md"],
                             accept_multiple_files=True)
    c1, c2, c3 = st.columns(3)
    n_mcq = c1.number_input("MCQs (1 mark)", 0, 20, 5)
    n_short = c2.number_input("Short (2 marks)", 0, 10, 3)
    n_long = c3.number_input("Long (5 marks)", 0, 5, 2)

    if st.button("Generate question paper", type="primary", disabled=not files):
        with st.status("Working...", expanded=True) as status:
            try:
                pages = [p for f in files for p in qgen.extract_text(f.name, f.getvalue())]
                chunks = qgen.chunk_pages(pages)
                if not chunks:
                    raise ValueError("No readable text found in the material (scanned PDF?).")
                st.session_state.questions = qgen.generate_paper(
                    client, chunks, subject,
                    {"mcq": n_mcq, "short": n_short, "long": n_long}, progress=st.write)
                st.session_state.subject = subject
                st.session_state.pop("result", None)
                status.update(label="Done", state="complete")
            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(str(e))

    qs = st.session_state.get("questions")
    if qs:
        st.success(f"{len(qs)} questions · {m(sum(q.max_score for q in qs))} marks")
        st.download_button("⬇ Download question paper (PDF)",
                           to_pdf(f"{subject} - Question Paper", paper_html(qs, False)),
                           "question_paper.pdf")
        for q in qs:
            st.markdown(f"**Q{q.question_number}.** {q.question_statement}  \n"
                        f"*[{m(q.max_score)} marks · {q.topic}]*")
    else:
        st.info("Upload material and click Generate.")

# ------------------------------------------------------------------ 2. rubric
with tab2:
    qs = st.session_state.get("questions")
    if not qs:
        st.info("Generate a question paper first.")
    else:
        st.download_button("⬇ Download rubric (PDF)",
                           to_pdf(f"{st.session_state.subject} - Marking Rubric", paper_html(qs, True)),
                           "rubric.pdf")
        for q in qs:
            with st.expander(f"Q{q.question_number} · {m(q.max_score)} marks"):
                st.markdown(q.question_statement)
                st.markdown(f"**Model answer:** {q.expected_answer}")
                st.table([{"Point": p.description, "Marks": m(p.marks),
                           "Needs": ", ".join(p.depends_on) or "-"} for p in q.rubric_points])

# ------------------------------------------------------------------ 3. evaluate
with tab3:
    qs = st.session_state.get("questions")
    if not qs:
        st.info("Generate a question paper first.")
    else:
        sheet = st.file_uploader("Scanned handwritten answer sheet (PDF)", type=["pdf"])
        if st.button("Evaluate", type="primary", disabled=not sheet):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(sheet.getvalue())
            try:
                cfg = EvaluationConfig.from_env()
                cfg.allow_local_paths = True
                cfg.enable_s3 = False
                req = EvaluationRequest(
                    user_id="demo", user_test_id="demo-1", skill=st.session_state.subject.lower(),
                    class_type="demo", answer_sheet_url=tmp.name, questions_list=qs)
                with st.spinner("Reading handwriting and marking against the rubric (1-3 min)..."):
                    st.session_state.result = EvaluationPipeline(cfg).evaluate_sync(req)
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
            finally:
                os.remove(tmp.name)

        res = st.session_state.get("result")
        if res:
            if not res.is_valid:
                st.error(f"Answer sheet rejected: {res.rejection_reason}")
            else:
                m1, m2 = st.columns(2)
                m1.metric("Score", f"{m(res.total_marks)} / {m(res.max_marks)}")
                m2.metric("Percentage", f"{res.percentage:.1f}%")
                if res.needs_human_review:
                    st.warning("Flagged for human review: " +
                               ", ".join(r.value for r in res.review_reasons))
                by_id = {q.id: q for q in qs}
                for r in res.responses:
                    q = by_id[r.id]
                    icon = "✅" if r.is_correct else ("🟡" if r.score > 0 else "❌")
                    with st.expander(f"{icon} Q{q.question_number} · {m(r.score)}/{m(r.max_score)}"):
                        st.markdown(f"**Question:** {q.question_statement}")
                        crops = res.answer_crops.get(r.id, [])
                        if crops:
                            st.markdown("**Student's handwriting:**")
                            for c in crops:
                                st.image(base64.b64decode(c["image_b64"]),
                                         caption=f"Page {c['page']}", use_container_width=True)
                        st.markdown(f"**Transcribed answer:** {plain(r.user_answer) if r.user_answer else '_not found_'}")
                        st.markdown(f"**Feedback:** {r.feedback}")
                        if r.rubric_breakdown:
                            st.table([{"Point": a.description,
                                       "Awarded": f"{m(a.marks_awarded)}/{m(a.marks_possible)}",
                                       "Why": a.rationale} for a in r.rubric_breakdown])
