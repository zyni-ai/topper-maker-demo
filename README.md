# Exam demo: material → question paper → rubric → evaluation

1. **Question paper** – upload study material (PDF/txt/md). It is chunked, embedded and retrieved
   per topic (RAG, `qgen.py`); an LLM writes MCQ / short / long questions.
2. **Rubric** – each question gets value-point marking (marks sum to the question's max, with
   `depends_on` cascades for numericals). Download question paper and rubric as PDFs.
3. **Evaluate** – upload a scanned handwritten answer sheet PDF. The vendored
   `topper_maker/` evaluation pipeline (from topper-maker-evaluation-pipeline) transcribes it,
   maps answers to questions and marks them against the generated rubric.

## Deploy (Streamlit Community Cloud)
share.streamlit.io -> New app -> pick this repo, branch `main`, main file `app.py`. Under Advanced settings -> Secrets add `OPENROUTER_API_KEY = "sk-or-..."` and `APP_PASSWORD = "choose-one"` (the app asks for it before use).

## Run locally
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
copy .env.example .env      # put your OpenRouter key in it
.venv\Scripts\streamlit run app.py
```
Optional env: `GEN_MODEL`, `EMBED_MODEL` (generation); `HTR_MODEL`, `EVAL_TEXT_MODEL`, … (evaluation).

Note: the generator is a standalone simplification of topper-maker-agents (which needs a Laravel
backend). Answer sheets are student data – don't enable LangSmith tracing on real ones.
