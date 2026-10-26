"""
Streamlit demo: upload a .java / .py / .st file -> extract class comments -> classify into 19 labels.

    streamlit run demo/app.py
"""
import html
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from inference import (LABEL_INFO, LANG_LABELS, LANG_NAMES, ROOT, classify, detect_lang, load_bundle, load_expected,
                       match_status, sample_files, trained_runs)

os.chdir(ROOT)  # paths in the configs (runs/, configs/, data/) are relative to the repo root

DEFAULT_FILENAMES = {"java": "Example.java", "python": "example.py", "pharo": "Example.class.st"}
LANG_COLORS = {"java": "#E0782F", "python": "#2C6BAA", "pharo": "#7A5BB5"}
LABEL_LANG = {lab: lang for lang, labs in LANG_LABELS.items() for lab in labs}
STATUS_TEXT = {"exact": "✔ correct", "partial": "~ partial", "wrong": "✘ wrong"}
STATUS_COLOR = {"exact": "#2E8B57", "partial": "#C08A00", "wrong": "#C0392B"}


@st.cache_resource(show_spinner="Loading model…")
def cached_bundle(config_path, ckpt_path, thr_path):
    return load_bundle(config_path, ckpt_path or None, thr_path or None)


def chip(label, prob=None, outline=False):
    color = LANG_COLORS.get(LABEL_LANG.get(label), "#888888")
    text = html.escape(label) + (f" · {prob:.2f}" if prob is not None else "")
    style = (f"border:1.5px dashed {color};" if outline else f"background:{color}26;border:1.5px solid {color};")
    return (f'<span title="{html.escape(LABEL_INFO.get(label, ""))}" style="{style}border-radius:12px;'
            f'padding:2px 10px;margin:2px 4px 2px 0;display:inline-block;font-size:0.85rem">{text}</span>')


# ===================== Sidebar =====================

st.set_page_config(page_title="Code Comment Classification", layout="wide")
st.title("Code Comment Classification")
st.caption("Code file (.java / .py / .st) → class comments → sentences normalized as in NLBSE’23 → "
           "multi-label transformer (19 labels)")

with st.sidebar:
    st.header("Model")
    runs = trained_runs()
    if not runs:
        st.error("No trained model found in runs/ (needs config_used.yaml + final_model.pt). "
                 "Train one with `python scripts/run_experiments.py`.")
        st.stop()
    run_dir = st.selectbox("Trained model", runs,
                           format_func=lambda p: f"{p.name}  ({(p / 'config_used.yaml').read_text().split('model_name:')[1].split()[0]})")
    with st.expander("Advanced"):
        ckpt_path = st.text_input("Checkpoint (empty = final_model.pt of the run)", "")
        thr_path = st.text_input("Threshold file (only for best_model_fold*.pt)", "")
    st.header("Prediction options")
    use_lang_mask = st.checkbox("Only labels of the file's language", value=True,
                                help="Java files only get Java labels, etc.")
    at_least_one = st.checkbox("Always assign at least 1 label", value=True,
                               help="In the training data, every sentence has at least 1 label.")
    with st.expander("What do the labels mean?"):
        for lang, labs in LANG_LABELS.items():
            st.markdown(f"**{LANG_NAMES[lang]}**")
            st.markdown("\n".join(f"- `{lab}`: {LABEL_INFO[lab]}" for lab in labs))

# ===================== Code source =====================

source_mode = st.radio("Code source", ["Sample file", "Upload a file", "Paste code"], horizontal=True)
source, filename = None, None
if source_mode == "Sample file":
    sample = st.selectbox("Sample file (demo/examples = showcase, demo/test_files = tests and edge cases)",
                          sample_files(),
                          format_func=lambda p: f"{p.parent.name}/{p.name}  ·  {LANG_NAMES[detect_lang(p.name)]}"
                          + ("  ·  with expected labels" if load_expected(p.name) else ""))
    filename, source = sample.name, sample.read_text(encoding="utf-8")
elif source_mode == "Upload a file":
    up = st.file_uploader("Choose a .java, .py or .st file (Pharo Tonel / File Out)", type=["java", "py", "st"])
    if up is not None:
        source, filename = up.getvalue().decode("utf-8", errors="replace"), up.name
else:
    c1, c2 = st.columns([1, 3])
    lang_choice = c1.selectbox("Language", list(LANG_NAMES), format_func=LANG_NAMES.get)
    filename = c2.text_input("File name (Java uses the file name as class_id)", DEFAULT_FILENAMES[lang_choice])
    code = st.text_area("Code", height=260)
    source = code if code.strip() else None

if not source:
    st.info("Choose, upload or paste a code file to start.")
    st.stop()
if detect_lang(filename) is None:
    st.error("The file name must end with .java, .py or .st")
    st.stop()
lang = detect_lang(filename)
with st.expander(f"View code · {filename}", expanded=False):
    st.code(source, language={"java": "java", "python": "python"}.get(lang))

# ===================== Classification =====================

try:
    bundle = cached_bundle(str(run_dir / "config_used.yaml"), ckpt_path, thr_path)
except Exception as e:
    st.error(f"Cannot load the model: {e}")
    st.stop()
try:
    with st.spinner(f"Classifying on {bundle['device']}…"):
        rows, probs, allowed = classify(bundle, source, filename, use_lang_mask, at_least_one)
except SyntaxError as e:
    st.error(f"Python syntax error in the file: {e}")
    st.stop()
if not rows:
    st.warning("No class comment found (a Javadoc /** */ right before a class, a Python class docstring, "
               "or a Pharo class comment). Method comments and file headers are not used by the model.")
    st.stop()

label_list = bundle["label_list"]
statuses = [match_status(r["pred"], r["expected"]) for r in rows]
checked = [s for s in statuses if s]
cols = st.columns(5 if checked else 3)
cols[0].metric("Classes", len({r["class"] for r in rows}))
cols[1].metric("Sentences", len(rows))
cols[2].metric("Labels predicted", sum(len(r["pred"]) for r in rows))
if checked:
    cols[3].metric("Correct vs expected", f"{checked.count('exact')}/{len(checked)}",
                   help="Exact match with the hand-written labels in demo/examples/expected_labels.json")
    cols[4].metric("Partial / wrong", f"{checked.count('partial')} / {checked.count('wrong')}")
st.caption(f"Model `{bundle['config']['model_name']}` · checkpoint `{bundle['checkpoint']}` · "
           f"device {bundle['device']} · thresholds: {bundle['thr_source']}")

tab_results, tab_pre, tab_table, tab_probs = st.tabs(
    ["Results by class", "1 · Preprocessing", "2 · Prediction table", "3 · Probabilities"])

with tab_results:
    current = None
    for i, (r, status) in enumerate(zip(rows, statuses)):
        if r["class"] != current:
            current = r["class"]
            st.markdown(f"#### class `{current}` <span style='opacity:0.6;font-size:0.85rem'>line {r['line']}</span>",
                        unsafe_allow_html=True)
        chips = "".join(chip(lab, probs[i, label_list.index(lab)]) for lab in r["pred"])
        extra = ""
        if status:
            extra = (f'<span style="color:{STATUS_COLOR[status]};font-weight:600;margin-left:6px">'
                     f'{STATUS_TEXT[status]}</span>')
            missed = [lab for lab in r["expected"] if lab not in r["pred"]]
            if missed:
                extra += ' <span style="opacity:0.7;font-size:0.85rem">expected:</span> ' + "".join(
                    chip(lab, outline=True) for lab in missed)
        st.markdown(f'<div style="padding:4px 0;border-bottom:1px solid rgba(128,128,128,0.2)">'
                    f'<code>{html.escape(r["raw"])}</code><br>{chips}{extra}</div>', unsafe_allow_html=True)

with tab_pre:
    st.markdown("Class comments are split into sentences and normalized exactly like the training data "
                "(lowercase, punctuation removed, `{@link X}` → `@link x`), then prefixed with the language and "
                "class: `[LANG] class_id | sentence`.")
    st.dataframe(pd.DataFrame({"Class": [r["class"] for r in rows], "Line": [r["line"] for r in rows],
                               "Original sentence": [r["raw"] for r in rows],
                               "Model input": [r["input"] for r in rows]}), width="stretch", hide_index=True)

result_df = pd.DataFrame({
    "Class": [r["class"] for r in rows],
    "Sentence": [r["raw"] for r in rows],
    "Predicted labels": [", ".join(r["pred"]) for r in rows],
    "Probability": [", ".join(f"{probs[i, label_list.index(lab)]:.2f}" for lab in r["pred"])
                    for i, r in enumerate(rows)],
})
if checked:
    result_df["Expected"] = [", ".join(r["expected"]) if r["expected"] else "" for r in rows]
    result_df["Match"] = [STATUS_TEXT.get(s, "") for s in statuses]

with tab_table:
    st.dataframe(result_df, width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sentences per label**")
        st.bar_chart(pd.Series([lab for r in rows for lab in r["pred"]]).value_counts())
    with c2:
        st.markdown("**Decision threshold per label** (median over the CV folds)")
        st.dataframe(pd.DataFrame({"Label": allowed,
                                   "Threshold": [float(bundle["thresholds"][label_list.index(l)]) for l in allowed]}),
                     width="stretch", hide_index=True)
    st.download_button("Download results (CSV)", result_df.to_csv(index=False).encode("utf-8"),
                       file_name=f"{Path(filename).stem}_predictions.csv", mime="text/csv")

with tab_probs:
    idx = [label_list.index(lab) for lab in allowed]
    prob_df = pd.DataFrame(probs[:, idx], columns=allowed)
    prob_df.insert(0, "Sentence", [r["raw"] for r in rows])
    st.dataframe(prob_df.style.format({lab: "{:.3f}" for lab in allowed}).background_gradient(
        cmap="Blues", subset=allowed, vmin=0, vmax=1), width="stretch", hide_index=True)
