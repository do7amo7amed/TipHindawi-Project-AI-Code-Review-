import streamlit as st

from backend import build_vectorstore, load_llm, parse_issues, run_all

st.set_page_config(page_title="AI Code Review Assistant", page_icon="🧑‍💻", layout="wide")

st.title("🧑‍💻 AI Code Review Assistant")
st.caption("Mistral-7B-Instruct-v0.2 (fp16, Kaggle GPU) + RAG (FAISS) code reviewer")

with st.sidebar:
    st.header(" Model")
    st.write("**Model:** `mistralai/Mistral-7B-Instruct-v0.2`")
    st.write("**Precision:** fp16 (Kaggle GPU has enough VRAM, no quantization needed)")
    st.info(
        "The model (~14GB) downloads and loads into GPU memory the first "
        "time you click **Run Review**. This can take a few minutes."
    )
    if st.button("Pre-load model now"):
        with st.spinner("Loading model into GPU memory..."):
            load_llm()
        st.success("Model loaded and cached.")

st.subheader("1. Provide the code to review")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload a Python file (.py)", type=["py"])
with col2:
    reference_file = st.file_uploader(
        "Optional: extra reference doc for RAG context (.py/.txt/.md)",
        
    )

pasted_code = st.text_area("...or paste code here", height=280, placeholder="def foo():\n    pass")

file_name = "pasted_code.py"
code = ""

if uploaded_file is not None:
    code = uploaded_file.read().decode("utf-8", errors="ignore")
    file_name = uploaded_file.name
elif pasted_code.strip():
    code = pasted_code

run_clicked = st.button("Run Review", type="primary", disabled=not code.strip())

if run_clicked:
    print("STEP 1: Button clicked")

    reference_text = code
    if reference_file is not None:
        reference_text += "\n\n" + reference_file.read().decode("utf-8", errors="ignore")
    print("STEP 2: Loading model")
    with st.spinner("Loading model (first run only)..."):
        load_llm()

    print("STEP 2 DONE")

    print("STEP 3: Building vectorstore")
    with st.spinner("Building retrieval index..."):
        vectordb = build_vectorstore(reference_text)

    print("STEP 3 DONE")

    print("STEP 4: Running review")
    with st.spinner("Running analysis..."):
        analysis, issues, improvements = run_all(file_name, code, vectordb)

    st.session_state["analysis"] = analysis
    st.session_state["issues"] = issues
    st.session_state["improvements"] = improvements

if "analysis" in st.session_state:
    st.subheader("2. Results")
    tab1, tab2, tab3 = st.tabs(["Analysis", " Structured Issues", " Improvements"])

    with tab1:
        st.markdown(st.session_state["analysis"])

    with tab2:
        parsed = parse_issues(st.session_state["issues"])
        if parsed:
            st.json(parsed)
        else:
            st.warning("Could not parse a structured JSON block — showing raw output below.")
        with st.expander("Raw model output"):
            st.text(st.session_state["issues"])

    with tab3:
        st.markdown(st.session_state["improvements"])
