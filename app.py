import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
from rag_utils import RAGVectorStore, chunk_text
from utils import (
    extract_text_from_file,
    get_text_stats,
    highlight_text,
    simulate_confidence,
    format_answer_block,
    generate_prompt_for_questions,
    evaluate_user_answers_prompt
)

# --- Load API Key ---
load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    st.error("API_KEY not found in .env file.")
    st.stop()
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")  # or "gemini-2.0-flash" if available

# --- Streamlit Config ---
st.set_page_config(page_title="Smart Assistant for Research Summarization", layout="wide")
st.title("Smart Assistant for Research Summarization")

# --- Upload Section ---
uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
if uploaded_file:
    with st.spinner("Extracting and chunking text..."):
        doc_text = extract_text_from_file(uploaded_file)
        paragraphs = doc_text.split("\n\n")
        chunks = chunk_text(doc_text)

        vector_store = RAGVectorStore()
        vector_store.build_index(chunks)

    stats = get_text_stats(doc_text)
    with st.expander("Document Stats"):
        st.write(f"**Words:** {stats['words']:,}")
        st.write(f"**Characters:** {stats['characters']:,}")
        st.write(f"**Chunks Indexed:** {len(chunks)}")

    with st.spinner("Generating summary..."):
        summary_prompt = f"Summarize the following document in 150 words:\n\n{doc_text[:3000]}"
        response = model.generate_content(summary_prompt)
        summary = response.text
    with st.expander("Summary", expanded=True):
        st.markdown(summary)

    # --- Mode Toggle ---
    st.markdown("## Choose Mode")
    mode = st.radio("Select Mode", ["Ask Anything", "Challenge Me"])

    # --- Q&A Mode ---
    if mode == "Ask Anything":
        question = st.text_input("Ask a question about the document")
        if question:
            with st.spinner("Retrieving context and answering..."):
                top_chunks = vector_store.query(question, top_k=5)
                context = "\n\n".join(top_chunks)
                prompt = f"""ONLY use the text provided in the "Context" section below.Do NOT use any outside knowledge or make up facts.If the answer is NOT found in the context, respond ONLY with:"Answer not found in the provided document."Mention paragraph if applicable.

Document:
{context}

Question:
{question}
"""
                try:
                    answer = model.generate_content(prompt).text
                except Exception as e:
                    st.error(f"Gemini error: {e}")
                    st.stop()

                confidence = simulate_confidence(answer)
                st.markdown("Response")
                st.markdown(format_answer_block(answer, confidence), unsafe_allow_html=True)

    # --- Challenge Me Mode ---
    elif mode == "Challenge Me":
        if st.button("Generate Challenge Questions"):
            with st.spinner("Creating questions..."):
                q_prompt = generate_prompt_for_questions(chunks)
                response = model.generate_content(q_prompt)
                questions = [q.strip("-•1234567890. ") for q in response.text.strip().split("\n") if q.strip()]
                st.session_state.questions = questions[:3]
                st.session_state.answers = [""] * 3
                st.session_state.results = None
                st.success("Questions generated!")

        if "questions" in st.session_state:
            for i, q in enumerate(st.session_state.questions):
                st.markdown(f"**Q{i+1}: {q}**")
                st.session_state.answers[i] = st.text_area("Your Answer", key=f"ans_{i}")

            if st.button("Submit Answers"):
                with st.spinner("Evaluating..."):
                    eval_prompt = evaluate_user_answers_prompt(
                        st.session_state.questions,
                        st.session_state.answers,
                        chunks
                    )
                    evaluation = model.generate_content(eval_prompt).text
                    st.session_state.results = evaluation

        if st.session_state.get("results"):
            st.markdown("### Evaluation")
            st.markdown(st.session_state.results)