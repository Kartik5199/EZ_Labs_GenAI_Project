import fitz  # PyMuPDF
from typing import List, Dict


def extract_text_from_file(uploaded_file) -> str:
    """Extracts text from PDF or TXT using PyMuPDF or utf-8 decoding."""
    if uploaded_file.name.endswith(".pdf"):
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            return "\n".join(page.get_text() for page in doc)
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    else:
        return "Unsupported file type."


def get_text_stats(text: str) -> Dict[str, int]:
    """Returns basic stats for display."""
    return {
        "words": len(text.split()),
        "characters": len(text)
    }


def simulate_confidence(answer: str) -> float:
    """Simulates a fake confidence score based on answer length."""
    if not answer or len(answer) < 20:
        return round(40 + len(answer) * 2, 1)
    return round(85 + (len(answer) % 10), 1)


def highlight_text(text: str, answer: str, window: int = 100) -> str:
    """Highlights the portion of the text around the found answer."""
    if not text or not answer:
        return "No context available."
    try:
        pos = text.lower().find(answer.lower())
        if pos == -1:
            return text[:window] + "..."
        start = max(0, pos - window)
        end = min(len(text), pos + len(answer) + window)
        excerpt = text[start:end]
        highlighted = excerpt.replace(answer, f"**:blue[{answer}]**")
        if start > 0: highlighted = "..." + highlighted
        if end < len(text): highlighted += "..."
        return highlighted
    except Exception as e:
        return f"Error in highlighting: {str(e)}"


def format_answer_block(answer: str, confidence: float, paragraph_ref: str = "") -> str:
    """Formats the answer block with confidence, reference, and background color."""
    color = "green" if confidence >= 80 else "orange" if confidence >= 60 else "red"
    return f"""
<div style="background-color: #2b2b2b; padding: 1rem; border-radius: 6px; border: 1px solid #ddd;">
    <h4 style="margin-top: 0; color: white;">Answer:</h4>
    <p style="font-size: 16px; color: white;">{answer}</p>
    <p style="margin-top: 1rem; color: white;">
        <strong>Confidence:</strong> <span style="color:{color}; font-weight:bold;">{confidence}%</span><br/>
    </p>
    <p style="font-size: 12px; color: #ccc;">Model: Gemini Flash 2.0</p>
</div>
"""


def generate_prompt_for_questions(chunks: List[str]) -> str:
    """Creates a prompt to generate challenge questions from chunks."""
    joined_text = "\n\n".join(chunks[:10])
    return f"""You are a tutor assistant. Based on the document below, generate 3 logic-based or comprehension-focused questions:

{joined_text}

Only return the questions, one per line."""


def evaluate_user_answers_prompt(questions: List[str], answers: List[str], doc_chunks: List[str]) -> str:
    """Creates a strict grading prompt for user's answers."""
    qna = "\n".join([f"Q{i+1}: {q}\nA{i+1}: {a}" for i, (q, a) in enumerate(zip(questions, answers))])
    doc_context = "\n\n".join(doc_chunks[:15])

    return f"""You are an exam evaluator. Below are some questions and user's answers.

Document:
{doc_context}

User's Answers:
{qna}

Evaluate each answer using STRICT document facts only. Use the format:

---
**Qx: [question]**

- **User Answer**: [user's answer]
- **Evaluation**: Correct / Incorrect / Partial
- **Justification**: [why you marked it]
- **Correct Answer**: [best answer from document]
- **Supporting Chunk(s)**: [chunk or section where answer was found]
---
"""
