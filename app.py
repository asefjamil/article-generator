import os
import re
import json
import requests
import io
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

# Optional: google generative AI SDK
try:
    from google import generativeai as genai
except Exception:
    genai = None

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="1-Click Article Generator", page_icon="📝", layout="wide")

# -----------------------------
# Keys and config
# -----------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") if st.secrets else None
GEMINI_API_KEY = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY") if st.secrets else None
OPENROUTER_API_KEY = OPENROUTER_API_KEY or os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

HAS_GEMINI = bool(GEMINI_API_KEY and genai)
HAS_OPENROUTER = bool(OPENROUTER_API_KEY)

if not HAS_GEMINI:
    st.error("Gemini API not configured or google-genai SDK missing. Add GEMINI_API_KEY and install google-genai.")
    st.stop()

# configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# -----------------------------
# Utility: sanitizer
# -----------------------------

def sanitize_generated_text(text: str) -> str:
    if not text:
        return ""
    t = text
    # Remove LaTeX envs and commands
    t = re.sub(r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}", "\n", t, flags=re.DOTALL)
    t = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", t)
    t = re.sub(r"\\[a-zA-Z]+", "", t)
    t = re.sub(r"\$[^$]*\$", "", t)
    # Remove markdown
    t = re.sub(r"[`*_#>]+", "", t)
    # Normalize blank lines
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t

# -----------------------------
# System rubric (Gemini + OpenRouter)
# -----------------------------
RUBRIC = (
    "You are a professional ARTICLE writer. Produce a rigorous, print-friendly article.\n"
    "Form: Title on first line; Abstract: paragraph; numbered sections. Full paragraphs. No LaTeX/Markdown/HTML.\n"
    "Method: triangulate claims (expert, history, human example). Include >=8 stats and >=6 citation placeholders [ref]. Add 2-3 'Quote:' lines.\n"
    "Required sections: 1. Introduction and Context; 2. Method and Sources; 3. Economic Footprint; 4. Worker Safety and Health; 5. Environmental Impacts and Compliance; 6. Counterpoints and Industry View; 7. Forward Scenarios 2030/2040; 8. Action Agenda; 9. Conclusion."
)

# -----------------------------
# Generator: Gemini primary
# -----------------------------

def generate_with_gemini(topic: str, context: str = "", temperature: float = 0.7) -> str:
    try:
        prompt = f"{RUBRIC}\n\nTopic: {topic}\nContext: {context}"
        model = genai.GenerativeModel("gemini-1.5-flash")
        gen_cfg = {"temperature": temperature, "max_output_tokens": 2048}
        resp = model.generate_content(prompt, generation_config=gen_cfg)
        # robust extraction
        text = (getattr(resp, "text", None) or "")
        if not text:
            # try candidates
            candidates = getattr(resp, "candidates", None)
            if candidates and len(candidates) > 0:
                text = getattr(candidates[0], "content", {}).get("parts", [None])[0] or ""
        return sanitize_generated_text(text.strip())
    except Exception as e:
        st.warning(f"Gemini generation error: {e}")
        return ""

# -----------------------------
# Generator: DeepSeek fallback via OpenRouter
# -----------------------------

def generate_with_deepseek(topic: str, context: str = "") -> str:
    if not HAS_OPENROUTER:
        return ""
    payload = {
        "model": "deepseek/deepseek-chat-v3-0324",
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"Topic: {topic}\nContext: {context}"},
        ],
    }
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    try:
        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=90)
        if resp.status_code == 200:
            raw = resp.json().get("choices", [])[0].get("message", {}).get("content", "")
            return sanitize_generated_text(raw)
        st.warning(f"DeepSeek error HTTP {resp.status_code}")
    except Exception as e:
        st.warning(f"DeepSeek request error: {e}")
    return ""

# -----------------------------
# Validation and repair
# -----------------------------

def validate_article(text: str) -> dict:
    lines = [l for l in text.split('\n') if l.strip()]
    has_abstract = any(l.strip().lower().startswith('abstract:') for l in lines[:12])
    section_lines = [l for l in lines if re.match(r'^\d+\.\s', l)]
    num_sections = len(section_lines)
    needed = ['Introduction', 'Method', 'Economic', 'Worker', 'Environmental', 'Counterpoints', 'Scenario', 'Action', 'Conclusion']
    missing_sections = [k for k in needed if not any(k.lower() in l.lower() for l in section_lines)]
    stats_count = len(re.findall(r'\d', text))
    quotes = len(re.findall(r'(^|\n)\s*Quote:', text))
    citations = len(re.findall(r'\[ref\]|\(\d{4}\)|\[[0-9]+\]', text))
    return {
        'has_abstract': has_abstract,
        'num_sections': num_sections,
        'missing_sections': missing_sections,
        'stats_count': stats_count,
        'quotes': quotes,
        'citations': citations,
        'passes': has_abstract and num_sections >= 5 and stats_count >= 8 and quotes >= 2 and citations >= 6 and len(missing_sections) <= 3,
    }


def repair_with_gemini(current_text: str, report: dict, topic: str, context: str) -> str:
    try:
        gaps = json.dumps(report, ensure_ascii=False)
        instr = (
            "Revise the ARTICLE to satisfy the rubric. Keep Title and Abstract. Ensure numbered sections, "
            "add missing sections or details, include >=8 statistics and >=6 citation placeholders [ref], and add >=2 'Quote:' lines. "
            "No LaTeX/Markdown/emojis/HTML."
        )
        prompt = f"{instr}\n\nTOPIC: {topic}\nCONTEXT: {context}\nGAPS: {gaps}\n\nCURRENT ARTICLE:\n{current_text}"
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(prompt, generation_config={"temperature": 0.5, "max_output_tokens": 2048})
        revised = (getattr(resp, "text", None) or "").strip()
        if not revised:
            candidates = getattr(resp, "candidates", None)
            if candidates and len(candidates) > 0:
                revised = getattr(candidates[0], "content", {}).get("parts", [None])[0] or ""
        return sanitize_generated_text(revised)
    except Exception as e:
        st.warning(f"Repair error: {e}")
        return current_text

# -----------------------------
# PDF generation (returns bytes)
# -----------------------------

def build_pdf_bytes(article_text: str, topic: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('title', parent=styles['Title'], fontSize=18, alignment=1, spaceAfter=20)
    section_style = ParagraphStyle('section', parent=styles['Heading2'], fontSize=14, spaceAfter=10)
    body_style = ParagraphStyle('body', parent=styles['Normal'], fontSize=12, spaceAfter=8)

    text = sanitize_generated_text(article_text)
    elements = [Paragraph(topic.strip(), title_style)]

    for raw in text.split('\n'):
        line = raw.strip()
        if not line:
            elements.append(Spacer(1, 0.18 * inch))
            continue
        if re.match(r'^\d+\.\s', line):
            elements.append(Paragraph(line, section_style))
        else:
            elements.append(Paragraph(line, body_style))

# --- Append References section if present ---
    if "References:" in article_text:
        _, refs = article_text.split("References:", 1)
        if refs.strip():
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("References", title_style))
            for ref in refs.strip().splitlines():
                if ref.strip():
                    elements.append(Paragraph(ref.strip(), body_style))

    doc.build(elements)
    buffer.seek(0)
    return buffer.read()



# -----------------------------
# Streamlit UI (simplified version)
# -----------------------------
st.title("1-Click Article Generator")
st.caption("Produce a timeless, print-friendly article. Gemini primary; DeepSeek fallback.")

# Inputs
topic = st.text_input("Article topic")
context = st.text_area("Additional context (optional)", height=160)

# Generate button logic
if st.button("Generate Article"):
    if not topic.strip():
        st.error("Topic is required.")
        st.stop()
    
    with st.spinner("Generating article…"):
        article_text = generate_with_gemini(topic, context)
        if not article_text and HAS_OPENROUTER:
            article_text = generate_with_deepseek(topic, context)
    
    if not article_text:
        st.error("Generation failed.")
        st.stop()
    
    # PDF export
    pdf_bytes = build_pdf_bytes(article_text, topic)
    filename = f"article_{re.sub(r'[^a-zA-Z0-9]+', '_', topic).strip('_')}.pdf"
    st.download_button("📄 Download PDF", data=pdf_bytes, file_name=filename, mime="application/pdf")
    
    # Show article directly
    st.subheader("Generated Article")
    st.markdown(article_text)

# Footer
st.caption("Verify all citations and facts before publishing.")