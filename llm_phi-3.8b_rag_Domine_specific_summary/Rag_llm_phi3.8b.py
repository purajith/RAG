
import os
import streamlit as st
import pdfplumber
import docx
import replicate
import tiktoken  # OpenAI tokenizer

# --------------------
# CONFIG
# --------------------
REPLICATE_API_TOKEN = ""
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
MAX_TOTAL_TOKENS = 4000
DESIRED_OUTPUT_TOKENS = 512
INPUT_LIMIT = MAX_TOTAL_TOKENS - DESIRED_OUTPUT_TOKENS

SUMMARY_PROMPT = "Summarize in two lines with heading of summary"

# OpenAI tokenizer (used for both count & chunk)
enc = tiktoken.encoding_for_model("gpt-4o-mini")  # same as OpenAI's method

# --------------------
# PROMPTS
# --------------------
def COMPLIANCE_PROMPT(query, document_text):
    return f"""<|system|>
    You are an expert-level AI compliance auditor operating in strict audit mode.
    You must determine whether the requirement is **fully**, **partially**, or **not at all** met based strictly on what is explicitly stated in the document.

    --- Evaluation Rules ---
    - ✅ **Compliant**: All required controls explicitly stated, mandatory, enforceable
    - ⚠️ **Partially Compliant**: Some controls stated, minor gaps
    - ❌ **Not Compliant**: Missing major controls, vague language, or uncertainty
    <|end|>
    <|user|>
    Analyze the following document and answer:
    "{query}"

    Respond in this format:
    Compliance Status: ✅ Compliant / ⚠️ Partially Compliant / ❌ Not Compliant
    Explanation: [Why it's compliant or not]
    Recommendation: [What can be improved]

    Document Content:
    {document_text}
    <|end|>
    <|assistant|>
    """

# --------------------
# HELPERS
# --------------------
def extract_text(file_path):
    """Extract text from PDF or DOCX."""
    if file_path.endswith(".pdf"):
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    elif file_path.endswith(".docx"):
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format. Only .pdf and .docx are supported.")

def openai_token_count(text):
    """Count tokens using OpenAI tokenizer."""
    return len(enc.encode(text))

def chunk_text(text, max_tokens):
    """Chunk text into segments using OpenAI tokenizer."""
    tokens = enc.encode(text)
    return [
        enc.decode(tokens[i:i+max_tokens])
        for i in range(0, len(tokens), max_tokens)
    ]

def llm_model(context_text, max_tokens, system_prompt):
    """Run Phi-3 model via Replicate."""
    output = replicate.run(
        "microsoft/phi-3-mini-4k-instruct:e17386e6ae2e351f63783fa89f427fd0ed415524a7b3d8c122f6ac80ad0166b1",
        input={
            "prompt": context_text,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "system_prompt": system_prompt
        }
    )

    if hasattr(output, "__iter__") and not isinstance(output, (str, list)):
        output = "".join(list(output))
    if isinstance(output, list):
        return "".join(output)
    return str(output)

def process_document(text, query=None, mode="summary"):
    """Process doc with summary or compliance check."""
    token_used = openai_token_count(text)
    st.write(f"🔹 **OpenAI Token Count (Estimate)**: {token_used}")

    if token_used <= INPUT_LIMIT:
        if mode == "summary":
            return llm_model(text, DESIRED_OUTPUT_TOKENS, SUMMARY_PROMPT)
        else:
            prompt_sum = COMPLIANCE_PROMPT(query, text)
            return llm_model(prompt_sum, DESIRED_OUTPUT_TOKENS, "Compliance Check")

    # If too large, split into chunks
    chunks = chunk_text(text, INPUT_LIMIT)
    summaries = [llm_model(chunk, DESIRED_OUTPUT_TOKENS, SUMMARY_PROMPT) for chunk in chunks]
    merged_summary = "\n".join(summaries)

    if mode == "summary":
        return llm_model(merged_summary, DESIRED_OUTPUT_TOKENS, SUMMARY_PROMPT)
    else:
        merged_summary = COMPLIANCE_PROMPT(query, merged_summary)
        return llm_model(merged_summary, DESIRED_OUTPUT_TOKENS, "Compliance Check")

# --------------------
# STREAMLIT UI
# --------------------
st.set_page_config(page_title="RAG Compliance & Summary (Phi-3 + OpenAI Token Count)", layout="centered")
st.title("📄 RAG Compliance & Summary Tool (Phi-3 + OpenAI Token Count)")

uploaded_file = st.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx"])

if uploaded_file:
    file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    query = st.text_input("Enter your compliance query:")

    col1, col2 = st.columns(2)

    if col1.button("🔍 Compliance RAG"):
        if query.strip():
            with st.spinner("Processing compliance check..."):
                text = extract_text(file_path)
                result = process_document(text, query=query, mode="compliance")
                st.subheader("Compliance Result")
                st.write(result)
        else:
            st.warning("Please enter a compliance query.")

    if col2.button("📝 Summary"):
        with st.spinner("Generating summary..."):
            text = extract_text(file_path)
            result = process_document(text, mode="summary")
            st.subheader("Summary")
            st.write(result)
else:
    st.info("Please upload a document to begin.")
