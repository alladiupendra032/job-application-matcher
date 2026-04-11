import gradio as gr
from huggingface_hub import InferenceClient
import pypdf


def extract_text(file):
    """Extract text from an uploaded PDF (runs only when the user submits)."""
    if file is None:
        raise ValueError("File is not uploaded.")
    try:
        text = ""
        path = getattr(file, "name", None) or str(file)
        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if not text.strip():
            raise ValueError("Could not extract any text from the PDF.")

        return text.strip()
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"PDF text extraction failed: {str(e)}") from e


def compute_similarity(api_key, resume_text, job_desc):
    """Sentence similarity via HF Inference API (lazy: client created per request)."""
    if not api_key.strip():
        raise ValueError("API key is missing.")
    if not job_desc.strip():
        raise ValueError("Job description is empty.")

    try:
        client = InferenceClient(
            provider="hf-inference",
            api_key=api_key.strip(),
        )

        result = client.sentence_similarity(
            job_desc,
            [resume_text],
            model="sentence-transformers/all-MiniLM-L6-v2",
        )

        if isinstance(result, list) and len(result) > 0:
            score = result[0]
        else:
            raise ValueError("Unexpected response format from API.")

        return round(float(score) * 100, 2)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"API call failed: {str(e)}") from e


def analyze_skills(api_key, resume_text, job_desc):
    """LLM comparison (lazy: client and request only on submit; smaller model for faster responses)."""
    try:
        client = InferenceClient(token=api_key.strip())

        messages = [
            {
                "role": "system",
                "content": "You are a professional HR assistant. Help the user compare the resume to the job description.",
            },
            {
                "role": "user",
                "content": (
                    f"Compare this resume and job description.\n\nResume:\n{resume_text[:1500]}\n\n"
                    f"Job description:\n{job_desc[:1500]}\n\nProvide:\n"
                    "1. What skills match\n2. What skills are missing\n3. Short improvement advice"
                ),
            },
        ]

        response = client.chat_completion(
            messages=messages,
            model="Qwen/Qwen2.5-7B-Instruct",
            max_tokens=350,
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate analysis: {str(e)}"


def process(api_key, file, job_desc):
    """Main pipeline; all API work happens here, not at import or page load."""
    try:
        if not (api_key or "").strip():
            return "❌ Error: API key is missing.", ""
        if file is None:
            return "❌ Error: File is not uploaded.", ""
        if not (job_desc or "").strip():
            return "❌ Error: Job description is empty.", ""

        try:
            resume_text = extract_text(file)
        except ValueError as e:
            return f"❌ Error: {str(e)}", ""

        try:
            score = compute_similarity(api_key, resume_text, job_desc)
        except ValueError as e:
            return f"❌ Error: {str(e)}", ""

        analysis_text = analyze_skills(api_key, resume_text, job_desc)

        return f"✅ Match Score: {score}%", analysis_text
    except Exception as e:
        return f"❌ An unexpected error occurred: {str(e)}", ""


_UI_CSS = """
.gradio-container { max-width: 1120px !important; margin: auto !important; }
.hero-wrap {
  border-radius: 18px;
  padding: 1.35rem 1.5rem 1.5rem;
  margin-bottom: 1.25rem;
  background: linear-gradient(125deg, #4f46e5 0%, #7c3aed 42%, #0ea5e9 100%);
  box-shadow: 0 12px 40px rgba(79, 70, 229, 0.35);
  color: #f8fafc;
}
.hero-wrap h1 { margin: 0; font-size: 1.65rem; font-weight: 700; letter-spacing: -0.02em; }
.hero-wrap p { margin: 0.55rem 0 0; font-size: 0.98rem; line-height: 1.5; opacity: 0.95; }
.hero-wrap code { background: rgba(15,23,42,0.35); padding: 0.12rem 0.4rem; border-radius: 6px; font-size: 0.88em; }
.panel-in {
  border-radius: 14px !important;
  padding: 0.35rem !important;
  background: linear-gradient(145deg, rgba(99,102,241,0.12), rgba(14,165,233,0.08)) !important;
}
.panel-out {
  border-radius: 14px !important;
  padding: 0.35rem !important;
  background: linear-gradient(145deg, rgba(16,185,129,0.1), rgba(99,102,241,0.06)) !important;
}
.match-btn-wrap button {
  font-weight: 600 !important;
  font-size: 1.02rem !important;
  padding: 0.75rem 1rem !important;
  border-radius: 12px !important;
  background: linear-gradient(90deg, #4f46e5, #7c3aed, #0ea5e9) !important;
  border: none !important;
  box-shadow: 0 8px 24px rgba(79, 70, 229, 0.4) !important;
}
.match-btn-wrap button:hover { filter: brightness(1.08); transform: translateY(-1px); }
"""

_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("DM Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill_dark="#0f172a",
    block_background_fill_dark="#1e293b",
    block_border_width="1px",
    block_label_text_size="sm",
    input_background_fill_dark="#334155",
    button_primary_background_fill="linear-gradient(90deg, *primary_500, *secondary_500)",
)

with gr.Blocks(
    title="Job application matcher",
    theme=_theme,
    css=_UI_CSS,
) as demo:
    gr.HTML(
        """
<div class="hero-wrap">
  <h1>📄 Resume &amp; job description matcher 🎯</h1>
  <p>
    Upload your <strong>PDF resume</strong>, paste the <strong>job description</strong>, then hit
    <strong>Match resume</strong>. Scoring uses Hugging Face Inference with
    <code>sentence-transformers/all-MiniLM-L6-v2</code> — your API key is only used when you submit.
  </p>
</div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, elem_classes=["panel-in"]):
            api_key_input = gr.Textbox(
                label="🔑 Hugging Face API key",
                type="password",
                placeholder="hf_...",
                info="Create a token under Settings → Access tokens (read is enough for many models).",
            )
            resume_input = gr.File(
                label="📁 Upload resume (PDF)",
                file_types=[".pdf"],
            )
            job_desc_input = gr.Textbox(
                label="📝 Job description",
                lines=8,
                placeholder="Paste the full job description here (role, requirements, nice-to-haves)...",
            )
            with gr.Row(elem_classes=["match-btn-wrap"]):
                match_btn = gr.Button("🚀 Match resume", variant="primary", scale=1)

        with gr.Column(scale=1, elem_classes=["panel-out"]):
            output_box = gr.Textbox(
                label="📊 Match result",
                lines=3,
                placeholder="Your match score will show here…",
            )
            analysis_box = gr.Textbox(
                label="💡 Analysis — matches, gaps & advice",
                lines=14,
                placeholder="Structured tips from the model will appear here…",
            )

    match_btn.click(
        fn=process,
        inputs=[api_key_input, resume_input, job_desc_input],
        outputs=[output_box, analysis_box],
    )

# Root URL "/" serves the Gradio UI (sufficient for Space health checks once the server is listening).
demo.queue()

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
    )
