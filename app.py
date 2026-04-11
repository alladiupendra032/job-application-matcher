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


with gr.Blocks(title="Job application matcher") as demo:
    gr.Markdown("# Resume and job description matcher")
    gr.Markdown(
        "Upload your resume (PDF) and paste the job description. "
        "Matching uses Hugging Face Inference API (`all-MiniLM-L6-v2`). "
        "Enter your API key when you run a match — nothing is loaded until you click **Match resume**."
    )

    with gr.Row():
        with gr.Column(scale=1):
            api_key_input = gr.Textbox(
                label="Hugging Face API key",
                type="password",
                placeholder="hf_...",
            )
            resume_input = gr.File(label="Resume (PDF)", file_types=[".pdf"])
            job_desc_input = gr.Textbox(
                label="Job description",
                lines=7,
                placeholder="Paste the job description here...",
            )
            match_btn = gr.Button("Match resume", variant="primary")

        with gr.Column(scale=1):
            output_box = gr.Textbox(label="Match result", lines=2)
            analysis_box = gr.Textbox(
                label="Analysis (matches, gaps, advice)",
                lines=10,
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
