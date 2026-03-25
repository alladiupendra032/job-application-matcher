import gradio as gr
from huggingface_hub import InferenceClient
import PyPDF2

def extract_text(file):
    """
    Extracts text from an uploaded PDF file.
    """
    if file is None:
        raise ValueError("File is not uploaded.")
    try:
        text = ""
        # The file object from Gradio has a 'name' attribute containing the temp filepath
        with open(file.name, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    
        if not text.strip():
            raise ValueError("Could not extract any text from the PDF.")
            
        return text.strip()
    except Exception as e:
        raise ValueError(f"PDF text extraction failed: {str(e)}")

def compute_similarity(api_key, resume_text, job_desc):
    """
    Computes sentence similarity between resume text and job description
    using Hugging Face InferenceClient.
    """
    if not api_key.strip():
        raise ValueError("API key is missing.")
    if not job_desc.strip():
        raise ValueError("Job description is empty.")
    
    try:
        # Initialize client using the provided API key
        client = InferenceClient(
            provider="hf-inference",
            api_key=api_key
        )
        
        # Call the sentence similarity API
        result = client.sentence_similarity(
            job_desc,
            [resume_text],
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Result is typically a list of scores corresponding to each sentence
        if isinstance(result, list) and len(result) > 0:
            score = result[0]
        else:
            raise ValueError("Unexpected response format from API.")
            
        # Convert score to percentage
        percentage = round(score * 100, 2)
        return percentage
        
    except Exception as e:
        raise ValueError(f"API call failed: {str(e)}")

def analyze_skills(api_key, resume_text, job_desc):
    """
    Analyzes the resume and job description to extract matching skills,
    missing skills, and improvement advice using an LLM.
    """
    try:
        # Initialize client using the provided API key (token parameter is used for backward compatibility)
        # Avoid provider="hf-inference" as it uses the strict router API which causes 404s for non-partner models
        client = InferenceClient(token=api_key)
        
        messages = [
            {"role": "system", "content": "You are a professional HR assistant. Help the user compare the resume to the job description."},
            {"role": "user", "content": f"Compare this resume and job description.\n\nResume:\n{resume_text[:1500]}\n\nJob description:\n{job_desc[:1500]}\n\nProvide:\n1. What skills match\n2. What skills are missing\n3. Short improvement advice"}
        ]
        
        # Qwen2.5 72B Instruct is currently the default and fully supported open model on HF Serverless
        response = client.chat_completion(
            messages=messages,
            model="Qwen/Qwen2.5-72B-Instruct",
            max_tokens=350
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Could not generate analysis: {str(e)}"

def process(api_key, file, job_desc):
    """
    Main pipeline function to process the inputs and return the output.
    """
    try:
        # 1. Error handling for inputs
        if not api_key.strip():
            return "❌ Error: API key is missing.", ""
        if file is None:
            return "❌ Error: File is not uploaded.", ""
        if not job_desc.strip():
            return "❌ Error: Job description is empty.", ""

        # 2. Extract text from PDF
        try:
            resume_text = extract_text(file)
        except ValueError as e:
            return f"❌ Error: {str(e)}", ""

        # 3. Compute similarity
        try:
            score = compute_similarity(api_key, resume_text, job_desc)
        except ValueError as e:
            return f"❌ Error: {str(e)}", ""

        # 4. Analyze skills
        analysis_text = analyze_skills(api_key, resume_text, job_desc)

        # 5. Return final formatted output
        return f"✅ Match Score: {score}%", analysis_text
        
    except Exception as e:
        return f"❌ An unexpected error occurred: {str(e)}", ""

# Define the Gradio Interface using Blocks
with gr.Blocks(theme=gr.themes.Soft(), title="job application matcher") as app:
    # Title and description using Markdown
    gr.Markdown("# 📄 Resume & Job Description Matcher 🎯")
    gr.Markdown(
        "Upload your resume as a PDF and paste the job description below. "
        "We'll use Hugging Face's `all-MiniLM-L6-v2` model to analyze how well your "
        "resume matches the job description!"
    )
    
    # Layout with rows and columns
    with gr.Row():
        with gr.Column(scale=1):
            # API Key Input
            api_key_input = gr.Textbox(
                label="🔑 Hugging Face API Key", 
                type="password", 
                placeholder="Enter your Hugging Face API key here..."
            )
            
            # File Upload For Resume
            resume_input = gr.File(
                label="📁 Upload Resume (PDF only)", 
                file_types=[".pdf"]
            )
            
            # Job Description Input
            job_desc_input = gr.Textbox(
                label="📝 Job Description", 
                lines=7, 
                placeholder="Paste the target job description here..."
            )
            
            # Match Button
            match_btn = gr.Button("🚀 Match Resume", variant="primary")
            
        with gr.Column(scale=1):
            # Output Display
            output_box = gr.Textbox(
                label="📊 Match Result",
                lines=2
            )
            
            # Analysis Display
            analysis_box = gr.Textbox(
                label="💡 Analysis (Matches, Missing & Advice)",
                lines=8
            )
            
    # Connect button click to process function
    match_btn.click(
        fn=process,
        inputs=[api_key_input, resume_input, job_desc_input],
        outputs=[output_box, analysis_box]
    )

if __name__ == "__main__":
    # Launching on a new port to bypass your older frozen server instances
    app.launch(server_port=7865, share=True, ssr_mode=False)