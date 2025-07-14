
# Text summarization of a PDF using Gemma2 model from groqAPI

from PyPDF2 import PdfReader

# 1. Read and extract text from PDF
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text



pdf_path = "Keerthana_DataScientist-Resume.pdf"
text = extract_text_from_pdf(pdf_path)



# 2: Use Groq API with Gemma 2 to summarize
import openai
import os

from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = "----"

client = openai.OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"  # Use Groq-compatible endpoint
)

#  Call Gemma model for summarization
response = client.chat.completions.create(
    model="Gemma2-9b-It",
    messages=[
        {"role": "system", "content": "You are a helpful assistant that summarizes documents."},
        {"role": "user", "content": f"Summarize the following text:\n\n{text[:3000]}"}
    ],
    temperature=0.5
)

# 3. Output the summary
summary = response.choices[0].message.content
print("📄 Summary:\n", summary)


# Evaluation of summary using BERT and COSINE SIMILARITY


import openai
from bert_score import score

# Evaluate using BERTScore
P, R, F1 = score([summary], [text], lang="en", verbose=True)

# Print average BERTScore metrics
print(f"\nBERTScore Evaluation:")
print(f"Precision: {P[0]:.4f}")
print(f"Recall:    {R[0]:.4f}")
print(f"F1 Score:  {F1[0]:.4f}")

# Evaluation using sentence embedding -cosine simialrity

import openai
from sentence_transformers import SentenceTransformer, util

#Load sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, accurate for semantic similarity

#  Convert document and summary to embeddings
doc_embedding = model.encode(text, convert_to_tensor=True)
summary_embedding = model.encode(summary, convert_to_tensor=True)

#  Compute cosine similarity
cosine_sim = util.pytorch_cos_sim(summary_embedding, doc_embedding).item()

print(f"\n✅ Cosine Similarity Score: {cosine_sim:.4f} (range: 0–1)")

