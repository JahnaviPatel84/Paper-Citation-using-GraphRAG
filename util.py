import re
import fitz

def load_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text("text")  # Extract clean text
        cleaned_text = preprocess_text(text)
        return cleaned_text
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

# Preprocess extracted text (removes unwanted characters, special tokens, etc.)
def preprocess_text(text):
    # Remove special tokens like <EOS>, <pad>, etc.
    text = re.sub(r'<\s*EOS\s*>', '', text)  # Remove <EOS> tokens
    text = re.sub(r'<\s*pad\s*>', '', text)  # Remove <pad> tokens
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = text.strip()  # Strip leading and trailing spaces
    return text

# 1. Source Documents → Text Chunks
def split_documents_into_chunks(document, chunk_size=3500, overlap_size=300):
    chunks = []
    # Split the document into chunks based on chunk size and overlap
    for i in range(0, len(document), chunk_size - overlap_size):
        chunk = document[i:i + chunk_size]
        # Check if chunk is too short, if so, join with the previous chunk
        if len(chunk) < chunk_size:
            if chunks:
                chunks[-1] += " " + chunk
            else:
                chunks.append(chunk)
        else:
            chunks.append(chunk)
    return chunks