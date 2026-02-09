"""
COREP Own Funds Reporting Assistant using Pydantic AI
Extracts text from PDFs and stores in vector DB for retrieval
"""

import pdfplumber
import os
from typing import List, Dict
from pathlib import Path

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
    except Exception as e:
        print(f"Error extracting {pdf_path}: {e}")
    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Chunk text into smaller pieces with overlap."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to end at a sentence or paragraph boundary
        if end < len(text):
            # Look for period, newline, or space
            for delimiter in ['.\n', '. ', '\n\n', '\n', ' ']:
                last_delim = chunk.rfind(delimiter)
                if last_delim > chunk_size * 0.5:  # At least 50% of chunk
                    end = start + last_delim + len(delimiter)
                    chunk = text[start:end]
                    break
        
        chunks.append({
            'text': chunk.strip(),
            'start': start,
            'end': end
        })
        
        start = end - overlap
    
    return chunks

def process_pdfs(pdf_paths: List[str], output_dir: str = "data/processed"):
    """Process all PDFs and save extracted chunks."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_chunks = []
    
    for pdf_path in pdf_paths:
        print(f"Processing {pdf_path}...")
        filename = Path(pdf_path).stem
        
        # Extract text
        text = extract_text_from_pdf(pdf_path)
        
        # Save full text
        with open(f"{output_dir}/{filename}.txt", 'w') as f:
            f.write(text)
        
        # Create chunks
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            chunk['source'] = filename
            chunk['chunk_id'] = f"{filename}_{i}"
            all_chunks.append(chunk)
        
        print(f"  Extracted {len(chunks)} chunks from {filename}")
    
    return all_chunks

if __name__ == "__main__":
    pdf_files = [
        "/home/sohanx1/Downloads/AKION/corep-own-funds-instructions.pdf",
        "/home/sohanx1/Downloads/AKION/Own Funds (CRR)_08-02-2026.pdf",
        "/home/sohanx1/Downloads/AKION/Reporting (CRR)_08-02-2026.pdf",
        "/home/sohanx1/Downloads/AKION/ss3415-october-2025.pdf"
    ]
    
    chunks = process_pdfs(pdf_files)
    print(f"\nTotal chunks: {len(chunks)}")
