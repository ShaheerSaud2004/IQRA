#!/usr/bin/env python3
"""
Simple test script to verify PDF text extraction from IQRA Constitution
"""

import PyPDF2
import os

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            print(f"PDF has {len(pdf_reader.pages)} pages")
            
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += page_text + "\n"
                print(f"Page {i+1}: {len(page_text)} characters extracted")
                
                # Show first few lines of first page as sample
                if i == 0:
                    lines = page_text.split('\n')[:5]
                    print(f"Sample content from page 1:")
                    for line in lines:
                        if line.strip():
                            print(f"  {line.strip()}")
                    
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return None
    return text

def main():
    print("🔍 Testing PDF Text Extraction")
    print("=" * 50)
    
    pdf_path = "IQRAConstitution (1).pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        print("Please make sure the file is in the current directory")
        return
    
    print(f"📄 Found PDF: {pdf_path}")
    file_size = os.path.getsize(pdf_path) / 1024  # KB
    print(f"📊 File size: {file_size:.1f} KB")
    
    print("\n🔄 Extracting text...")
    text = extract_text_from_pdf(pdf_path)
    
    if text:
        print(f"\n✅ Success!")
        print(f"📝 Total text length: {len(text)} characters")
        print(f"📄 Word count (approx): {len(text.split())}")
        
        # Show some statistics
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        print(f"📊 Lines: {len(lines)} total, {len(non_empty_lines)} non-empty")
        
        # Show first few sentences
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        print(f"\n📖 First few meaningful sentences:")
        for i, sentence in enumerate(sentences[:3]):
            print(f"  {i+1}. {sentence}.")
            
        print(f"\n🎯 PDF extraction test completed successfully!")
        print(f"The system should be able to process this document.")
        
    else:
        print("❌ Failed to extract text from PDF")
        print("Please check if the PDF is corrupted or password-protected")

if __name__ == "__main__":
    main() 