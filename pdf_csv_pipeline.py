import os
import pdfplumber
import pandas as pd
import hashlib
import re

class PDFPipeline:
    def __init__(self, pdf_directory, output_csv, clean_csv):
        """
        Initializes the PDFPipeline with the directory of PDFs and output file paths.

        Parameters:
        - pdf_directory (str): Path to the directory containing PDF files.
        - output_csv (str): Path to the CSV file where extracted data will be saved.
        - clean_csv (str): Path to the CSV file where cleaned data will be saved.
        """
        self.pdf_directory = pdf_directory
        self.output_csv = output_csv
        self.clean_csv = clean_csv

    def extract_text_from_pdf(self, pdf_path):
        """Extracts text from a PDF file using pdfplumber."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ''
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + '\n'
                return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return None

    def compute_file_hash(self, pdf_path):
        """Computes the MD5 hash of a PDF file."""
        hash_md5 = hashlib.md5()
        try:
            with open(pdf_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"Error computing hash for {pdf_path}: {e}")
            return None

    def process_pdfs_in_directory(self):
        """Processes all PDFs in the specified directory and extracts text and hash."""
        pdf_data = []
        
        for filename in os.listdir(self.pdf_directory):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(self.pdf_directory, filename)
                text = self.extract_text_from_pdf(pdf_path)
                file_hash = self.compute_file_hash(pdf_path)
                
                pdf_data.append({
                    'filename': filename,
                    'text': text,
                    'hash': file_hash
                })

        return pd.DataFrame(pdf_data)

    def clean_text(self, text):
        """Cleans the text by removing line breaks, unicode characters, and repeated symbols."""
        if pd.isna(text):
            return ''
        text = text.replace('\n', ' ')
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = re.sub(r'(.)\1+', r'\1', text)
        return text

    def clean_data(self, df):
        """Applies text cleaning to the DataFrame."""
        df['text'] = df['text'].apply(self.clean_text)
        return df

    def run(self):
        """Runs the complete PDF processing pipeline."""
        # step 1: Process PDFs and save to CSV
        pdf_df = self.process_pdfs_in_directory()
        pdf_df.to_csv(self.output_csv, index=False, sep='|', quoting=1, escapechar='\\')
        print(f"PDF data extracted and saved to {self.output_csv}")

        # step 2: Clean the text data and save to a new CSV
        clean_df = self.clean_data(pdf_df)
        clean_df.to_csv(self.clean_csv, index=False, sep='|')
        print(f"Cleaned data saved to {self.clean_csv}")


if __name__ == "__main__":
    pdf_directory = './pdf_files'
    output_csv = 'pdf_data.csv'
    clean_csv = 'pdf_clean_data.csv'

    pipeline = PDFPipeline(pdf_directory, output_csv, clean_csv)
    pipeline.run()
