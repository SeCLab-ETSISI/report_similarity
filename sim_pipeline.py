import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from scipy.sparse import csr_matrix

nltk.download('punkt')

class SimilarityPipeline:
    def __init__(self, reports_path: str, phrases_path: str, report_identifier: str = 'hash', phrase_field: str = 'text', threshold: float = 0.75, doc_title_field: str = 'doc_title'):
        """
        Initializes the Pipeline with paths to the report CSV and phrases JSON/CSV files.

        Parameters:
        - reports_path (str): Path to the CSV file containing reports.
        - phrases_path (str): Path to the JSON/CSV file containing phrases.
        - report_identifier (str): Column name in the reports CSV that serves as an identifier. Default is 'hash'.
        - phrase_field (str): Field name in the JSON/CSV that contains the phrases. Default is 'text'.
        - threshold (float): Cosine similarity threshold to consider a match. Default is 0.75.
        - doc_title_field (str): Field name in the JSON/CSV that contains the document titles. Default is 'doc_title'.
        """
        self.reports_path = reports_path
        self.phrases_path = phrases_path
        self.report_identifier = report_identifier
        self.phrase_field = phrase_field
        self.threshold = threshold
        self.doc_title_field = doc_title_field
        self.reports_df = None
        self.phrases_df = None
        self.results_df = None

    def load_data(self) -> None:
        """Loads the reports CSV and phrases JSON/CSV into dataframes."""
        print(f"[📂] Loading reports from {self.reports_path}")
        self.reports_df = pd.read_csv(self.reports_path, sep='|')

        print(f"[📂] Loading phrases from {self.phrases_path}")
        if self.phrases_path.endswith('.json'):
            with open(self.phrases_path, 'r') as file:
                phrases_data = json.load(file)
            self.phrases_df = pd.DataFrame(phrases_data)
        elif self.phrases_path.endswith('.csv'):
            self.phrases_df = pd.read_csv(self.phrases_path)
        else:
            raise ValueError("Unsupported file format. Please provide a JSON or CSV file.")

    def preprocess_reports(self) -> None:
        """Tokenizes and preprocesses the text field in the reports dataframe."""
        print("[🗃️] Tokenizing and preprocessing reports")
        self.reports_df['sentences'] = self.reports_df['text'].apply(lambda x: sent_tokenize(str(x).lower()))

    def flatten_sentences(self) -> (list, list):
        """Flattens the list of sentences in the reports and keeps the associated hash.

        Returns:
        - sentences (list): List of sentences.
        - hashes (list): List of corresponding report hashes.
        """
        print("[+] Flattening sentences list")
        sentences, hashes = [], []
        for idx, row in self.reports_df.iterrows():
            for sentence in row['sentences']:
                sentences.append(sentence)
                hashes.append(row[self.report_identifier])
        return sentences, hashes

    def vectorize_and_calculate_similarity(self, sentences: list) -> csr_matrix:
        """Vectorizes the sentences and phrases, and calculates cosine similarities.

        Parameters:
        - sentences (list): List of sentences to vectorize.

        Returns:
        - sparse_cosine_similarities (csr_matrix): Sparse cosine similarity matrix.
        """
        print("[+] Vectorizing sentences and phrases")
        vectorizer = TfidfVectorizer().fit(sentences + self.phrases_df[self.phrase_field].tolist())
        tfidf_sentences = vectorizer.transform(sentences)
        tfidf_phrases = vectorizer.transform(self.phrases_df[self.phrase_field])
        
        print("[🧮] Calculating 'global' cosine similarity")
        sparse_cosine_similarities = cosine_similarity(tfidf_phrases, tfidf_sentences, dense_output=False)
        
        print(f"[+] Cosine similarities shape: {sparse_cosine_similarities.shape}")
        return sparse_cosine_similarities

    def filter_results(self, sentences: list, hashes: list, cosine_similarities: csr_matrix) -> None:
        """Filters the results based on the similarity threshold and constructs a dataframe.

        Parameters:
        - sentences (list): List of sentences.
        - hashes (list): List of report hashes.
        - cosine_similarities (csr_matrix): Sparse cosine similarity matrix.
        """
        print("[📊] Filtering results")
        results = []
        for idx_phrase, similarities in enumerate(cosine_similarities):
            similar_indices = similarities.indices[similarities.data >= self.threshold]
            print(f"\t╰─Phrase {idx_phrase} has {len(similar_indices)} similar sentences")
            for idx_sentence in similar_indices:
                result = {
                    'phrase': self.phrases_df.loc[idx_phrase, self.phrase_field],
                    'found_report_hash': hashes[idx_sentence],
                    'similarity': similarities[0, idx_sentence]
                }
                if self.doc_title_field in self.phrases_df.columns:
                    result['phrase_report_title'] = self.phrases_df.loc[idx_phrase, self.doc_title_field]
                results.append(result)
        self.results_df = pd.DataFrame(results).drop_duplicates()
    
    def save_full_report_text(self, output_file: str) -> None:
        """Merges the results with the full report texts and saves to a CSV file.

        Parameters:
        - output_file (str): Path to save the CSV file with full report text results.
        """
        merged_df = pd.merge(self.results_df, self.reports_df, left_on='found_report_hash', right_on=self.report_identifier, how='left')
        output_df = merged_df[['found_report_hash', 'phrase', 'similarity', 'text']]
        if self.doc_title_field in merged_df.columns:
            output_df['phrase_report_title'] = merged_df[self.doc_title_field]
        output_df.columns = ['hash', 'phrase', 'similarity', 'full_report_text', 'phrase_report_title']
        print(f"[💾] Saving full report text results to {output_file}")
        output_df.to_csv(output_file, index=False)
    
    def save_adjacent_sentences(self, output_file: str, N: int) -> None:
        """Saves N adjacent sentences around the phrases to a CSV/JSON file.

        Parameters:
        - output_file (str): Path to save the CSV/JSON file with adjacent sentences results.
        - N (int): Number of adjacent sentences to include.
        """
        def get_adjacent_sentences(text: str, phrase: str, N: int) -> list:
            sentences = sent_tokenize(text)
            phrase_index = next((i for i, s in enumerate(sentences) if phrase in s), None)
            if phrase_index is None:
                return []
            start_index = max(0, phrase_index - N)
            end_index = min(len(sentences), phrase_index + N + 1)
            return sentences[start_index:end_index]
        
        merged_df = pd.merge(self.results_df, self.reports_df, left_on='found_report_hash', right_on=self.report_identifier, how='left')
        output_data = []
        for _, row in merged_df.iterrows():
            adjacent_sentences = get_adjacent_sentences(row['text'], row['phrase'], N)
            result = {
                'hash': row['found_report_hash'],
                'phrase': row['phrase'],
                'similarity': row['similarity'],
                'adjacent_sentences': adjacent_sentences
            }
            if self.doc_title_field in row:
                result['phrase_report_title'] = row[self.doc_title_field]
            output_data.append(result)

        if output_file.endswith('.json'):
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=4)
        else:
            output_df = pd.DataFrame(output_data)
            output_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    def run(self, output_file: str, N: int = 0) -> None:
        """Runs the entire pipeline.

        Parameters:
        - output_file (str): Path to save the results.
        - N (int): Number of adjacent sentences to include (0 for full reports).
        """
        self.load_data()
        self.preprocess_reports()
        sentences, hashes = self.flatten_sentences()
        cosine_similarities = self.vectorize_and_calculate_similarity(sentences)
        self.filter_results(sentences, hashes, cosine_similarities)
        
        if N == 0:
            self.save_full_report_text(output_file)
        else:
            self.save_adjacent_sentences(output_file, N)
        print("DONE!")

pipeline = SimilarityPipeline(
    reports_path='./data/pdf_clean_data.csv',             # file with the full reports
    phrases_path='./data/cti_to_mitre_test_dataset.csv',  # file with the sentences
    report_identifier='hash',
    phrase_field='sentence',                              # field for sentences in the sentences file
    threshold=0.75,
    doc_title_field='doc_title'                           # field for document titles
)

pipeline.run(
    output_file='./final_results/N2_adjacent_results.csv',
    N=2
)
