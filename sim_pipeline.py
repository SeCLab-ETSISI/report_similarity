import pandas as pd
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

class SimilarityPipeline:
    def __init__(self, reports_path, phrases_path, report_identifier='hash', phrase_field='text', threshold=0.75):
        """
        Initializes the Pipeline with paths to the report CSV and phrases JSON files.

        Parameters:
        - reports_path (str): Path to the CSV file containing reports.
        - phrases_path (str): Path to the JSON file containing phrases.
        - report_identifier (str): Column name in the reports CSV that serves as an identifier. Default is 'hash'.
        - phrase_field (str): Field name in the JSON that contains the phrases. Default is 'text'.
        - threshold (float): Cosine similarity threshold to consider a match. Default is 0.75.
        """
        self.reports_path = reports_path
        self.phrases_path = phrases_path
        self.report_identifier = report_identifier
        self.phrase_field = phrase_field
        self.threshold = threshold
        self.reports_df = None
        self.phrases_df = None
        self.results_df = None

    def load_data(self):
        """Loads the reports CSV and phrases JSON into dataframes."""
        print(f"[📂] Loading reports from {self.reports_path}")
        self.reports_df = pd.read_csv(self.reports_path, sep='|')  #! separator may change

        print(f"[📂] Loading phrases from {self.phrases_path}")
        with open(self.phrases_path, 'r') as file:
            phrases_data = json.load(file)
        self.phrases_df = pd.DataFrame(phrases_data)

    def preprocess_reports(self):
        """Tokenizes and preprocesses the text field in the reports dataframe."""
        print("[🗃️] Tokenizing and preprocessing reports")
        self.reports_df['sentences'] = self.reports_df['text'].apply(lambda x: sent_tokenize(str(x).lower()))

    def flatten_sentences(self):
        """Flattens the list of sentences in the reports and keeps the associated hash."""
        print("[+] Flattening sentences list")
        sentences = []
        hashes = []
        for idx, row in self.reports_df.iterrows():
            for sentence in row['sentences']:
                sentences.append(sentence)
                hashes.append(row[self.report_identifier])
        return sentences, hashes

    def vectorize_and_calculate_similarity(self, sentences):
        """Vectorizes the sentences and phrases, and calculates cosine similarities."""
        print("[+] Vectorizing sentences and phrases")
        vectorizer = TfidfVectorizer().fit(sentences + self.phrases_df[self.phrase_field].tolist())
        tfidf_sentences = vectorizer.transform(sentences)
        tfidf_phrases = vectorizer.transform(self.phrases_df[self.phrase_field])
        
        print("[🧮] Calculating 'global' cosine similarity")
        cosine_similarities = cosine_similarity(tfidf_phrases, tfidf_sentences)
        
        print(f"[+] Cosine similarities shape: {cosine_similarities.shape}")
        return cosine_similarities

    def filter_results(self, sentences, hashes, cosine_similarities):
        """Filters the results based on the similarity threshold and constructs a dataframe."""
        print("[📊] Filtering results")
        results = []
        for idx_phrase, similarities in enumerate(cosine_similarities):
            similar_indices = np.where(similarities >= self.threshold)[0]
            print(f"\t╰─Phrase {idx_phrase} has {len(similar_indices)} similar sentences")

            for idx_sentence in similar_indices:
                results.append({
                    'phrase': self.phrases_df.loc[idx_phrase, self.phrase_field],  # the phrase from TRAM
                    'phrase_report_title': self.phrases_df.loc[idx_phrase, 'doc_title'],  # the title of the associated report in TRAM
                    'found_report_hash': hashes[idx_sentence],  # hash of the report (in our data) where the sentence was found
                    'similarity': similarities[idx_sentence]  # similarity score
                })

        self.results_df = pd.DataFrame(results).drop_duplicates()
    
    def save_full_report_text(self, output_file):
        """Merges the results with the full report texts and saves to a CSV file."""
        merged_df = pd.merge(self.results_df, self.reports_df, left_on='found_report_hash', right_on=self.report_identifier, how='left')
        output_df = merged_df[['found_report_hash', 'phrase', 'similarity', 'text']]
        output_df.columns = ['hash', 'phrase', 'similarity', 'full_report_text']

        print(f"[💾] Saving full report text results to {output_file}")
        output_df.to_csv(output_file, index=False)
    
    def save_adjacent_sentences(self, output_file, N):
        """Saves N adjacent sentences around the phrases to a CSV/JSON file."""
        def get_adjacent_sentences(text, phrase, N):
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
            output_data.append({
                'hash': row['found_report_hash'],
                'phrase': row['phrase'],
                'similarity': row['similarity'],
                'adjacent_sentences': adjacent_sentences
            })

        if output_file.endswith('.json'):
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=4)
        else:
            output_df = pd.DataFrame(output_data)
            output_df.to_csv(output_file, index=False)

        print(f"Results saved to {output_file}")

    def run(self, full_text_output, adjacent_sentences_output=None, N=0):
        """Runs the entire pipeline."""
        self.load_data()
        self.preprocess_reports()
        sentences, hashes = self.flatten_sentences()
        cosine_similarities = self.vectorize_and_calculate_similarity(sentences)
        self.filter_results(sentences, hashes, cosine_similarities)
        
        if N == 0:
            self.save_full_report_text(full_text_output)
        else:
            self.save_adjacent_sentences(adjacent_sentences_output, N)
        print("DONE!")


pipeline = SimilarityPipeline(
    reports_path='../pdf_clean_data.csv',               # path to the CSV file containing reports
    phrases_path='../tram_data/single_label.json',      # path to the JSON file containing phrases
    report_identifier='hash',                           # report identifier may change with JSON structure
    phrase_field='text',                                # phrase field may change with JSON structure
    threshold=0.75                                      # might have to specify the text field for reports
)

pipeline.run(
    full_text_output='../similarity-results/PIPELINE_full_report_similarity.csv',             # (str) path to save full report text results
    adjacent_sentences_output='./final_results/PIPELINE_similarities_with_full_reports.json', # (str) path to save adjacent sentences results
    N=0                                                                                       # N=0 for full reports
)
