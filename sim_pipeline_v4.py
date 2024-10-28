import pandas as pd
import json, csv
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import gc


class SimilarityPipeline:
    def __init__(self, reports_path, phrases_path, report_identifier='hash', phrase_field='sentence', label_field='label', threshold=0.75, batch_size=500):
        """
        Initializes the Pipeline with paths to the report CSV and phrases files.

        Parameters:
        - reports_path (str): Path to the CSV file containing reports.
        - phrases_path (str): Path to the CSV/JSON file containing phrases.
        - report_identifier (str): Column name in the reports CSV that serves as an identifier. Default is 'hash'.
        - phrase_field (str): Field name in the phrases file that contains the phrases. Default is 'sentence'.
        - label_field (str): Field name in the phrases file that contains the labels.
        - threshold (float): Cosine similarity threshold to consider a match. Default is 0.75.
        - batch_size (int): Number of sentences to process in each batch for similarity calculation.
        """
        self.reports_path = reports_path
        self.phrases_path = phrases_path
        self.report_identifier = report_identifier
        self.phrase_field = phrase_field
        self.label_field = label_field
        self.threshold = threshold
        self.batch_size = batch_size
        self.reports_df = None
        self.phrases_df = None
        self.results_df = None

    def load_data(self):
        """Loads the reports CSV and phrases file into dataframes."""
        print(f"[📂] Loading reports from {self.reports_path}")
        self.reports_df = pd.read_csv(self.reports_path, sep='|', quoting=csv.QUOTE_ALL)
        print(f"Report columns: {self.reports_df.columns}")

        print(f"[📂] Loading phrases from {self.phrases_path}")
        if self.phrases_path.endswith('.json'):
            with open(self.phrases_path, 'r') as file:
                phrases_data = json.load(file)
            self.phrases_df = pd.DataFrame(phrases_data)
        elif self.phrases_path.endswith('.csv'):
            self.phrases_df = pd.read_csv(self.phrases_path)
        else:
            raise ValueError("Unsupported file format for phrases. Only .csv and .json are supported.")
        print(f"Phrase columns: {self.phrases_df.columns}")

    def preprocess_reports(self):
        """Tokenizes and preprocesses the text field in the reports dataframe."""
        print("[🗃️ ] Tokenizing and preprocessing reports")
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
        """Vectorizes the sentences and phrases, and calculates cosine similarities in batches."""
        print("[+] Vectorizing sentences and phrases")
        if self.phrase_field not in self.phrases_df.columns:
            raise KeyError(f"The specified phrase_field '{self.phrase_field}' does not exist in the phrases dataframe. Available columns: {self.phrases_df.columns}")

        # Fit the vectorizer on both sentences and phrases
        vectorizer = TfidfVectorizer().fit(sentences + self.phrases_df[self.phrase_field].tolist())
        tfidf_sentences = vectorizer.transform(sentences)
        tfidf_phrases = vectorizer.transform(self.phrases_df[self.phrase_field])

        print("[🧮] Calculating cosine similarity in batches")
        batch_results = []
        num_batches = len(sentences) // self.batch_size + (1 if len(sentences) % self.batch_size != 0 else 0)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(sentences))

            # Batch-wise similarity calculation
            batch_similarities = cosine_similarity(tfidf_phrases, tfidf_sentences[start_idx:end_idx])

            # Append the numpy array directly
            batch_results.append(batch_similarities)

            print(f"Processed batch {batch_idx + 1}/{num_batches}")
            del batch_similarities  # Free memory from the batch
            gc.collect()  # Trigger garbage collection to free memory

        # Concatenate all batch results
        print("Done with all batches, concatenating final results")
        return np.hstack(batch_results)

    def filter_results(self, sentences, hashes, cosine_similarities):
        """Filters the results based on the similarity threshold and constructs a dataframe."""
        print("[📊] Filtering results")
        results = []
        for idx_phrase, similarities in enumerate(cosine_similarities):
            similar_indices = np.where(similarities >= self.threshold)[0]
            print(f"\t╰─Phrase {idx_phrase} has {len(similar_indices)} similar sentences")

            for idx_sentence in similar_indices:
                results.append({
                    'phrase': self.phrases_df.loc[idx_phrase, self.phrase_field],
                    'phrase_report_title': self.phrases_df.loc[idx_phrase, self.label_field],
                    'found_report_hash': hashes[idx_sentence],
                    'similarity': similarities[idx_sentence],
                    'found_sentence': sentences[idx_sentence],  # add the found sentence to retrieve adjacent sentences later
                    'label': self.phrases_df.loc[idx_phrase, self.label_field],  # Storing the label as well
                })

        self.results_df = pd.DataFrame(results).drop_duplicates()

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


####################
#  Pipeline setup  #
####################

pipeline = SimilarityPipeline(
    reports_path='./data/processed_documents.csv',
    phrases_path='./data/cti2mitre_train.csv',
    report_identifier='_id',
    phrase_field='sentence',                # TRAM
    label_field='label',                    # TRAM
    threshold=0.75,
    batch_size=100
)

pipeline.run(
    full_text_output='./similarity-results/full_report_similarity.csv',
    adjacent_sentences_output='./similarity-results/oct15.json',
    N=3
)
