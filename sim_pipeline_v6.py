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
        self.reports_df = pd.read_csv(self.reports_path, sep='|', quoting=csv.QUOTE_ALL, on_bad_lines='skip', engine='python')
        print(f"Report columns: {self.reports_df.columns}")

        print(f"[📂] Loading phrases from {self.phrases_path}")
        if self.phrases_path.endswith('.json'):
            with open(self.phrases_path, 'r') as file:
                phrases_data = json.load(file)
            self.phrases_df = pd.DataFrame(phrases_data)
        elif self.phrases_path.endswith('.csv'):
            self.phrases_df = pd.read_csv(self.phrases_path, low_memory=False, on_bad_lines='skip')
        else:
            raise ValueError("Unsupported file format for phrases. Only .csv and .json are supported.")
        print(f"Phrase columns: {self.phrases_df.columns}")
        print(f"Dataframe shape: {self.phrases_df.shape}")
        print(f"Dataframe memory usage: {self.phrases_df.memory_usage().sum() / 1024**2:.2f} MB")



    def preprocess_reports(self):
        print("[🗃️ ] Tokenizing and preprocessing reports")
        self.reports_df['sentences'] = self.reports_df['text'].apply(lambda x: sent_tokenize(str(x).lower()))

    def flatten_sentences(self):
        print("[+] Flattening sentences list")
        sentences = []
        hashes = []
        for idx, row in self.reports_df.iterrows():
            for sentence in row['sentences']:
                sentences.append(sentence)
                hashes.append(row[self.report_identifier])
        return sentences, hashes

    def vectorize_and_calculate_similarity(self, sentences):
        print("[+] Vectorizing sentences and phrases")
        if self.phrase_field not in self.phrases_df.columns:
            raise KeyError(f"The specified phrase_field '{self.phrase_field}' does not exist in the phrases dataframe. Available columns: {self.phrases_df.columns}")

        vectorizer = TfidfVectorizer().fit(sentences + self.phrases_df[self.phrase_field].tolist())
        tfidf_sentences = vectorizer.transform(sentences)
        tfidf_phrases = vectorizer.transform(self.phrases_df[self.phrase_field])

        print("[🧮] Calculating cosine similarity in batches")
        batch_results = []
        num_batches = len(sentences) // self.batch_size + (1 if len(sentences) % self.batch_size != 0 else 0)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(sentences))

            batch_similarities = cosine_similarity(tfidf_phrases, tfidf_sentences[start_idx:end_idx])
            batch_results.append(batch_similarities)
            print(f"Processed batch {batch_idx + 1}/{num_batches}")
            del batch_similarities  # Free memory from the batch
            gc.collect()  # Trigger garbage collection to free memory

        print("Done with all batches, concatenating final results")
        return np.hstack(batch_results)

    def filter_results(self, sentences, hashes, cosine_similarities):
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
                    'found_sentence': sentences[idx_sentence],
                    'label': self.phrases_df.loc[idx_phrase, self.label_field],
                })

        self.results_df = pd.DataFrame(results).drop_duplicates()

    '''
    def save_full_report_text(self, output_path):
        """Saves or appends results to a CSV file."""
        print(f"[💾] Saving full report text to {output_path}")
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path)
            combined_df = pd.concat([existing_df, self.results_df]).drop_duplicates().reset_index(drop=True)
        else:
            combined_df = self.results_df
        combined_df.to_csv(output_path, index=False)
    '''

    def save_full_report_text(self, output_path):
        """Saves results to a CSV file in batches, freeing memory after each batch."""
        print(f"[💾] Saving full report text to {output_path} in batches")
        write_header = not os.path.exists(output_path)  # Write header only if the file doesn't exist

        with open(output_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=self.results_df.columns)
            if write_header:
                writer.writeheader()

            # Write in batches
            num_batches = len(self.results_df) // self.batch_size + (1 if len(self.results_df) % self.batch_size != 0 else 0)
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min((batch_idx + 1) * self.batch_size, len(self.results_df))

                batch = self.results_df.iloc[start_idx:end_idx].to_dict(orient='records')
                writer.writerows(batch)
                print(f"Saved batch {batch_idx + 1}/{num_batches}")
                del batch  # Free memory for the batch
                gc.collect()  # Trigger garbage collection to release unused memory

        print("[✅] Finished saving full report text")


    '''
    def save_adjacent_sentences(self, output_path, N):
        """Saves or appends results to a JSON file with N adjacent sentences."""
        print(f"[💾] Saving adjacent sentences to {output_path}")
        if os.path.exists(output_path):
            with open(output_path, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = []

        new_data = self.results_df.to_dict(orient='records')
        combined_data = existing_data + new_data

        with open(output_path, 'w') as file:
            json.dump(combined_data, file, indent=2)
    '''

    def save_adjacent_sentences(self, output_path, N):
        """Saves or appends results to a JSON file in batches, freeing memory after each batch."""
        print(f"[💾] Saving adjacent sentences to {output_path} in batches")
        write_mode = 'w' if not os.path.exists(output_path) else 'r+'

        if os.path.exists(output_path):
            with open(output_path, 'r', encoding='utf-8') as file:
                existing_data = json.load(file)
        else:
            existing_data = []

        num_batches = len(self.results_df) // self.batch_size + (1 if len(self.results_df) % self.batch_size != 0 else 0)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(self.results_df))

            batch_data = self.results_df.iloc[start_idx:end_idx].to_dict(orient='records')
            combined_batch = existing_data + batch_data

            with open(output_path, write_mode, encoding='utf-8') as file:
                json.dump(combined_batch, file, indent=2)
                write_mode = 'w'  # Switch to write mode after the first append
                print(f"Saved batch {batch_idx + 1}/{num_batches}")
            existing_data = []  # Clear after initial load to avoid duplicating data
            del batch_data, combined_batch  # Free memory for the batch
            gc.collect()

        print("[✅] Finished saving adjacent sentences")


    def run(self, full_text_output, adjacent_sentences_output=None, N=0):
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


############################################################
#                   Pipeline setup                         #   
#       If the output file alreay exists, the results      #
#       will be appended to the existing file.             #
############################################################


pipeline = SimilarityPipeline(
   
    reports_path='./data/reports/reports_18k.csv',
    phrases_path='./data/cti_to_mitre/split_files/cti_to_mitre_full.csv', ### OJO!!! PART 1
    report_identifier='_id',
    phrase_field='sentence',                    # TRAM
    label_field='label_tec',                    # TRAM
    threshold=0.75,
    batch_size=50
)

pipeline.run(
    full_text_output='./similarity-results/full_report_similarity.csv',
    adjacent_sentences_output='./similarity-results/nov14.json',
    N=3
)
