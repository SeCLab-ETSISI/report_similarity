import polars as pl
import json, csv, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import gc
from io import StringIO


def get_adjacent_sentences(batch_df, idx_sentence, N):
    """Returns N adjacent sentences before and after the target sentence."""
    start_idx = max(0, idx_sentence - N)
    end_idx = min(len(batch_df) - 1, idx_sentence + N)
    return batch_df[start_idx:end_idx + 1].to_dicts()


class SimilarityPipeline:
    def __init__(self, reports_path, phrases_path, report_identifier='_id', phrase_field='sentence',
                 label_field='label_tec', threshold=0.75, batch_size=500, N=2):
        self.reports_path = reports_path
        self.phrases_path = phrases_path
        self.report_identifier = report_identifier
        self.phrase_field = phrase_field
        self.label_field = label_field
        self.threshold = threshold
        self.batch_size = batch_size
        self.N = N
        
    def load_data(self):
        """Loads phrases into a Polars DataFrame."""
        print(f"[📂] Loading phrases from {self.phrases_path}")

        if self.phrases_path.endswith('.csv'):
            self.phrases_df = pl.read_csv(self.phrases_path)
        elif self.phrases_path.endswith('.json'):
            with open(self.phrases_path, 'r') as file:
                phrases_data = json.load(file)
            self.phrases_df = pl.DataFrame(phrases_data)
        else:
            raise ValueError("Unsupported file format for phrases. Only .csv and .json are supported.")

        print(f"Phrase columns: {self.phrases_df.columns}")
        print(f"Dataframe shape: {self.phrases_df.shape}")

    def process_reports(self):
        """Yields tokenized sentences from the reports CSV file."""
        print(f"[📂] Preprocessing reports from {self.reports_path}")

        with open(self.reports_path, 'r', encoding='utf-8') as file:
            data = file.read().replace('\x00', '')
            file_no_nulls = StringIO(data)
            reader = csv.DictReader(file_no_nulls, delimiter='|', quoting=csv.QUOTE_ALL)

            for row in reader:
                text = row.get('text')
                if text:
                    sentences = sent_tokenize(text.lower().strip())
                    for sentence in sentences:
                        yield {'sentence': sentence, 'hash': row[self.report_identifier]}

    def vectorize_and_calculate_similarity(self, batch_df):
        """Vectorizes sentences and calculates cosine similarity."""
        print("[+] Vectorizing batch sentences and phrases")

        vectorizer = TfidfVectorizer()
        tfidf_phrases = vectorizer.fit_transform(self.phrases_df[self.phrase_field].to_list())

        tfidf_sentences = vectorizer.transform(batch_df['sentence'].to_list())
        similarities = cosine_similarity(tfidf_phrases, tfidf_sentences)
        return csr_matrix(similarities)

    def filter_results(self, batch_df, similarities):
        """Filters results and includes N adjacent sentences."""
        results = []
        for idx_phrase, sims in enumerate(similarities):
            similar_indices = sims.indices[sims.data >= self.threshold]

            for idx_sentence in similar_indices:
                adjacent_sentences = get_adjacent_sentences(batch_df, idx_sentence, self.N)
                for sentence_info in adjacent_sentences:
                    results.append({
                        'phrase': self.phrases_df[self.phrase_field][idx_phrase],
                        'phrase_report_title': self.phrases_df[self.label_field][idx_phrase],
                        'found_report_hash': sentence_info['hash'],
                        'similarity': sims.data[sims.indices == idx_sentence][0],
                        'found_sentence': sentence_info['sentence'],
                        'label': self.phrases_df[self.label_field][idx_phrase],
                    })
        return results

    def ensure_directory_exists(self, file_path):
        """Ensure the directory for the given file path exists."""
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[📂] Created directory: {directory}")

    def save_results_to_csv(self, results, output_path):
        """Writes results to CSV in streaming mode."""
        self.ensure_directory_exists(output_path)
        print(f"[💾] Writing results to {output_path}")

        write_header = not os.path.exists(output_path)
        with open(output_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=[
                'phrase', 'phrase_report_title', 'found_report_hash',
                'similarity', 'found_sentence', 'label'
            ])
            if write_header:
                writer.writeheader()
            writer.writerows(results)

    def run(self, output_path):
        self.load_data()
        report_sentences = list(self.process_reports())

        num_batches = len(report_sentences) // self.batch_size + (
            1 if len(report_sentences) % self.batch_size != 0 else 0
        )

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(report_sentences))

            # Print the batch info with actual sentence indices
            print(f"[🚀] Processing batch {batch_idx + 1}/{num_batches}")

            batch_df = pl.DataFrame(report_sentences[start_idx:end_idx])
            similarities = self.vectorize_and_calculate_similarity(batch_df)
            results = self.filter_results(batch_df, similarities)
            self.save_results_to_csv(results, output_path)

            # Free memory
            del batch_df, similarities, results
            gc.collect()

        print("[✅] Finished processing!")


############################################################
#                   Pipeline Execution                    #
############################################################

def main():
    pipeline = SimilarityPipeline(
        reports_path='./data/reports/reports_18k.csv',
        phrases_path='./data/cti_to_mitre/cti_to_mitre_full.csv',
        report_identifier='_id',
        phrase_field='sentence',
        label_field='label_tec',
        threshold=0.75,
        batch_size=5000,
        N=2
    )

    pipeline.run(
        output_path='./similarity-results/full_report_similarity.csv'
    )


if __name__ == '__main__':
    csv.field_size_limit(10**9)
    main()

