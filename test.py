import polars as pl
import csv, json, os, gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize

class SimilarityPipeline:
    def __init__(self, reports_path, phrases_path, output_path, 
                 report_identifier='_id', phrase_field='sentence',
                 label_field='label_tec', threshold=0.75, N=2):
        self.reports_path = reports_path
        self.phrases_path = phrases_path
        self.output_path = output_path
        self.report_identifier = report_identifier
        self.phrase_field = phrase_field
        self.label_field = label_field
        self.threshold = threshold
        self.N = N

    def load_data(self):
        print(f"[📂] Loading reports from {self.reports_path}")
        reports_df = pl.read_csv(self.reports_path, separator="|", null_values="\0", truncate_ragged_lines=True)
        reports_df = reports_df.with_columns(
            pl.col("text").map_elements(lambda text: sent_tokenize(text.lower().strip())).alias("sentences")
        ).drop("text")

        print(f"[📂] Loading phrases from {self.phrases_path}")
        phrases_df = pl.read_csv(self.phrases_path, separator=",", null_values="\0", truncate_ragged_lines=True)
        # print("REPORTS DF:")
        # print(reports_df.head(3))

        # print("PHRASES DF:")
        # print(phrases_df.head(3))

        return reports_df, phrases_df

    def find_sentences(self, reports_df, phrases_df):
        vectorizer = TfidfVectorizer()
        tfidf_phrases = vectorizer.fit_transform(phrases_df[self.phrase_field].to_list())

        results = []
        for idx_phrase, phrase in enumerate(phrases_df[self.phrase_field]):
            for report in reports_df.iter_rows(named=True):
                sentences = report['sentences']
                tfidf_sentences = vectorizer.transform(sentences)
                similarities = cosine_similarity(tfidf_phrases[idx_phrase], tfidf_sentences)
                
                similar_indices = similarities[0].argsort()[::-1]
                for idx_sentence in similar_indices:
                    if similarities[0, idx_sentence] >= self.threshold:
                        adj_sentences = self.get_adjacent_sentences(sentences, idx_sentence)
                        results.append({
                            'label': phrases_df[self.label_field][idx_phrase],
                            'found_report_hash': report[self.report_identifier],
                            'phrase': phrase,
                            'similarity': similarities[0, idx_sentence],
                            'adjacent_sentences': adj_sentences,
                        })
        return results

    def get_adjacent_sentences(self, sentences, idx, N=None):
        N = N or self.N
        start_idx = max(0, idx - N)
        end_idx = min(len(sentences), idx + N + 1)
        return sentences[start_idx:end_idx]

    def save_results_to_csv(self, results):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        print(f"[💾] Writing results to {self.output_path}")
        with open(self.output_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['phrase', 'label', 'found_report_hash', 'similarity', 'found_sentence'])
            writer.writeheader()
            writer.writerows(results)

    def run(self):
        reports_df, phrases_df = self.load_data()
        results = self.find_sentences(reports_df, phrases_df)
        self.save_results_to_csv(results)
        del reports_df, phrases_df, results
        gc.collect()
        print("[✅] Finished processing!")


# Execution
def main():
    pipeline = SimilarityPipeline(
        reports_path='./data/reports/reports_500.csv',
        phrases_path='./data/cti_to_mitre/split_files/cti_to_mitre_part_1.csv',
        output_path='./similarity-results/full_report_similarity.csv',
        report_identifier='_id',
        phrase_field='sentence',
        label_field='label_tec',
        threshold=0.75,
        N=2
    )
    pipeline.run()

if __name__ == '__main__':
    csv.field_size_limit(10**9)
    main()
