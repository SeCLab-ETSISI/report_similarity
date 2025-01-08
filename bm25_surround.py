# 
# IMPORTANT NOTE:
#    - phrases refer to phrases extracted from the TRAM (or equivalent) dataset
#    - sentences refer to the sentences extracted from the reports (sentence tokenized)
# 

import os
import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from itertools import islice

def load_data(reports_path, phrases_path):
    print(f"[📂] Loading reports from {reports_path}")
    reports_df = pl.read_csv(reports_path, separator="|", null_values="\0", truncate_ragged_lines=True)
    
    if 'text' in reports_df.columns and '_id' in reports_df.columns:
        print("[📄] Tokenizing report text into sentences.")
        reports_df = reports_df.with_columns(
            pl.col("text").map_elements(lambda text: sent_tokenize(text.lower().strip()), return_dtype=pl.List(pl.Utf8)).alias("sentences")
        )
    else:
        print("[❌] The 'text' or '_id' column is missing.")
        raise ValueError("Input CSV must contain 'text' and '_id' columns.")
    
    print(f"[📂] Loading phrases from {phrases_path}")
    phrases_df = pl.read_csv(phrases_path, separator=",", null_values="\0", truncate_ragged_lines=True)
    
    # lowercase the 'sentence' column of the phrases (higher similarity scores)
    phrases_df = phrases_df.with_columns(
        pl.col("sentence").str.to_lowercase().alias("sentence")
    )
    
    return reports_df, phrases_df

def batch_iterable(iterable, batch_size):
    """Yields batches from an iterable."""
    iterator = iter(iterable)
    for first in iterator:
        yield [first, *islice(iterator, batch_size - 1)]

def tokenize_sentences(sentences):
    print(f"[📝] Tokenizing {len(sentences)} sentences.")
    return [word_tokenize(sentence) for sentence in sentences]

def get_surrounding_sentences(sentences, idx, num_surrounding=2):
    """Fetches the surrounding sentences before and after the matched sentence, ensuring accurate boundaries."""
    # adjust start and end indices, making sure we don't go out of bounds
    start_idx = max(0, idx - num_surrounding)
    end_idx = min(len(sentences), idx + num_surrounding + 1)

    # sentences before and after the one we found based on the adjusted indices
    before = sentences[start_idx:idx]
    after = sentences[idx+1:end_idx]

    return before, sentences[idx], after

if __name__ == '__main__':
    # future params for pipeline
    #@todo: adapt to pipeline class
    reports_path = './data/reports/reports_18k.csv'
    phrases_path = './data/cti_to_mitre/cti_to_mitre_full.csv'
    output_file = './batch_outputs/all_results_final.csv'
    batch_size = 5000  # report sentences per batch: do not mistake with dataset phrases

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print("[📂] Loading data...")
    reports_df, phrases_df = load_data(reports_path, phrases_path)

    # flatten sentences with report IDs because dimension errors
    print(f"[🔄] Flattening report sentences with IDs.")
    report_sentences = [
        (sentence, report_id)
        for report_id, sentences in zip(reports_df['_id'].to_list(), reports_df['sentences'].to_list())
        if sentences is not None
        for sentence in sentences
    ]
    print(f"[✅] Flattened {len(report_sentences)} sentences from reports.")

    print("[🔍] Starting batch processing...")

    # Open the output file once and write header if necessary
    if not os.path.exists(output_file):
        header_needed = True
    else:
        header_needed = False

    with open(output_file, 'a', newline='', encoding='utf-8') as output_csv:
        # Create CSV writer once for the entire run
        csv_writer = None

        for batch_idx, batch in enumerate(batch_iterable(report_sentences, batch_size)):
            print(f"[🔄] Processing batch {batch_idx + 1}")
            
            # Separate sentences and report IDs
            batch_sentences, batch_ids = zip(*batch)

            # Tokenize batch sentences
            tokenized_reports = tokenize_sentences(batch_sentences)
            
            # Prepare all sentences (reports and phrases) for TF-IDF vectorization
            print("[📈] Preparing sentences for TF-IDF vectorization.")
            all_sentences = list(batch_sentences) + phrases_df['sentence'].to_list()  # All sentences in the dataset (a lo bestia)

            
            vectorizer = TfidfVectorizer()      # fit to both report sentences and phrases
            tfidf_matrix = vectorizer.fit_transform(all_sentences)

            # Split the TF-IDF matrix: the first part corresponds to report sentences, and the second to phrases
            report_tfidf_matrix = tfidf_matrix[:len(batch_sentences)]
            phrase_tfidf_matrix = tfidf_matrix[len(batch_sentences):]

            # results for the current batch
            results = []

            # cosine similarity between report sentences and phrases in the TRAM/whatever dataset
            for idx, phrase in enumerate(phrases_df['sentence']):
                phrase_tfidf = phrase_tfidf_matrix[idx]

                # cosine similarity between the current phrase and all report sentences
                cosine_scores = cosine_similarity(phrase_tfidf, report_tfidf_matrix).flatten()

                # store results where the cosine similarity is above the established threshold
                for rank, (cosine_score, sentence, report_id) in enumerate(zip(cosine_scores, batch_sentences, batch_ids)):
                    if cosine_score >= 0.7:
                        before, matching_sentence, after = get_surrounding_sentences(batch_sentences, rank)

                        # append results 
                        results.append({
                            'label_tec': phrases_df['label_tec'][idx],  # label_tec from phrases_df
                            'original_phrase': phrase,
                            'before': ' '.join(before),
                            'matching_report_sentence': matching_sentence,
                            'after': ' '.join(after),
                            'report_id': report_id,
                            'rank': rank + 1,
                            'cosine_score': cosine_score
                        })

            # write results for the current batch to the single CSV file
            if results:
                results_df = pd.DataFrame(results)
                if header_needed:
                    #! header only once
                    results_df.to_csv(output_csv, index=False, escapechar="\\", header=True)
                    header_needed = False  #! after writing the header, no need to write it again
                else:       # just append
                    results_df.to_csv(output_csv, index=False, escapechar="\\", header=False)

                # clear results from memory
                del results
                print(f"[✅] Batch {batch_idx + 1} processed and saved to single file.")

    print("[🏁] Batch processing complete. Check the output file for all results.")
