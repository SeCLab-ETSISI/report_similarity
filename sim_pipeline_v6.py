import polars as pl
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import gc  # Garbage collection module

# Function to load data
def load_data(reports_path, phrases_path):
    print(f"[📂] Loading reports from {reports_path}")
    reports_df = pl.read_csv(reports_path, separator="|", null_values="\0", truncate_ragged_lines=True)
    reports_df = reports_df.with_columns(
        pl.col("text").map_elements(lambda text: sent_tokenize(text.lower().strip())).alias("sentences")
    ).drop("text")
    
    print(f"[📂] Loading phrases from {phrases_path}")
    phrases_df = pl.read_csv(phrases_path, separator=",", null_values="\0", truncate_ragged_lines=True)
    
    return reports_df, phrases_df

# Function to retrieve surrounding sentences
def get_surrounding_sentences(sentences, index, n):
    # Get N sentences before, the sentence itself, and N sentences after
    start = max(0, index - n)
    end = min(len(sentences), index + n + 1)
    surrounding_sentences = sentences[start:end]
    return surrounding_sentences

# Function to process and save output to file
def process_and_save_output(reports_df, phrases_df, output_path, top_n=1, surrounding_n=2, similarity_threshold=0.75):
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Open output CSV file
    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['phrase_to_search', 'sentences_before', 'sentences_after', 'found_at (offset)', 'similarity_score'])
        
        # Process each report
        for report_idx, report_row in reports_df.iterrows():
            print(f"Processing report {report_idx + 1}/{len(reports_df)}")
            
            first_report_sentences = report_row['sentences']
            
            # Process each phrase
            for idx, first_phrase in enumerate(phrases_df['sentence']):
                print(f"Finding for phrase: {first_phrase}")
                
                # Use a pre-filtering mechanism to quickly ignore highly dissimilar sentences
                filtered_sentences = [
                    sentence for sentence in first_report_sentences
                    if len(sentence.split()) > 3  # Avoid very short sentences that are unlikely to match
                ]
                
                # Score each sentence in the report against the phrase
                sentence_scores = []
                for i, sentence in enumerate(filtered_sentences):
                    documents = [sentence, first_phrase]
                    tfidf_matrix = vectorizer.fit_transform(documents)
                    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0, 0]
                    del tfidf_matrix  # Delete TF-IDF matrix after use
                    gc.collect()  # Run garbage collection
                    
                    # Only append to sentence_scores if similarity is above the threshold
                    if similarity_score >= similarity_threshold:
                        sentence_scores.append((i, sentence, similarity_score))
                    
                    del similarity_score  # Delete similarity score after appending
                    gc.collect()  # Run garbage collection
                
                # If we found valid sentences (with similarity >= 0.75)
                if sentence_scores:
                    # Sort sentences by similarity score in descending order
                    sorted_sentences = sorted(sentence_scores, key=lambda x: x[2], reverse=True)
                    
                    # Get the most similar sentence (top 1)
                    sentence_index, top_sentence, score = sorted_sentences[0]
                    
                    # Retrieve surrounding sentences
                    surrounding_sentences = get_surrounding_sentences(first_report_sentences, sentence_index, surrounding_n)
                    sentences_before = ' '.join(surrounding_sentences[:surrounding_n])
                    sentences_after = ' '.join(surrounding_sentences[surrounding_n + 1:])
                    
                    # Write to CSV
                    writer.writerow([first_phrase, sentences_before, sentences_after, sentence_index, score])
                    print(f"Score: {score:.4f} | Sentence: {top_sentence}")
            
            # Clear sentence_scores after processing each report
            del sentence_scores  # Delete the sentence_scores list
            gc.collect()  # Run garbage collection

# Main Execution
if __name__ == '__main__':
    # File paths
    reports_path = './data/reports/reports_18k.csv'
    phrases_path = './data/cti_to_mitre/split_files/cti_to_mitre_part_1.csv'
    output_path = './similarity_output2.csv'
    
    # Load data
    reports_df, phrases_df = load_data(reports_path, phrases_path)
    
    # Process and save the output
    process_and_save_output(reports_df, phrases_df, output_path)
