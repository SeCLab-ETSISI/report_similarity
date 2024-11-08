import pandas as pd

# Set the path to your input file
input_file = 'data/cti_to_mitre/cti_to_mitre_full.csv'

# Set the path for the output directory and prefix for the smaller files
output_dir = 'data/cti_to_mitre/split_files/'
output_prefix = 'cti_to_mitre_part'

# Set the chunk size (number of rows per smaller file)
chunk_size = 5000

# Read and split the CSV file
try:
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        # Create filename for each chunk
        output_file = f"{output_dir}{output_prefix}_{i + 1}.csv"

        # Write chunk to a new CSV file
        chunk.to_csv(output_file, index=False)

        print(f"Created: {output_file}")

    print("File split complete.")
except Exception as e:
    print(f"An error occurred: {e}")