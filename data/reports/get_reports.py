#########################################################################################
#                                                                                       #        
#   fetches all the reports from the MongoDB database and exports them to a .csv file.  #
#                                                                                       #        
#########################################################################################


import os
import csv
from pymongo import MongoClient
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# env and constants
load_dotenv("../.env")

CONNECTION_STRING = os.getenv("CONNECTION_STRING")
DATABASE = "APT"
COLLECTION = "reports_v2"
OUTPUT_FILENAME = "reports_18k.csv"

client = MongoClient(CONNECTION_STRING)
db = client[DATABASE]
collection = db[COLLECTION]

documents = list(collection.find())
total_docs = len(documents)

data = []
with tqdm(total=total_docs, desc="Processing documents") as pbar:
    for doc in documents:
        # remove line breaks
        text_single_line = doc.get("text", "").replace("\n", " ")

        data.append({
            "_id": str(doc.get("_id", "")),
            "text": text_single_line,
            "hashes": doc.get("hashes", []),
            "ip_addrs": doc.get("ip_addrs", []),
            "domains": doc.get("domains", []),
            "url": doc.get("url", "")
        })

        pbar.update(1)

print("[+] Creating dataframe")
df = pd.DataFrame(data)
print("[+] Exporting to .csv")
df.to_csv(OUTPUT_FILENAME, sep="|", index=False, quoting=csv.QUOTE_MINIMAL, escapechar="\\")

print(f"Data exported successfully to {OUTPUT_FILENAME}")
