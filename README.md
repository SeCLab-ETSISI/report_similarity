# report_similarity


# Project Setup


It is recommended to create a virtual environment:
   ```sh
   python -m venv venv
   ```

and install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Project Structure

* The `final_results` folder contains example outputs for sentences and full report similarities.
* `pdf_csv_pipeline.py` allows to create a .csv file from a report full of .pdf files, extracting their md5 hash (identifier) and text.
* `sim_pipeline.py` allows to find sentences in full texts and obtain the `N` adjacent sentences, use `N=0` for full texts.


```txt
├── LICENSE
├── README.md
├── final_results
│   ├── PIPELINE_adjacent_sentences_similarity.json
│   ├── PIPELINE_full_report_similarity.csv
│   └── adjacent_sentences_similarity.json
├── pdf_csv_pipeline.py
├── requirements.txt
└── sim_pipeline.py
```

## How to use
Create a csv file using `pdf_csv_pipeline.py`, and use the resulting .csv file in `sim_pipeline.py`.

## Contribute

### Commit && PR Messages

```txt
[MODULE][FIX|ADD|DELETE] Summary of modifications

* [MODULE2][ADD] List of modifications from a general perspective
```

#### Example

```txt
* [README][ADD] Execution && Component diagram
* [PIPELINES][ADD] PDF functionality
* [RESULTS][DELETE]
```

