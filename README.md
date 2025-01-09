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
* `tfidf_pipeline.py` allows to find sentences in full texts and obtain the `N` adjacent sentences.
* The `data` folder contains the full reports dataset *reports_18k.csv*, and a script to retrieve the aforementioned texts. The [CTI to mitre](https://github.com/dessertlab/cti-to-mitre-with-nlp) and [TRAM](https://github.com/center-for-threat-informed-defense/tram) datasets are included as sentences for retrieval. 
Split versions of *CTI to mitre* are included for smaller scale tests.
* The root folder contains the main scripts, `tfidf_pipeline.py` and `tfidf_script.py`. The former contains code readability upgrades and allows to comfortably adjust parameters such as batch size, similarity threshold, datasets and output files. The latter is maintained in order to conduct paralellization tests.

Some example results are kept under the `batch_outputs` folder, and results obtained in previous experiments can be found within the `old_results` directory.

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

