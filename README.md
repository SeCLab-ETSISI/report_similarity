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

## Contribute

### Commit && PR Messages

```txt
[MODULE][FIX|ADD|DELETE] Summary of modifications

* [MODULE2][ADD] List of modifications from a general perspective
```

#### Example

```txt
[SC2-RL][FIX] Diagrams && subprocess
* [README][ADD] Execution && Component diagram
* [BOT][ADD] Change from `aioredis` to `redis.asyncio`
* [RUN_GAME][DELETE]
* [SC2ENV][FIX] Subprocess platform
```

