# Thesis

A social media auditing pipeline that uses LLMs to annotate social media data.
The pipeline covers dataset sampling, automated LLM-based annotation, and case study analysis.

# Structure

- Dataset_Sampling/ - samples and prepares social media data for annotation
- LLM_annotations/ - runs LLM annotation over the sampled dataset
- case_studies/ - analysis and results for individual case studies

# Requirements

- Python 3.x
- datasets
- pandas
- numpy
- huggingface_hub
- duckdb
- pyarrow
- tqdm

# Usage

    git clone https://github.com/meshchem/Thesis.git
    cd Thesis
    pip install -r requirements.txt
    jupyter notebook

# Pipeline

1. Sample social media data using notebooks in Dataset_Sampling/
2. Run LLM annotations over the sampled data using LLM_annotations/
3. Explore results and case studies in case_studies/