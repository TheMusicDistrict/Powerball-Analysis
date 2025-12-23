# Powerball Analysis Apps

This repository contains two Streamlit dashboards for exploring historical lottery draws:

- **`streamlit_app.py`** – Mega Millions analysis with frequency heatmaps, co-occurrence charts, ticket samplers, and PDF/HTML export.
- **`Powerballv2.py`** – Powerball-focused analysis with expanded statistics (recency weighting, streaks, pair correlations, "virtual seeds" when available) and candidate ticket generators.

Both apps are intended for educational visualizations. Lottery draws are random; these tools highlight historical patterns only.

## Quick start

Requirements: Python 3.9+ and Streamlit (installed via `requirements.txt`).

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch one of the dashboards:

   - Mega Millions:
     ```bash
     streamlit run streamlit_app.py
     ```
   - Powerball (advanced):
     ```bash
     streamlit run Powerballv2.py
     ```

Streamlit will print a local URL to open the app in your browser.

### Typical workflow inside the app

1. Choose a data source on the sidebar:
   - **Auto-download** to pull the latest drawings from the New York Open Data feed.
   - **Upload CSV** if you have your own history file (see format below).
2. Pick the analysis options you want to render (frequency, recency weighting, co-occurrence charts, etc.).
3. Use the candidate ticket generator to create sample plays. You can adjust how numbers are balanced and filter out duplicates.
4. Download any generated tickets or reports (CSV/PDF/HTML) directly from the app.

## Using your own data

Both apps can auto-download recent drawings from the New York Open Data feeds, or you can upload a CSV with the following columns:

- `date` – Draw date in a format Pandas can parse.
- `w1` `w2` `w3` `w4` `w5` – The five white ball numbers.
- `mega` – Mega Ball/Powerball number.

Column names are case-insensitive; the apps will sort the data by date automatically.

Example CSV (Mega Millions):

```csv
date,w1,w2,w3,w4,w5,mega
2024-07-02,9,13,27,31,45,4
2024-06-28,2,18,21,32,44,14
```

## Key features

### Mega Millions (`streamlit_app.py`)
- Frequency heatmaps (plain and recency-weighted) for white balls.
- Recency-weighted Mega Ball bar chart.
- Optional 70×70 co-occurrence heatmap for white ball pairs.
- Candidate ticket generators (fair coverage or recency-weighted) with CSV export.
- PDF and HTML report builders that embed top numbers and charts.

### Powerball (`Powerballv2.py`)
- Recency-weighted and plain frequency analysis for white balls and Powerball.
- Randomness checks (chi-square), streak statistics, and pair correlation rankings.
- Advanced candidate ticket generation with range balancing and statistical scoring.
- Optional co-occurrence heatmap for white ball pairs (69×69 grid).
- Virtual seed experimentation and backtesting when `Seeds.py` is available.
- CSV downloaders for generated ticket sets.

## Notes

- Some visualizations (pair co-occurrence heatmaps) are computationally heavy across long date ranges; disable them if performance is an issue.
- Network access is required when using the auto-download mode to fetch the latest draws from the New York Open Data endpoints.
