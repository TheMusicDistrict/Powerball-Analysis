# Powerball Analysis Apps

This repository contains two Streamlit dashboards for exploring historical lottery draws:

- **`streamlit_app.py`** – Mega Millions analysis with frequency heatmaps, co-occurrence charts, ticket samplers, and PDF/HTML export.
- **`Powerballv2.py`** – Powerball-focused analysis with expanded statistics (recency weighting, streaks, pair correlations, "virtual seeds" when available) and candidate ticket generators.

Both apps are intended for educational visualizations. Lottery draws are random; these tools highlight historical patterns only.

## Quick start

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

## Using your own data

Both apps can auto-download recent drawings from the New York Open Data feeds, or you can upload a CSV with the following columns:

- `date` – Draw date in a format Pandas can parse.
- `w1` `w2` `w3` `w4` `w5` – The five white ball numbers.
- `mega` – Mega Ball/Powerball number.

Column names are case-insensitive; the apps will sort the data by date automatically.

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
