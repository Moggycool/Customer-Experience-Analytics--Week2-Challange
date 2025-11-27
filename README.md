# Customer-Experience-Analytics--Week2-Challange
This week’s challenge centers on analyzing customer satisfaction with mobile banking apps by collecting and processing user reviews from the Google Play Store for three Ethiopian banks: Commercial Bank of Ethiopia (CBE); Bank of Abyssinia (BOA); Dashen Bank
## Project Structure
```

```
Customer-Experience-Analytics--Week2-Challange
├─ .env
├─ data
│  ├─ preprocessed
│  │  └─ google_play_processed_reviews.csv
│  ├─ raw
│  │  ├─ google_play_app_info.csv
│  │  └─ google_play_raw_reviews.csv
│  └─ README.md
├─ notebooks
│  ├─ preprocessing_eda.ipynb
│  └─ scraping_eda.ipynb
├─ README.md
├─ requirements.txt
├─ scripts
├─ src
│  ├─ config.py
│  ├─ preprocessing.py
│  ├─ scraping.py
│  ├─ __init__.py
│  └─ __pycache__
│     ├─ config.cpython-313.pyc
│     ├─ preprocessing.cpython-313.pyc
│     ├─ scraping.cpython-313.pyc
│     └─ __init__.cpython-313.pyc
└─ workflows
   ├─ CI.yml
   └─ unittest.yml

`
---

## Features

### 1. Git Setup
- GitHub repository initialized.
- `.gitignore` included to avoid tracking unnecessary files.
- `requirements.txt` included for reproducible environment.
- All work done on the `task-1` branch.
- Frequent commits with meaningful messages to capture logical chunks of work.

### 2. Web Scraping
- Uses [`google-play-scraper`](https://pypi.org/project/google-play-scraper/) to collect reviews from Google Play.
- Data collected for three Ethiopian banks.
- Each review includes:
  - `review` text
  - `rating` (1–5 stars)
  - `date` of review
  - `app` name
  - `bank_code`
- Targeted a minimum of **400+ reviews per bank**, totaling over 1,200 reviews.

### 3. Preprocessing
- Remove duplicate reviews.
- Handle missing or incomplete data.
- Normalize dates to `YYYY-MM-DD` format.
- Save preprocessed data to CSV with the following columns: review, rating, date, bank, source
---

## 4. Visualizations & Analysis

### a) Ratings Distribution
- Shows the count of each star rating (1–5) across all reviews.
- Helps identify overall customer satisfaction.

### b) Reviews per Bank

- Displays the total number of reviews collected for each bank.
- Useful for comparing data coverage.

### c) Rating Count per Bank

- Shows the number of reviews for each star rating, grouped by bank.
- Highlights differences in customer satisfaction among banks.
## 5. How to Run
- clone repository
- create virtual environments
- run preprocessing_eda.ipynb
### 6.Dependencies
- pandas
- google-play-scraper
- tqdm
- numpy
- matplotlib
- seaborn
## 6. Notes
- Ensure you have stable internet when running the scraper.
- Frequent commits help track progress and revert changes if needed.
- Preprocessing ensures clean, normalized, and ready-to-analyze data.
- Visualizations help understand customer sentiment and review trends per bank.