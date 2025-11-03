```markdown
# Airline On-Time Performance & Passenger Experience (Part A)

This project analyzes the 2019 US DOT flight delays dataset to explore airline on-time performance and passenger experience. The analysis includes data cleaning, exploratory data analysis (EDA), hypothesis testing, and regression modeling.

## Dataset
Source: Kaggle — "Airline Delay and Cancellation Data 2019"  
URL: https://www.kaggle.com/datasets/usdot/flight-delays  
File used: `flights.csv`

## Files in this repository
- `mini_project_partA.py`: Main analysis script.
- `results/` (created after running the script):
  - `cleaned_data/clean_flights.csv` — cleaned dataset used for analysis.
  - `plots/` — saved figures (histogram, boxplots, heatmap, regression plots).
  - `summary_reports/airline_summary.txt` — short textual summary of results.

## How to run
1. (Optional) If running in Colab or locally and you want the script to download the data automatically, add your Kaggle credentials:

```bash
mkdir -p ~/.kaggle
echo '{"username":"YOUR_KAGGLE_USERNAME","key":"YOUR_KAGGLE_API_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

2. Install required Python packages if not already present:

```bash
python -m pip install pandas numpy seaborn matplotlib scipy scikit-learn kaggle
```

3. Run the script (ensure `flights.csv` is present or kaggle credentials are configured):

```bash
python mini_project_partA.py
```

4. After completion, check the `results/` folder for outputs.

## Expected outputs
- Cleaned CSV: `results/cleaned_data/clean_flights.csv`
- Plots: saved under `results/plots/` (e.g., `delay_distribution.png`, `airline_comparison_boxplot.png`, `correlation_heatmap.png`, `regression_fit_distance.png`)
- Summary: `results/summary_reports/airline_summary.txt`

## Notes
- The script samples large datasets for plotting and model fitting to keep runtime reasonable.
- For production-grade modeling, include additional features such as scheduled time-of-day, weather, and airport congestion.

## Contact
Student: Manish Kumar Singh  
Course: BAD702 — Statistical Machine Learning for Data Science
```
