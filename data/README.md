# Data

The data is **not stored in this repository**. The notebook downloads it at run time.

## Source

- **Dataset:** Kaggle, *Formula 1 Dataset with Weather and Tyre Features*, `navenkumar1998/formula-1-dataset-with-weather-and-tyre-features`
- **Page:** https://www.kaggle.com/datasets/navenkumar1998/formula-1-dataset-with-weather-and-tyre-features
- **File used:** `f1_all.parquet`. The notebook's saved output shows 69,230 rows × 30 columns, three seasons from 2021: race and lap results, tyre compound, tyre life, stint and weather.

## How to get it

The first notebook cell does this:

```python
import kagglehub
path = kagglehub.dataset_download("navenkumar1998/formula-1-dataset-with-weather-and-tyre-features")
# then: pandas.read_parquet(os.path.join(path, "f1_all.parquet"))   # needs pyarrow
```

`kagglehub` stores the files in its own cache (for example `~/.cache/kagglehub/...`), not in this folder. If Kaggle asks for authentication, sign in to Kaggle and create an API token (Kaggle → Settings → API), then follow the `kagglehub` documentation. Never commit the token or the downloaded files.

## Terms of use

Check the licence and terms on the Kaggle dataset page before using or sharing the data. Formula 1 timing and results data may also be subject to the terms of its original sources. This repository does not redistribute the data.
