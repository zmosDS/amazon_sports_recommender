# Recommendation System for Amazon Sports & Outdoor Products

Predicting Amazon Sports & Outdoors ratings and generating personalized top-10 product recommendations using large-scale review and metadata features.

## Goal

The goal of this project is to build a recommendation engine for Amazon’s Sports & Outdoors category by predicting user ratings and ranking products for personalized top-10 recommendations. The workflow includes dataset ingestion, feature engineering, sparsity analysis, baseline models, matrix factorization, regression models, and a hybrid method that delivers the strongest predictive performance.

## Results

| Model                       | RMSE     | MAE     | Precision@10 | Recall@10 | nDCG@10 |
|-----------------------------|----------|---------|---------------|-----------|---------|
| Hybrid (0.6 SVD + 0.4 Ridge)| **0.7505** | **0.5540** | 0.8388 | **0.9977** | **0.9981** |
| SVD Matrix Factorization    | 1.0303  | 0.7129 | **0.8389** | 0.9970 | 0.9753 |
| User + Item Bias            | 1.0846  | 0.7125 | 0.8390 | 0.9971 | 0.9755 |
| Ridge Regression            | 1.0851  | 0.8132 | 0.8381 | 0.9970 | 0.9739 |
| Global Average Baseline     | 1.0951  | 0.8325 | 0.8388 | 0.9969 | 0.9680 |

**Highlights**

- Hybrid model improved RMSE by **31.5%** over the global-average baseline  
- Achieved near-perfect **nDCG@10 = 0.9981**, the strongest indicator of ranking quality  
- Extremely high Recall@10 across models due to rating distribution characteristics  

## Built With

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-Learn
- Surprise (SVD)
- Jupyter Notebooks
- CSV / JSONL Amazon datasets

## Features

- End-to-end pipeline built from Amazon 5-core reviews, full reviews, and product metadata  
- Cleaning, merging, and flattening nested item attributes (brand, details, images, videos, categories)
- Creation of user-level and item-level features (review counts, rating deviation, temporal fields, text length)
- Category hierarchy extraction and product name parsing
- Exploratory analysis of ratings, distributions, sparsity, verified purchases, helpful votes, and pricing patterns
- Dense subset creation for comparison (users and items with 5+ reviews)
- Baseline models: global average and user + item bias
- Advanced models: SVD matrix factorization, Ridge regression, and a hybrid recommender
- Ranking metrics including: RMSE & nDCG@10
- Final top-10 product recommendations generated from predicted ratings

## Files
```
amazon_sports_recommender/
├── project_deliverables/
│ └── amazon_sports_recommender_fullNB.ipynb  # Complete workflow (EDA, cleaning, modeling, ranking)
├── workbook.html                             # HTML export of final deliverable
├── data_processing.ipynb                     # Data ingestion, merging, feature engineering
├── eda.ipynb                                 # Exploratory analysis, sparsity, quality indicators
├── modeling.ipynb                            # Baseline, SVD, Ridge, hybrid models + ranking metrics
└── README.md                                 # Project overview and usage details
```
## Data Source

[McAuley et al. (2024) "Bridging Language and Items for Retrieval and
Recommendation" Amazon Sports & Outdoors Reviews](https://amazon-reviews-2023.github.io)

Amazon Sports & Outdoors review and metadata files:
- `Sports_and_OutDoors_5core.csv`
- `Sports_and_OutDoors.jsonl`
- `meta_Sports_and_OutDoors.jsonl`

## Author

Zack Mosley
