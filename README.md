# Recommendation System for Amazon Sports & Outdoor Products

Predicting Amazon Sports & Outdoors ratings and generating personalized top-10 product recommendations using large-scale review and metadata features.

## Goal

The goal of this project is to build a recommendation engine for Amazon’s Sports & Outdoors category by predicting user ratings and ranking products for personalized top-10 recommendations. The work includes review ingestion, metadata merging, feature engineering, sparsity and quality analysis, baseline models, matrix factorization, regression models, and a hybrid recommender.

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
- Ranking metrics including Precision@10, Recall@10, and nDCG@10
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
