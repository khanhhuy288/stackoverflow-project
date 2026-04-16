# Sprint 2 - Data Modeling

## Week 4-6

1. Feature engineering and transforming raw text into "model-ready" data, for example:
* Create a binary column for each technology (e.g., `has_python`, `has_C`, etc.)
* If necessary for high cardinal features, combine values into broader group
* If necessary, convert values into binned categories
* Create new features you find valuable for your analysis (e.g., `dev_score` based on how many different tools a developer uses)
* Merge all engineered features into a single modelling dataset. Do not forget to update data folder and add information on new features (`processed_data_dictionary.md`)
* Split into train (80%) and test (20%) for final evaluation

2. The modelling strategy balances interpretability with predictive performance:
* Linear Regression: baseline interpretability.
* LASSO Regression: feature selection and regularisation using k-fold cross-validation
* Random Forest Regressor: non‑linear relationships and interactions
* Gradient Boosting Machines (GBM): high‑performance ensemble modelling
* Is there another model you would like to fit? 

Linear models capture broad trends and provide interpretability. More complex models capture non‑linear effects. Tuning parameters helps find a better model in terms of model metrics.

3. Primary Metric
* Root Mean Squared Error (RMSE) measures typical prediction error in nightly price. Lower RMSE indicates better generalisation
* Mean Absolute Error (MAE)
* R² (explained variance)

The model with the best cross‑validated metric (RMSE, MAE or R²)  and stable performance across folds is selected as the final predictive engine.

### Data products
* Trained machine learning models
* All data manipulations and model tunings are scripted for reproducibility and uploaded to GitHub Repository
* Documentation includes reports, notebooks and pitch-presentation (slides in .pdf for 5-minute presentation of results achieved in Sprint 2) 
