# Project Proposal: 2025 Stack Overflow Developer Survey

#ReDI  #26s_dcp_data_circle #data_project

## Objective
This project uses a structured, end-to-end analytical workflow to investigate global developer trends using the [Stack Overflow 2025 Developer Survey](https://survey.stackoverflow.co/) dataset. It ensures statistical rigour, reproducibility, and alignment with industry benchmarks to provide actionable insights into the evolving professional landscape, identify high-value skill sets, and support data-driven career positioning. 

## Data Source and Scope
This project uses the [2025 Stack Overflow Developer Survey](https://survey.stackoverflow.co/) dataset, a comprehensive primary source representing the global software development ecosystem. [Stack Overflow](https://stackoverflow.com/) conducted a structured online survey targeting developers of all experience levels and specialisations.

## Case Study: Predictive Modelling of Developer Compensation

This case study examines the complex factors that influence global developer salaries using advanced regression techniques on the [2025 Stack Overflow](https://survey.stackoverflow.co/) dataset. The primary objective is to quantify the predictive power of various indicators — age, formal education level, professional experience, developer roles, employment, work environment, geographic location, etc. — on annual compensation. The study will serve as a data-driven benchmark for students to assess their career prospects and negotiate compensation based on objective global market trends.

## 9‑Week Project Schedule

### Week 1 — Project Kickoff & Data Acquisition
* Form teams (2-3 students) and assign roles
* Identify required datasets
* Get raw data and check if it is complete
* Set up project environment, GitHub repository, and documentations

### Week 2 — Data Cleaning & Preparation
* Choose required features to answer your questions
* Clean and standardise raw fields
* Handle missing values, duplicates, and inconsistent formats
* Normalise numeric fields and parse text‑based features
* Produce a first “clean dataset” for exploration

### Week 3 — Exploratory Data Analysis (EDA)
* Analyse distributions, outliers, and skewness
* Explore relationships between features
* Generate geographic visualisations
* Document insights that inform feature engineering and modelling

### Week 4 — Feature Engineering
* Create new features (programming languages, frameworks, tools)
* Finalise modelling dataset with all engineered features
* Split the final dataset into training (75-80% observations) and test (25-30% observations) datasets

### Week 5 — Baseline Modelling 
* Train baseline models (e.g., linear regression) for interpretability
* Evaluate performance metrics
* Identify feature gaps or data issues requiring refinement

### Week 6 — Advanced Modeling & Hyperparameter Tuning
* Train ensemble models (e.g., Random Forest, Gradient Boosting, CatBoost)
* Tune hyperparameters using k‑fold cross‑validation
* Compare models on cross‑validated performance metrics
* Select top candidates for final evaluation

### Week 7 — Model Evaluation & Diagnostics
* Evaluate final models on the test set
* Interpret model results to identify which features have the strongest impact (SHAP or similar techniques)
* Create visualizations showing feature importance
* Document model strengths, limitations, and interpretability findings

### Week 8 — Dashboard Development & Insights Packaging
* Build interactive dashboard (Streamlit)
* Include maps visualisations, predictions, feature importance

### Week 9 — Final Review, Documentation & Delivery
* Compile full project report (methodology, results, recommendations)
* Deliver model, data dictionary, and reproducibility notes
* Give next‑phase recommendations (e.g., deployment, monitoring, further improvements)
