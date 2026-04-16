# Project Structure 


```
├── data/
│   ├── data_dictionary.md   # Metadata on an initial raw dataset
    ├── processed_data_dictionary.md  # Metadata on data prepared for modelling
│   └── .gitignore           # (Contains: *.csv, *.zip)
├── notebooks/               # Jupyter Notebooks (numbered sequentially)
├── reports/                 # Written report & Presentation slides
├── src/                     # Modular Python scripts
├── requirements.txt         # list of all the specific Python libraries and their versions 
├── workflow.md              # track the development lifecycle
└── README.md                # objective, setup, and key findings

```

Standardize your repository to ensure reproducibility:

* data/: 
    * data_dictionary.md including download link and a brief description of the 2025 survey 
    * pdf file of 2025 Developer Survey with list of all 62 questions
* notebooks/: Sequentially numbered research files (01_cleaning_survey_data_author.ipynb, 02_eda_salary_analysis_author.ipynb, 03_residual_analysis_model_v1_author.ipynb, etc.)
* src/: Modular, reusable production code in Python 
    * e.g. `download_data.py` script that fetches the dataset from the Stack Overflow URL
* README.md: The project's description (objective, setup, and key findings).
