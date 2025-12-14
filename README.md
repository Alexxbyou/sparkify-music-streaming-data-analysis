# Sparkify Music Streaming Data Analysis

## Key Files

* **Project Summary**: [Documents/ProjectSummary.md](Documents/ProjectSummary.md) - Methodology, analysis and key findings, MLOps approaches.
* **Notebooks**: notebooks are splitted into 3 main parts in [Scripts/](Scripts/):
  * `01_DataUnderstanding.ipynb`: Data understanding and initial exploration using ydata-profiling. [link](Scripts/01_DataUnderstanding.ipynb)
  * `02_DataPreprocessing.ipynb`: Data cleaning, preprocessing and feature engineering. [link](Scripts/02_DataPreprocessing.ipynb)
  * `03_EDAModelling.ipynb`: Exploratory Data Analysis (EDA), modelling and evaluation. [link](Scripts/03_EDAModelling.ipynb)
* **Rendered HTML Reports**: 
  * Data understanding report: [Output/data_understanding/Report.html](Output/data_understanding/Report.html)
  * EDA summary report: [Output/EDA/Summary/DataSummary.html](Output/EDA/Summary/DataSummary.html)
  * EDA modelling HTML report: [Scripts/html/03_EDAModelling.html](Scripts/html/03_EDAModelling.html)


## Project Structure

```
.
├── Data # Input data files, gitignored.
├── Documents
│   ├── ProjectSummary.md # main project summary document
│   └── Screenshots # folder for screenshots
├── Output
│   ├── data_proc_eng
│   │   ├── se_data_processed_pd.csv # output from jupyter notebook data processing
│   │   └── se_data_processed_prod.csv # output from production data processing
│   ├── data_understanding
│   │   └── Report.html # data understanding report using ydata-profiling, from 01_DataUnderstanding.ipynb
│   ├── EDA
│   │   └── Summary
│   │       └── DataSummary.html # EDA summary report using ydata-profiling, from 03_EDAModelling.ipynb
│   └── Modelling # modelling outputs
├── Production
│   ├── DataProcessing.py # ✅ rerunable
│   ├── Modelling.py #❗not rerunable due to the use of self-defined `dsmodelling` library
│   └── run.sh # bash script to run the production pipeline
├── README.md
├── Scripts
│   ├── 01_DataUnderstanding.ipynb # ✅ rerunable
│   ├── 02_DataPreprocessing.ipynb # ✅ rerunable
│   ├── 03_EDAModelling.ipynb #❗not rerunable due to the use of self-defined `dsmodelling` library
│   ├── deployment
│   │   ├── __pycache__
│   │   │   └── fastapi_app.cpython-312.pyc
│   │   ├── deployed_model.pkl
│   │   ├── fastapi_app.py
│   │   ├── requirements.txt
│   │   ├── streamlit_app.py
│   │   └── test_data.csv
│   └── html # folder for EDAModelling html outputs from 03_EDAModelling.ipynb
└── tests # unit tests for consistency between production and notebook outputs, gitignored
    └── test_autorun_migration_success.py
```