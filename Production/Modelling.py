import pandas as pd
import pickle
from dsmodelling import stratifiedSampling, models, runModels
import numpy as np
import shap
import joblib
from pathlib import Path


RETRAIN = False
PROJECT_ROOT = Path(__file__).parent.parent  # Go up to project root
OUTPUT_FOLDER = PROJECT_ROOT / "Output"
MODELLING_FOLDER = OUTPUT_FOLDER / "Modelling"
DEPLOYMENT_FOLDER = PROJECT_ROOT / "Scripts" / "deployment"

# Create directories if they don't exist
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
MODELLING_FOLDER.mkdir(parents=True, exist_ok=True)
DEPLOYMENT_FOLDER.mkdir(parents=True, exist_ok=True)

INPUT_DATA_PATH = OUTPUT_FOLDER / "data_proc_eng" / "se_data_processed_prod.csv"
INPUT_DATA_PATH_ALT = OUTPUT_FOLDER / "data_proc_eng" / "se_data_processed_pd.csv"


def train_model():
    if INPUT_DATA_PATH.exists():
        se_data = pd.read_csv(INPUT_DATA_PATH)
        print("[train_model] Loaded data from production data processing pipeline.")
    elif INPUT_DATA_PATH_ALT.exists():
        se_data = pd.read_csv(INPUT_DATA_PATH_ALT)
        print("[train_model] Loaded data from Jupyter Notebook output.")
    else:
        print("[train_model] No processed data found..")
        return
    print(f"[train_model]Data loaded, shape: {se_data.shape}")

    se_data.churn.value_counts()
    se_data = se_data.set_index('userId')
    exclude_cols = ['cancel_date','start_date','end_date','has_activity','psn_city','psn_state']
    se_data_mdl = se_data[se_data['has_activity']].drop(columns=exclude_cols).copy()
    se_data_mdl['churn'] = se_data_mdl['churn'].astype(int)

    model_ready_data = pd.get_dummies(se_data_mdl)
    model_ready_data = stratifiedSampling(model_ready_data,stratify_col='churn', test_size=0.2, random_seed=42)
    train = model_ready_data[model_ready_data.train_ind == True].drop(columns = ['train_ind'])
    test = model_ready_data[model_ready_data.train_ind == False].drop(columns = ['train_ind'])
    y_train = train[['churn']].reset_index(drop=True)
    y_test = test[['churn']].reset_index(drop=True)
    X_train = train.drop(columns='churn')
    X_test = test.drop(columns='churn')

    mdl = models['lgb']
    model_sel_data = X_train
    model_saved_path = MODELLING_FOLDER / "model_sel_dict.pickle"
    model_sel_dict = dict()
    model_sel_dict['lgb'] = runModels(model_sel_data.reset_index(drop=True), y_train, mdl, model_sel_data)

    shap_save_path = MODELLING_FOLDER / "shap_values.pickle"
    shap_model = model_sel_dict['lgb'].model  # Access model attribute from ModelResult

    if shap_save_path.exists() and not RETRAIN:
        with open(shap_save_path, 'rb') as f:
            shap_values = pickle.load(f)
    else:
        shap.initjs()
        explainer = shap.Explainer(shap_model)
        shap_values = explainer(X_test)
        with open(shap_save_path, 'wb') as f:
            pickle.dump(shap_values, f)

    deployed_model_path = DEPLOYMENT_FOLDER / "deployed_model.pkl"
    with open(deployed_model_path, "wb") as f:
        joblib.dump(model_sel_dict['lgb'].model, f)  # Access model attribute from ModelResult

    test_data_path = DEPLOYMENT_FOLDER / "test_data.csv"
    X_test.head().to_csv(test_data_path, index=False)


if __name__ == "__main__":
    train_model()