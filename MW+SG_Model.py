import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error
import json
from flask import Flask, make_response, request, jsonify, abort

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import tensorflow as tf
keras = tf.keras
from lightgbm import LGBMRegressor
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)


def meta_model(X):


    # Load Models
    ###########################

    path_sg = os.getcwd() + r'\db\SG_models'
    path_mw = os.getcwd() + r'\db\MW_models'

    filename_xgboost_mw = path_mw + r'\XGBoost_model.sav'
    filename_xgboost_sg = path_sg + r'\XGBoost_model.sav'
    filename_GB_mw = path_mw + r'\GB_model.sav'
    filename_GB_sg = path_sg + r'\GB_model.sav'
    filename_RF_mw = path_mw + r'\RF_model.sav'
    filename_RF_sg = path_sg + r'\RF_model.sav'
    filename_LGBM_mw = path_mw + r'\LGBM_model.sav'
    filename_LGBM_sg = path_sg + r'\LGBM_model.sav'
    filename_NN_reg_mw = path_mw + r'\NN_reg_model.h5'
    filename_NN_reg_sg = path_sg + r'\NN_reg_model.h5'
    filename_ensemble_mw = path_mw + r'\ensemble_model.h5'
    filename_ensemble_sg = path_sg + r'\ensemble_model.h5'

    with open(filename_xgboost_mw, 'rb') as pickle_file:
        xgb_MW_model = pickle.load(pickle_file)
    with open(filename_xgboost_sg, 'rb') as pickle_file:
        xgb_SG_model = pickle.load(pickle_file)
    with open(filename_GB_mw, 'rb') as pickle_file:
        gb_MW_model = pickle.load(pickle_file)
    with open(filename_GB_sg, 'rb') as pickle_file:
        gb_SG_model = pickle.load(pickle_file)
    with open(filename_RF_mw, 'rb') as pickle_file:
        rf_MW_model = pickle.load(pickle_file)
    with open(filename_RF_sg, 'rb') as pickle_file:
        rf_SG_model = pickle.load(pickle_file)
    with open(filename_LGBM_mw, 'rb') as pickle_file:
        lgbm_MW_model = pickle.load(pickle_file)
    with open(filename_LGBM_sg, 'rb') as pickle_file:
        lgbm_SG_model = pickle.load(pickle_file)

    nn_reg_mw_model = keras.models.load_model(filename_NN_reg_mw)
    nn_reg_sg_model = keras.models.load_model(filename_NN_reg_sg)

    ensemble_mw_model = keras.models.load_model(filename_ensemble_mw)
    ensemble_sg_model = keras.models.load_model(filename_ensemble_sg)


    #  Model Archetecture
    ###########################

    xgb_MW_train = xgb_MW_model.predict(X)
    gb_MW_train = gb_MW_model.predict(X)
    rf_MW_train = rf_MW_model.predict(X)
    lgbm_MW_train = lgbm_MW_model.predict(X)
    nn_MW_train = nn_reg_mw_model.predict(X).flatten()

    xgb_SG_train = xgb_SG_model.predict(X)
    gb_SG_train = gb_SG_model.predict(X)
    rf_SG_train = rf_SG_model.predict(X)
    lgbm_SG_train = lgbm_SG_model.predict(X)
    nn_SG_train = nn_reg_sg_model.predict(X).flatten()

    Ensemble_MW_df = pd.DataFrame({
        'xgb': xgb_MW_train,
        'gb': gb_MW_train,
        'rf': rf_MW_train,
        'lgbm': lgbm_MW_train,
        'nn': nn_MW_train
    })

    Ensemble_SG_df = pd.DataFrame({
        'xgb': xgb_SG_train,
        'gb': gb_SG_train,
        'rf': rf_SG_train,
        'lgbm': lgbm_SG_train,
        'nn': nn_SG_train
    })

    ensemble_MW_prediction = ensemble_mw_model.predict(Ensemble_MW_df).flatten()
    ensemble_SG_prediction = ensemble_sg_model.predict(Ensemble_SG_df).flatten()

    return ensemble_MW_prediction, ensemble_SG_prediction
    



@app.route("/", methods =['POST'])
def GPA_2103():
    if request.method == 'POST':

        input_data = request.get_json()

        #X = input_data
        X = input_data

        X = pd.DataFrame(X)
        y = X.iloc[:,0]
        X = X.drop(['sample', 'N2'], axis=1)
        X = X.astype('float')

        mw_pred, sg_pred = meta_model(X)
 
        json_out = {
            "sample": y.tolist(),
            "MW Pressurized": mw_pred.tolist(),
            "SG Pressurized": sg_pred.tolist()
        }

        print(mw_pred)
        print(sg_pred)
        return jsonify(json_out)


        
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=4999)

