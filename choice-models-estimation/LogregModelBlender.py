import math
import numpy as np
import pandas as pd
import dill

import pyspark
from pyspark.sql import functions as f
from pyspark.sql.types import *


import ChoiceModelEstimator


class LogregModelBlender:
    def __init__(self, outer_feats, model_names, model_feats, 
                 regularization_coef=0.0,
                 # stop_criterion, opt_algorithm, ...
                 model_mfeat_col=None):
        self.outer_feats: list = outer_feats
        self.model_names: list = model_names
        self.model_feats: list = model_feats
        self.regularization_coef = regularization_coef
        self.coefs = {
            'outer_feats': {
                model: {feat: 0.0 for feat in outer_feats}
                for model in model_names
            },
            'model_feats_common': {
                mfeat: 0.0
                for mfeat in model_feats
            },
            'model_feats_individual': {
                model: {
                    mfeat: 0.0
                    for mfeat in model_feats
                }
                for model in model_names
            },    
        }
        self.model_mfeat_col = model_mfeat_col


    def fit(self, X, target_col:str):
        """
        Fits coefficients of the blender
        :param X: dataframe (pandas or spark) with all feats and blender targets
        :target_col: name of blender target column
        """
        if isinstance(X, pd.DataFrame):
            return self._fit_pandas(X, target_col=target_col)
        elif isinstance(X, pyspark.sql.DataFrame):
            return self._fit_spark(X, target_col=target_col)
        else:
            raise TypeError(f"Unsupported type: {type(X)}")


    def predict(self, X, pred_col=None, softmax=True, pred_label_col=None):
        """
        Predicts scores of each model and (optionally) determines the best model
        :param X: dataframe (pandas or spark) with all blender feats
        :pred_col: lambda: model name -> column name for predicted score of the model
        :softmax: if true, convert scores to probabilities (using softmax transformation)
        :pred_label_col: column for name of predicted best model. 
        If None, the best model is not determined, only model scores are calculated
        """

        if pred_col is None:
            pred_col = lambda model: f'blender_pred_{model}'
        preds = [pred_col(model) for model in self.model_names]

        if isinstance(X, pd.DataFrame):
            X = self._predict_pandas(X, pred_col=pred_col)
            if softmax: 
                X = self._apply_softmax_pandas(X, preds)
            if pred_label_col:
                X = self._append_pred_label_pandas(X, pred_label_col, proba_col=pred_col)
        elif isinstance(X, pyspark.sql.DataFrame):
            X = self._predict_spark(X, pred_col=pred_col)
            if softmax:
                X = self._apply_softmax_spark(X, preds)
            if pred_label_col:
                X = self._append_pred_label_spark(X, pred_label_col, proba_col=pred_col)
        else:
            raise TypeError(f"Unsupported type: {type(X)}")

        return X


    def save(self, path: str):
        with open(path, 'wb') as file:
            ser = self._serialize()
            dill.dump(ser, file)
    
    
    @classmethod
    def load(self_class, path: str):
        with open(path, 'rb') as file:
            ser = dill.load(file)
            return self_class._deserialize(ser)


    def _fit_pandas(self, X, target_col:str):
        # Since the fit is run on numpy, prepare the arrays from the table
        outer_features = np.expand_dims(X[self.outer_feats].values, axis=1)
        
        model_mfeat_render = sum([
            [
                self.model_mfeat_col(model, mfeat)
                for model in self.model_names
            ]
        for mfeat in self.model_feats
        ], [])
        model_features_shape = -1, len(self.model_names), len(self.model_feats)
        model_features = X[model_mfeat_render].values.reshape(model_features_shape)
        
        extracted_target = X[target_col].values.reshape((-1, 1))
        targets = np.hstack([extracted_target == model for model in self.model_names]).astype(float)
        
        estimator = ChoiceModelEstimator.ChoiceModelEstimator(
            data=(outer_features, model_features, targets),
#             initial_guess=None,
#             learning_rate=0.1,
            L2_regularization=self.regularization_coef,
        )
        estimator.fit()
        
        self.coefs = {
            'outer_feats': {
                model: {
                    feat: estimator.current_theta[0][j, k_o]
                    for k_o, feat in enumerate(self.outer_feats)
                }
                for j, model in enumerate(self.model_names)
            },
            'model_feats_common': {
                mfeat: estimator.current_theta[1][0, k_m]
                for k_m, mfeat in enumerate(self.model_feats)
            },
            'model_feats_individual': {
                model: {
                    mfeat: estimator.current_theta[2][j, k_m]
                    for k_m, mfeat in enumerate(self.model_feats)
                }
                for j, model in enumerate(self.model_names)
            },    
        }

  
    def _fit_spark(self, X, target_col:str):
        raise NotImplementedError()
 

    def _predict_pandas(self, X, pred_col=None):
        X = X.copy()
        for model in self.model_names:
            pred = pred_col(model)
            X[pred] = 0.0
            for feat, coef in self.coefs['outer_feats'][model].items():
                X[pred] += coef * X[feat]

            for mfeat in self.model_feats:
                coef1 = self.coefs['model_feats_common'][mfeat]
                coef2 = self.coefs['model_feats_individual'][model][mfeat]
                X[pred] += (coef1 + coef2) * X[self.model_mfeat_col(model, mfeat)]
        return X


    def _predict_spark(self, X, pred_col=None):
        preds = []
        for model in self.model_names:
            pred = f.lit(0.0)
            for feat, coef in self.coefs['outer_feats'][model].items():
                pred += float(coef) * f.col(feat)
            
            for mfeat in self.model_feats:
                coef1 = self.coefs['model_feats_common'][mfeat]
                coef2 = self.coefs['model_feats_individual'][model][mfeat]
                pred += float(coef1 + coef2) * f.col(self.model_mfeat_col(model, mfeat))
            preds.append(pred.alias(pred_col(model)))
        X = X.select(*X.columns, *preds)
        return X


    def _apply_softmax_pandas(self, X, cols):
        # For each row, substract max prediction score to avoid numerical problems
        X = X.copy()
        X[cols] = X[cols].sub(X[cols].max(axis=1), axis=0)
        X[cols] = np.exp(X[cols])
        X[cols] = X[cols].div(X[cols].sum(axis=1), axis=0)
        return X


    def _apply_softmax_spark(self, X, cols):
        @f.udf(ArrayType(DoubleType()))
        def udf_softmax(array):
            max_val = max(array)
            # Substract max prediction score to avoid numerical problems
            array = [(val - max_val) for val in array]
            array = [math.exp(val) for val in array]
            sum_exp = sum(array)
            array = [val / sum_exp for val in array]
            return array

        X = X.withColumn('softmaxed', udf_softmax(f.array(*[f.col(c) for c in cols])))
        X = X.select(
            *[c for c in X.columns if (c not in cols) and (c != 'softmaxed')],
            *[f.col('softmaxed')[i].alias(c) for i, c in enumerate(cols)]
        )
        return X


    def _append_pred_label_pandas(self, X, pred_label_col, proba_col=None):
        all_proba_cols = [proba_col(model) for model in self.model_names]
        pred_label_idx = X[all_proba_cols].values.argmax(axis=1)
        X[pred_label_col] = [self.model_names[idx] for idx in pred_label_idx]
        return X

    def _append_pred_label_spark(self, X, pred_label_col, proba_col=None):
        model_names = self.model_names
        @f.udf(StringType())
        def udf_best_label(array):
            if len(array) == 0:
                return None
            max_idx, max_val = 0, array[0]
            for i in range(1, len(array)):
                if array[i] > max_val:
                    max_idx, max_val = i, array[i]
            return model_names[max_idx]
        all_proba_cols = [proba_col(model) for model in self.model_names]
        X = X.withColumn(pred_label_col, udf_best_label(f.array(*[all_proba_cols])))
        return X


    def _serialize(self):
        ser = {
            'outer_feats': self.outer_feats,
            'model_names': self.model_names,
            'model_feats': self.model_feats,
            'regularization_coef': self.regularization_coef,
            'coefs': self.coefs,
            'model_mfeat_col': self._model_mfeat_col,
        }
        return ser


    @classmethod
    def _deserialize(self_class, ser):
        coefs = ser.pop('coefs')
        outer_feats = ser.pop('outer_feats')
        model_names = ser.pop('model_names')
        model_feats = ser.pop('model_feats')
        blender = self_class(outer_feats, model_names, model_feats, **ser)
        blender.coefs = coefs
        return blender


    @property
    def model_mfeat_col(self):
        if self._model_mfeat_col is None:
            return lambda model, mfeat: f'{model}_{mfeat}'
        else:
            return self._model_mfeat_col
    
    @model_mfeat_col.setter
    def model_mfeat_col(self, value):
        self._model_mfeat_col = value
