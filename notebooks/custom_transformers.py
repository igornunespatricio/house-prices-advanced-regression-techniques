import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class ReplaceNoFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mask = X == self.fill_value
        X[mask] = 0
        X[~mask] = 1
        return X


class SelectLowCorrelationFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold_target_corr=0.5, threshold_feature_corr=0.5):
        self.threshold_target_corr = threshold_target_corr
        self.threshold_feature_corr = threshold_feature_corr
        self.selected_features = []
        self.remove_cols = []

    def fit(self, X, y=None):
        # calculate correlation of all features with the target variable
        # sort them by biggest correlation with target variable and store these features in a list
        if y is None:
            raise ValueError("Target variable must be specified.")
        
        # Calculate Pearson correlation between features and target variable and sort 
        X = pd.DataFrame(X)
        X['target'] = y
        self.remove_cols.append('target')
        correlations = X.corr().sort_values(by='target', ascending=False)
        correlations = correlations[np.abs(correlations['target']) > self.threshold_target_corr]
        # Calculate Pearson correlation matrix for all features
        features_correlation = correlations.iloc[1:,:-1]
        # drop the ones with high correlation
        for feature in features_correlation.index:
            if feature not in self.remove_cols:
                # print(feature)
                feat_corr_temp = features_correlation.drop(feature, axis=1).loc[feature].sort_values(ascending=False)
                feature_mask = np.abs(feat_corr_temp) > self.threshold_feature_corr
                self.remove_cols.extend(feat_corr_temp[feature_mask].index.tolist())
                # print(self.remove_cols)
        self.selected_features = [col for col in X.columns.tolist() if col not in self.remove_cols]
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X)
        if self.selected_features is None:
            raise ValueError("Must call fit() before transform()")
        return X.loc[:, self.selected_features]