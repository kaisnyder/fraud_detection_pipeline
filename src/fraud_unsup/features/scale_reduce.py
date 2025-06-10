"""
Numeric scaling + optional PCA.  Added `fit_minmax=True` for AE.
"""
from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

class ScalerReducer:
    def __init__(self, n_components:int|None=None, minmax:bool=False):
        self.std = StandardScaler()
        self.minmax = MinMaxScaler() if minmax else None
        self.pca = PCA(n_components=n_components) if n_components else None
        self._fitted=False

    def fit(self, X:pd.DataFrame):
        x = self.std.fit_transform(X)
        if self.minmax: x = self.minmax.fit_transform(x)
        if self.pca: self.pca.fit(x)
        self._fitted=True; return self

    def transform(self, X):
        assert self._fitted
        x = self.std.transform(X)
        if self.minmax: x = self.minmax.transform(x)
        if self.pca: x = self.pca.transform(x)
        return x

    def fit_transform(self,X):
        self.fit(X); return self.transform(X)
