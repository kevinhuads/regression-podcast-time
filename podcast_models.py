# --- Standard lib
import math
import os

# --- Third-party
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_halving_search_cv  
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.model_selection import (
    GridSearchCV,
    HalvingRandomSearchCV,
    KFold,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from interpret.glassbox import ExplainableBoostingRegressor
from pytorch_tabnet.tab_model import TabNetRegressor

# --- Torch
import torch
import torch.nn as nn
import torch.optim as optim

# --- Project
from podcast_functions import *  # provides SEED, etc.
from podcast_hyperparameters import GRIDS, tabnet_params


# ==============================
# Global config
# ==============================
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
N_ITER = 50
FACTOR = 3
SCORING = "neg_mean_squared_error"
VERBOSE = 1
ENET_ITER = 25000
SVR_ITER = 100000
SVR_TOL = 1e-3

# ==============================
# Utilities
# ==============================
def search_kw(n_jobs = -1, include_seed = True, 
              halving=False, n_samples = 50000, factor=FACTOR, min_frac=0.10, **override):
    """
    Common kwargs for sklearn *SearchCV objects.
    Adds halving-specific keys when halving=True.
    """
    kw = {"scoring": SCORING, "cv": cv, "verbose": VERBOSE, "n_jobs": n_jobs}
    if include_seed:
        kw["random_state"] = SEED
    if halving:
        kw.update(
            resource="n_samples",
            max_resources=n_samples,
            min_resources=int(n_samples * min_frac),
            factor=factor,
        )
    kw.update(override)
    return kw


def expected_fits(search):
    """
    Estimate total fits for a SearchCV object to drive tqdm progress.
    """
    _cv = search.cv
    if isinstance(search, GridSearchCV):
        grid = search.param_grid
        n_points = int(np.prod([len(v) for v in grid.values()]))
        return n_points * _cv.get_n_splits()

    if isinstance(search, RandomizedSearchCV):
        return search.n_iter * _cv.get_n_splits()

    if isinstance(search, HalvingRandomSearchCV):
        n_rounds = math.floor(math.log(search.n_candidates, search.factor)) + 1
        cand_per_round = [max(1, math.floor(search.n_candidates / (search.factor**r))) for r in range(n_rounds)]
        return _cv.get_n_splits() * sum(cand_per_round)

    return None


def fit_model(search, X, y, **fit_kwargs):
    """
    Fit a SearchCV with a progress bar and persist it with joblib.
    """
    total_fits = expected_fits(search)
    with tqdm_joblib(tqdm(total=total_fits)):
        search.fit(X, y, **fit_kwargs)
    return search


def write_predictions(df, model, features, target, path, pretransform=False):
    """
    Write a CSV with id + predicted target.
    """
    output = df[["id"]].copy()
    if pretransform:
        output[target] = model.predict(preprocessor.transform(df[features]))  
    else:
        output[target] = model.predict(df[features])
    output.to_csv(path, index=False)


# ==============================
# Transformers
# ==============================
class CustomCapper(BaseEstimator, TransformerMixin):
    """
    Clip dataframe columns to [min_, max_] (when provided).
    Equivalent to the original np.where-per-column implementation.
    """
    def __init__(self, min_=None, max_=None):
        self.min_ = min_
        self.max_ = max_

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xc = X.copy()
        # pandas.DataFrame.clip matches original columnwise logic
        Xc[:] = Xc.clip(lower=self.min_, upper=self.max_, axis="columns")
        return Xc

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else getattr(self, "feature_names_in_", None)


class QuantileLimiter(BaseEstimator, TransformerMixin):
    """
    Cap values above the given per-column quantile (nan-aware).
    """
    def __init__(self, quantile=0.99):
        self.quantile = quantile
        self.thresholds_ = {}

    def fit(self, X, y=None):
        self.thresholds_ = {col: np.nanquantile(X[col], self.quantile) for col in X.columns}
        return self

    def transform(self, X):
        Xc = X.copy()
        for col, thr in self.thresholds_.items():
            Xc[col] = np.where(Xc[col] > thr, thr, Xc[col])
        return Xc

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else getattr(self, "feature_names_in_", None)


# ==============================
# Preprocessing
# ==============================
def make_preprocessor(X_num, X_cat, 
                      onehot=False, impute=False, sparse_output=True, scaler=False, ordinal = False):
    """
    Build the ColumnTransformer (and optional MICE imputer) used across models.
    """
    # Numeric blocks
    num_blocks = []
    if scaler:
        num_blocks.extend([
            ("len", Pipeline([
                ("cap", QuantileLimiter(quantile=0.99)),
                ("scale", StandardScaler()),
            ]), ["Episode_Length_minutes"]),
            ("gue", Pipeline([
                ("cap", CustomCapper(min_=0, max_=100)),
                ("scale", StandardScaler()),
            ]), ["Guest_Popularity_percentage", "Host_Popularity_percentage"]),
            ("ads", Pipeline([
                ("cap", CustomCapper(min_=0, max_=3)),
                ("scale", StandardScaler()),
            ]), ["Number_of_Ads"]),
        ])
    else:
        num_blocks.extend([
            ("len", QuantileLimiter(quantile=0.99), ["Episode_Length_minutes"]),
            ("gue", CustomCapper(min_=0, max_=100), ["Guest_Popularity_percentage", "Host_Popularity_percentage"]),
            ("ads", CustomCapper(min_=0, max_=3), ["Number_of_Ads"]),
        ])

    # Categorical
    if onehot:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)
        ohe = ohe.set_output(transform="default" if sparse_output else "pandas")
        cat_step = ("cat", ohe, X_cat)
        remainder_kw = "passthrough"
    else:
        if ordinal:
            cat_step = ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), X_cat)
        else:
            cat_step = ("cat", "passthrough", X_cat)
        remainder_kw = "drop"

    transformers = [*num_blocks, cat_step]
    core = ColumnTransformer(
        transformers=transformers,
        remainder=remainder_kw,
        verbose_feature_names_out=False,
    )

    if not impute:
        return core

    mice_block = ColumnTransformer(
        transformers=[
            ("mice",
             IterativeImputer(estimator=BayesianRidge(), max_iter=100, random_state=SEED)
             .set_output(transform="pandas"),
             X_num),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    return Pipeline([
        ("impute", mice_block),
        ("prep", core.set_output(transform="default")),
    ])



# ==============================
# Search builders
# ==============================
def _pipe(preprocessor, step_name, estimator):
    """Build a Pipeline with a preprocessing step and a named estimator step."""
    return Pipeline([("pre", preprocessor), (step_name, estimator)])

def _rand_search(pipe, grid_key, n_iter, n_jobs):
    """Create a RandomizedSearchCV over GRIDS[grid_key] with shared search defaults."""
    return RandomizedSearchCV(
        estimator=pipe,
        param_distributions=GRIDS[grid_key],
        n_iter=n_iter,
        **search_kw(n_jobs=n_jobs),
    )

def _grid_search(pipe, grid_key, n_jobs):
    """Create a GridSearchCV over GRIDS[grid_key] with shared search defaults."""
    return GridSearchCV(
        estimator=pipe,
        param_grid=GRIDS[grid_key],
        **search_kw(n_jobs=n_jobs, include_seed=False),
    )

def _halving_search(pipe, grid_key, n_candidates, X, factor, n_jobs):
    """Create a HalvingRandomSearchCV over GRIDS[grid_key] using sample-based resources."""
    return HalvingRandomSearchCV(
        estimator=pipe,
        param_distributions=GRIDS[grid_key],
        n_candidates=n_candidates,
        **search_kw(n_jobs=n_jobs, halving=True, n_samples=len(X), factor=factor),
    )

# ==============================
# Model runners
# ==============================

# --- Linear
def run_ElasticNet(X, y, preprocessor, n_jobs=-1):
    pipe = _pipe(preprocessor, "enet", ElasticNet(random_state=SEED, max_iter=ENET_ITER))
    search = _grid_search(pipe, "enet", n_jobs=n_jobs)
    fit_model(search, X, y)
    return search

# --- Gradient Boosting
def run_lightGBM(X, y, preprocessor, n_iter=N_ITER, n_jobs=-1):
    pipe = _pipe(preprocessor, "lgbm", LGBMRegressor(random_state=SEED, verbosity=-1))
    search = _rand_search(pipe, "lgbm", n_iter=n_iter, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search

def run_XGBoost(X, y, preprocessor,n_iter=N_ITER, n_jobs=-1):
    pipe = _pipe(preprocessor, "xgb", XGBRegressor(
        enable_categorical=True,
        objective="reg:squarederror",
        tree_method="hist",
        device="cuda",
        n_jobs=-1,
        random_state=SEED,
    ))
    search = _rand_search(pipe, "xgb", n_iter=n_iter, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search


def run_CatBoost(X, y, preprocessor, X_cat, n_iter=N_ITER, n_jobs=4):
    pipe = _pipe(preprocessor, "cat", CatBoostRegressor(
        loss_function="RMSE",
        random_state=SEED,
        verbose=False,
        allow_writing_files=False,
        cat_features=X_cat,
    ))
    search = _rand_search(pipe, "cat", n_iter=n_iter, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search


def run_HGB(X, y, preprocessor, n_candidates=N_ITER, factor=FACTOR, n_jobs=-1):
    pipe = _pipe(preprocessor, "hgb", HistGradientBoostingRegressor(random_state=SEED))
    search = _halving_search(pipe, "hgb", n_candidates=n_candidates, X=X, factor=factor, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search

# --- Bagging Trees
def run_randomForest(X, y, preprocessor, n_iter=N_ITER, n_jobs=-1):
    pipe = _pipe(preprocessor, "rf", RandomForestRegressor(
        random_state=SEED,
        n_jobs=1,
        oob_score=True,
        bootstrap=True,
    ))
    search = _rand_search(pipe, "rf", n_iter=n_iter, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search


def run_extraTrees(X, y, preprocessor, n_candidates=N_ITER, factor=FACTOR, n_jobs=-1):
    pipe = _pipe(preprocessor, "et", ExtraTreesRegressor(random_state=SEED, n_jobs=-1))
    search = _halving_search(pipe, "et", n_candidates=n_candidates, X=X, factor=factor, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search

# --- Kernel Based
def run_KNN(X, y, preprocessor, n_candidates=N_ITER, factor=FACTOR, n_jobs=-1):
    pipe = _pipe(preprocessor, "knn", KNeighborsRegressor(n_jobs=n_jobs))
    search = _halving_search(pipe, "knn", n_candidates=n_candidates, X=X, factor=factor, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search


def run_SVR(X, y, preprocessor, n_candidates=N_ITER, factor=FACTOR, n_jobs=-1):
    pipe = Pipeline([
        ("pre", preprocessor),
        ("rbf", RBFSampler(random_state=SEED)),
        ("svr", LinearSVR(max_iter=SVR_ITER, tol=SVR_TOL, random_state=SEED)),
    ])
    search = _halving_search(pipe, "svr", n_candidates=n_candidates, X=X, factor=factor, n_jobs=1)
    fit_model(search, X, y)
    return search


# --- Explainable additive
def run_EBM(X, y, preprocessor, n_candidates=N_ITER, factor=FACTOR, n_jobs=-1):
    pipe = _pipe(preprocessor, "ebm", ExplainableBoostingRegressor(random_state=SEED, n_jobs=n_jobs))
    search = _halving_search(pipe, "ebm", n_candidates=n_candidates, X=X, factor=factor, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search


class TorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_dim=16, epochs=100, lr=0.01, verbose=False):
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.verbose = verbose
        self.model = None

    def fit(self, X, y):
        X = X.toarray() if hasattr(X, "toarray") else X
        X, y = self._validate_data(X, y, accept_sparse=True, y_numeric=True, multi_output=False)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y.values if isinstance(y, pd.Series) else y, dtype=torch.float32)

        self.model = nn.Sequential(
            nn.Linear(self.n_features_in_, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        for epoch in range(self.epochs):
            self.model.train()
            opt.zero_grad()
            loss = loss_fn(self.model(X_t).squeeze(), y_t)
            loss.backward()
            opt.step()
            if self.verbose and (epoch + 1) % 25 == 0:
                print(f"epoch {epoch + 1}/{self.epochs} – loss {loss.item():.3f}")

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = X.toarray() if hasattr(X, "toarray") else X
        X_t = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_t).squeeze().numpy()

    def get_params(self, deep=True):
        return {"hidden_dim": self.hidden_dim, "epochs": self.epochs, "lr": self.lr, "verbose": self.verbose}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


def run_NeuralNetwork(X, y, preprocessor, n_candidates=N_ITER, factor=FACTOR, n_jobs=16):
    pipe = _pipe(preprocessor, "nn", TorchRegressor(verbose=False))
    search = _halving_search(pipe, "nn", n_candidates=n_candidates, X=X, factor=factor, n_jobs=n_jobs)
    fit_model(search, X, y)
    return search


class SkTabNet(RegressorMixin, BaseEstimator):
    def __init__(self, *, n_d=8, n_a=8, n_steps=3, gamma=1.3,
                 cat_idxs=(), cat_dims=(), cat_emb_dim=1, **kwargs):
        self.n_d, self.n_a, self.n_steps, self.gamma = n_d, n_a, n_steps, gamma
        self.cat_idxs, self.cat_dims, self.cat_emb_dim = tuple(cat_idxs), tuple(cat_dims), cat_emb_dim
        self.kwargs = kwargs

    def get_params(self, deep=True):
        return {
            **self.kwargs,
            "n_d": self.n_d, "n_a": self.n_a, "n_steps": self.n_steps, "gamma": self.gamma,
            "cat_idxs": self.cat_idxs, "cat_dims": self.cat_dims, "cat_emb_dim": self.cat_emb_dim,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y, **fit_params):
        X, y = self._validate_data(X, y, accept_sparse=False, y_numeric=True, multi_output=False)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self._model = TabNetRegressor(
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps, gamma=self.gamma,
            cat_idxs=list(self.cat_idxs), cat_dims=list(self.cat_dims),
            cat_emb_dim=self.cat_emb_dim, **self.kwargs,
        )
        self._model.fit(X, y, **fit_params)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self)
        y_pred = self._model.predict(X)
        return y_pred.squeeze()


def run_tabnet(X, y, X_num, X_cat, preprocessor, n_candidates=N_ITER, factor=FACTOR, n_jobs=-1):
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(y, pd.Series):
        y = y.to_numpy().reshape(-1, 1)
    elif y.ndim == 1:
        y = y.reshape(-1, 1)

    pipe = _pipe(preprocessor, "tab", SkTabNet(device_name=device_name, seed=SEED, verbose=0))
    search = _halving_search(pipe, "tab", n_candidates=n_candidates, X=X, factor=factor, n_jobs=n_jobs)

    fit_model(search, X, y, **tabnet_params)
    return search
