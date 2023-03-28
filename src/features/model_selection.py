import copy
import time
from collections import defaultdict
from itertools import product

import numpy as np
import optuna
import pandas as pd
import xarray as xr
from joblib import Parallel
from sklearn.base import clone, is_classifier
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import (
    _fit_and_score, _insert_error_scores, _warn_or_raise_about_fit_failures)
from sklearn.pipeline import Pipeline
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import _check_fit_params, indexable
from tqdm import tqdm

import wandb


class OptunaSearch:
    def __init__(
            self, 
            estimator: Pipeline, 
            params_optuna, 
            do_cv: bool = False, 
            index_test = None, 
            n_trials: int=100, 
            direction='maximize',
            callback=None
            ):
        
        self.estimator = estimator
        self.params_optuna = params_optuna
        self.do_cv = do_cv
        self.index_test = index_test
        self.study = optuna.create_study(direction=direction)
        self.n_trials = n_trials
        self.callback = callback


    def get_trial_param(self, trial):
        trial_params = {}
        for attr_name, (param_name, param_values) in self.params_optuna:
            params = copy.deepcopy(param_values)
            params['name'] = param_name
            trial_params[param_name] = getattr(trial, attr_name)(**params)
        return trial_params
    

    def fit(self, X: xr.Dataset, y: pd.DataFrame):
        def objectif(trial):
            param_estimator = self.get_trial_param(trial)
            estimator = self.estimator.set_params(**param_estimator)

            if self.do_cv:
                score = cross_val_score(estimator, X=X, y=y, cv=ObsKFold()).mean()
            else:
                index_train = [idx for idx in y.index.get_level_values('ts_obs').unique() if not idx in self.index_test]
                y_train = y.loc[index_train, :].reorder_levels(['ts_obs', 'ts_aug']).sort_index()
                estimator = estimator.fit(X.sel(ts_obs=index_train), y=y_train)
                y_test = y.loc[self.index_test, :].reorder_levels(['ts_obs', 'ts_aug']).sort_index()
                score = estimator.score(X.sel(ts_obs=self.index_test), y=y_test)
            
            # if self.callback:
            #     self.callback(score)

            return score

        self.study.optimize(objectif, self.n_trials)
        return self


class ObsKFold(KFold):
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X: pd.DataFrame, y=None, groups=None):

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X expected pd.DataFrame but received %s." % (type(X)))

        full_idx = X.index.get_level_values("ts_obs")
        uniq_idx = full_idx.unique()
        for train, test in super().split(X=uniq_idx, y=None, groups=None):
            yield full_idx.isin(train), full_idx.isin(test)


class WandbCallback():
    def __init__(self, project: str, tags: list[str], estimator_name: str) -> None:
        self.project = project
        self.tags = tags
        self.estimator_name = estimator_name
        
    def save(self, wandb_run, parameters, train_scores, test_scores, num_split)->None:
        run_name = "{}-{}".format(wandb_run['sweep_run_name'], num_split)
        run = wandb.init(
            project=self.project,
            tags=self.tags,
            group=wandb_run['sweep_id'],
            name=run_name,
            config=parameters,
            job_type=wandb_run['sweep_run_name'],
        )

        run.log(dict(test_scores=test_scores, train_scores=train_scores))
    
        wandb.join()


def _fit_score_callback(
    estimator,
    X,
    y,
    scorer,
    train,
    test,
    verbose,
    parameters,
    wandb_run,
    callback,
    fit_params,
    return_train_score=None,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
) -> dict:

    result = _fit_and_score(
        estimator,
        X,
        y,
        scorer,
        train,
        test,
        verbose,
        parameters,
        fit_params,
        return_train_score=True,
        return_parameters=return_parameters,
        return_n_test_samples=return_n_test_samples,
        return_times=return_times,
        return_estimator=True,
        split_progress=split_progress,
        candidate_progress=candidate_progress,
        error_score=error_score,
    )

    callback.save(
        wandb_run,
        parameters=result["estimator"].get_params(),
        train_scores=result["train_scores"],
        test_scores=result["test_scores"],
        num_split=split_progress[0],
    )

    return result


class WandBGridSearchCV(GridSearchCV):
    def __init__(
        self,
        estimator,
        param_grid,
        callback,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        super().__init__(
            estimator,
            param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.callback = callback

    def _wandb_sweep_run(self, n_candidates):
        for _ in range(n_candidates):
            sweep_run = wandb.init()
            sweep_id = sweep_run.id
            sweep_run_name = sweep_run.name
            # sweep_run.save()
            yield dict(sweep_id=sweep_id, sweep_run_name=sweep_run_name)

    def fit(self, X, y=None, *, groups=None, **fit_params):
        """Run fit with all sets of parameters.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set. Only used in conjunction with a "Group" :term:`cv`
            instance (e.g., :class:`~sklearn.model_selection.GroupKFold`).

        **fit_params : dict of str -> object
            Parameters passed to the `fit` method of the estimator.

            If a fit parameter is an array-like whose length is equal to
            `num_samples` then it will be split across CV groups along with `X`
            and `y`. For example, the :term:`sample_weight` parameter is split
            because `len(sample_weights) = len(X)`.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """

        # return super().fit(X=X, y=y, groups=groups, **fit_params)
        estimator = self.estimator
        refit_metric = "score"

        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(self.estimator, self.scoring)
        else:
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit

        X, y, groups = indexable(X, y, groups)
        fit_params = _check_fit_params(X, fit_params)

        cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
        n_splits = cv_orig.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_n_test_samples=True,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )
        results = {}
        with parallel:
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)

            def evaluate_candidates(candidate_params, cv=None, more_results=None):
                cv = cv or cv_orig
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                wandb_runs = self._wandb_sweep_run(n_candidates)

                if self.verbose > 0:
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )

                out = parallel(
                    delayed(_fit_score_callback)(
                        clone(base_estimator),
                        X,
                        y,
                        train=train,
                        test=test,
                        parameters=parameters,
                        split_progress=(split_idx, n_splits),
                        candidate_progress=(cand_idx, n_candidates),
                        wandb_run=wandb_run,
                        callback=self.callback,
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, (wandb_run, parameters)), (
                        split_idx,
                        (train, test),
                    ) in product(
                        enumerate(zip(wandb_runs, candidate_params)),
                        enumerate(cv.split(X, y, groups)),
                    )
                )

                if len(out) < 1:
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits:
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )

                _warn_or_raise_about_fit_failures(out, self.error_score)

                # For callable self.scoring, the return type is only know after
                # calling. If the return type is a dictionary, the error scores
                # can now be inserted with the correct key. The type checking
                # of out will be done in `_insert_error_scores`.
                if callable(self.scoring):
                    _insert_error_scores(out, self.error_score)

                all_candidate_params.extend(candidate_params)
                all_out.extend(out)

                if more_results is not None:
                    for key, value in more_results.items():
                        all_more_results[key].extend(value)

                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results

            self._run_search(evaluate_candidates)

            # multimetric is determined here because in the case of a callable
            # self.scoring the return type is only known after calling
            first_test_score = all_out[0]["test_scores"]
            self.multimetric_ = isinstance(first_test_score, dict)

            # check refit_metric now for a callabe scorer that is multimetric
            if callable(self.scoring) and self.multimetric_:
                self._check_refit_for_multimetric(first_test_score)
                refit_metric = self.refit

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = self._select_best_index(
                self.refit, refit_metric, results
            )
            if not callable(self.refit):
                # With a non-custom callable, we can select the best score
                # based on the best index
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]

        if self.refit:
            # we clone again after setting params in case some
            # of the params are estimators as well.
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers

        self.cv_results_ = results
        self.n_splits_ = n_splits

        return self
