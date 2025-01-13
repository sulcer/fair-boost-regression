import numpy as np
from fairlearn.metrics import MetricFrame
from sklearn.ensemble import AdaBoostRegressor
from sklearn.utils import check_array
import sklearn.metrics as skm


class FairBoostRegressor_v1(AdaBoostRegressor):
    """
    FairBoost Regressor is an extension of the AdaBoost Regressor
    that incorporates fairness-aware adjustments. It ensures balanced
    performance across protected groups defined by sensitive attributes.

    Parameters
    ----------
    estimator : object, default=None
        The base estimator to be boosted. If None, a `DecisionTreeRegressor`
        is used as the default base estimator.

    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.

    learning_rate : float, default=1.0
        Weight applied to each regressor at each boosting iteration.

    loss : {'linear', 'square', 'exponential'}, default='linear'
        The loss function to use when updating sample weights.

    random_state : int, RandomState instance, default=None
        Controls the randomness of the estimator.

    Z : array-like, shape (n_samples, n_features), default=None
        Protected features used for fairness evaluation. It can include
        one or multiple sensitive attributes.

    fairness_weight : float, default=1.0
        The weight controlling the impact of fairness adjustments on the
        model's objective.

    protected : {'single', 'multiple'}, default='single'
        Indicates whether a single or multiple protected attributes are
        considered for fairness evaluation.

    Attributes
    ----------
    grouped_metric_diff : list
        List of fairness metric differences between groups for each boosting iteration.

    rmse_by_group : list
        Root Mean Squared Error (RMSE) values calculated for each protected
        group at every iteration.

    mae_by_group : list
        Mean Absolute Error (MAE) values for each protected group at each
        iteration.

    min_target_ : float
        Minimum value in the target variable.

    max_target_ : float
        Maximum value in the target variable.

    target_range_ : float
        Range (max - min) of the target variable.

    Methods
    -------
    set_Z(Z):
        Assign the protected features used for fairness evaluation.

    set_fairness_weight(fairness_weight):
        Set the weight parameter controlling the importance of fairness
        during training.

    _boost(iboost, X, y, sample_weight, random_state):
        Perform a boosting iteration with fairness-aware adjustments.
    """

    def __init__(self,
                 estimator=None,
                 *,
                 n_estimators=50,
                 learning_rate=1.0,
                 loss="linear",
                 random_state=None,
                 Z=None,
                 fairness_weight=1.0,
                 protected='single'):

        super().__init__(estimator=estimator,
                         n_estimators=n_estimators,
                         learning_rate=learning_rate,
                         loss=loss,
                         random_state=random_state)

        self.fairness_weight = fairness_weight
        self.protected = protected
        self.grouped_metric_diff = []
        self.rmse_by_group = []
        self.mae_by_group = []

    def set_Z(self, Z):
        # Protected features
        self.Z = check_array(Z, dtype=None, ensure_2d=False)

        if len(self.Z.shape) < 2:
            self.Z = self.Z.reshape(-1, 1)

        if self.Z.shape[1] == 1:
            self.Z_tpl = tuple(self.Z[:, 0])
        else:
            self.Z_tpl = tuple(map(tuple, self.Z))

    def set_fairness_weight(self, fairness_weight):
        self.fairness_weight = fairness_weight

    # v1 - FairBoostRegressor based on existing classification
    def _boost(self, iboost, X, y, sample_weight, random_state):
        """
        Perform a single boosting iteration with fairness-aware adjustments.

        Parameters
        ----------
        iboost : int
            The index of the current boosting iteration.

        X : array-like, shape (n_samples, n_features)
            Training data features.

        y : array-like, shape (n_samples,)
            Target values.

        sample_weight : array-like, shape (n_samples,)
            Current sample weights.

        random_state : RandomState instance
            Random state for reproducibility.

        Returns
        -------
        sample_weight : array-like, shape (n_samples,)
            Updated sample weights after the boosting iteration.

        estimator_weight : float
            The weight of the current estimator in the ensemble.

        estimator_error : float
            The error of the current estimator, considering fairness constraints.

        Notes
        -----
        This method integrates group fairness metrics, such as RMSE, MAE into
        the boosting algorithm. Adjustments are made based on the defined
        fairness weight and protected attributes.
        """
        # Make a new estimator
        estimator = self._make_estimator(random_state=random_state)

        # Fit the current estimator
        estimator.fit(X, y, sample_weight=sample_weight)

        if iboost == 0:
            # Initialize attributes for the target variable during the first iteration
            self.min_target_ = np.min(y)  # Minimum target value
            self.max_target_ = np.max(y)  # Maximum target value
            self.target_range_ = self.max_target_ - self.min_target_  # Range of target values

        # Predict the target values
        y_predict = estimator.predict(X)

        # Calculate RMSE for this iteration
        rmse = np.sqrt(skm.mean_squared_error(y, y_predict, sample_weight=sample_weight))

        # Fairness metric for groups defined by Z
        grouped_metric = MetricFrame(metrics=skm.mean_squared_error,
                                     y_true=y,
                                     y_pred=y_predict,
                                     sensitive_features=self.Z)

        # Transform the grouped metric into a dictionary and get the minimum value of the metric
        by_Z = grouped_metric.by_group.to_dict()
        met_min = grouped_metric.group_min()

        # A threshold for small values
        threshold = 1e-5
        if met_min < threshold:
            # If the minimum value is too small, avoid division
            normalized_by_Z = {k: 0 if v == met_min else v / threshold for k, v in by_Z.items()}
        else:
            # Normalize by the minimum value
            normalized_by_Z = {k: v / met_min for k, v in by_Z.items()}

        # Handle storing grop-based metrics (RMSE) depending on the protected attribute (single or multiple)
        if self.protected != 'single':
            tmp = {}
            for k, v in grouped_metric.by_group.to_dict().items():
                tmp[' '.join(k)] = np.sqrt(v)  # Convert MSE to RMSE
        else:
            tmp = {k: np.sqrt(v) for k, v in by_Z.items()}  # Convert MSE to RMSE

        # Store the metric diff and RMSE by group
        self.grouped_metric_diff.append(grouped_metric.difference())
        self.rmse_by_group.append(tmp)

        # Calculate the MAE for this iteration
        groped_metric_mae = MetricFrame(metrics=skm.mean_absolute_error,
                                        y_true=y,
                                        y_pred=y_predict,
                                        sensitive_features=self.Z)

        # Handle storing grop-based metrics (MAE) depending on the protected attribute (single or multiple)
        if self.protected != 'single':
            tmp = {}
            for k, v in groped_metric_mae.by_group.to_dict().items():
                tmp[' '.join(k)] = v
        else:
            tmp = groped_metric_mae.by_group.to_dict()

        # Store the MAE by group
        self.mae_by_group.append(tmp)

        # Estimator error for regression
        estimator_error = rmse * (1 - self.fairness_weight) + grouped_metric.difference() * self.fairness_weight

        # Handle cases of perfect performance or excessive error
        if estimator_error < 1e-10:  # Nearly perfect predictor
            # Perfect performance or negligible fairness-adjusted error
            return sample_weight, 1.0, 0.0
        elif estimator_error > 0.95:  # Poor predictor
            # Discard the current estimator if its error is too high and if it isn't the only one
            if len(self.estimators_) > 1:
                self.estimators_.pop(-1)
            return None, None, None

        # Calculate estimator weight
        estimator_weight = self.learning_rate * estimator_error

        # Adjust estimator weight based on fairness
        metric_proportion = np.array([normalized_by_Z[t] for t in self.Z_tpl])
        estimator_weight *= metric_proportion

        # Update sample weights
        if not iboost == self.n_estimators - 1:
            sample_weight *= np.exp(
                estimator_weight * ((sample_weight > 0) | (estimator_weight < 0))
            )

        # Return the updated sample weights, estimator weight, and estimator error
        return sample_weight, 1.0, estimator_error