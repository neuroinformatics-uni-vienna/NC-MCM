import json
import os
import warnings
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score, mean_squared_error
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression


from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed


from rich import box
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


import torch

class SimpleMultiClassPredictor(torch.nn.Module):
    """
    Linear (single layer) multi-class decoder trained with cross-entropy.

    Used to decode behavior proportions (soft labels) from neuronal activity.

    Args:
        input_dim: number of input features (neurons).
        num_classes: number of behaviors (output logits).
    """

    class LossFunction(torch.nn.Module):
        """Cross-entropy between the predicted logits and the target (soft) labels."""

        def __init__(self):
            super().__init__()
            self.loss_fn = torch.nn.CrossEntropyLoss()

        def forward(self, outputs, targets):
            return self.loss_fn(outputs, targets)

    def __init__(self, input_dim, num_classes):
        super(SimpleMultiClassPredictor, self).__init__()
        self.fc = torch.nn.Linear(input_dim, num_classes)
        self.loss = self.LossFunction()

    def forward(self, x):
        """Return the logits for the input batch `x`."""
        return self.fc(x)

    def train_model(self, X_train, y_train, epochs=100, lr=0.01):
        """
        Train the decoder with Adam (batch size 32).

        Args:
            X_train: input tensor, shape (n_samples, input_dim).
            y_train: target tensor, shape (n_samples, num_classes).
            epochs: number of training epochs.
            lr: learning rate.
        """

        train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(epochs):
            print(f"[bold cyan]Training epoch {epoch + 1}/{epochs}[/bold cyan]", end="\r")
            for batch_X, batch_y in train_dataloader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = self.loss(outputs, batch_y)
                loss.backward()
                optimizer.step()
        print()  # Print a newline after the training is complete

    def predict(self, X):
        """
        Return the raw logits (no softmax) for `X`, a numpy array or a tensor
        of shape (n_samples, input_dim).
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        self.eval()
        with torch.no_grad():
            outputs = self.forward(X)
        return outputs  # Return raw outputs (logits) for further processing


class CrossDecoder:
    """
    Cross-decoding evaluation of a denoiser: decoders are trained on noisy
    and on denoised activity and each is tested on both test sets.

    Result keys follow the pattern ``<metric>_<train data>_<test data>``,
    e.g. ``bal_acc_noisy_denoised`` is the balanced accuracy of the decoder
    trained on noisy data and tested on denoised data; ``*_train`` keys are
    the train-on-train baselines.

    Args:
        behavior_type: one of 'discrete', 'proportions' or 'continuous'.
        n_permutations: number of permutations of the permutation test
            (discrete behavior only; 0 disables it).
        random_state: seed for the permutations.
        n_jobs: number of joblib workers for the permutation test.
        save_path: if given, every results dictionary is also saved there
            as JSON (parent directories are created if needed).
    """

    # (train data, test data) conditions evaluated by the permutation test
    CONDITIONS = ["noisy_noisy", "denoised_denoised", "noisy_denoised", "denoised_noisy"]

    def __init__(
        self,
        behavior_type: str,
        n_permutations: int = 1000,
        random_state: int | None = None,
        n_jobs: int = -1,
        save_path: str | None = None
    ):
        assert behavior_type in ["discrete", "proportions", "continuous"], "behavior_type must be one of 'discrete', 'proportions', or 'continuous'."

        self.behavior_type = behavior_type
        self.n_permutations = n_permutations
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.save_path = save_path

    # ============================================================
    # PUBLIC API
    # ============================================================

    def score(self, denoiser, X_prep, B_prep, train_indices, test_indices) -> dict:
        """
        Apply the denoiser to `X_prep` and compute the cross-decoding scores.

        The results are also saved to `save_path`, if one was given.

        Args:
            denoiser: object exposing ``pipeline(X_prep) -> (X_denoised, Y_denoised)``.
            X_prep: windowed noisy activity, shape (n_samples, 1, window, n_neurons).
            B_prep: behavior labels, shape (n_samples,) for 'discrete' or
                (n_samples, n_behaviors) for 'proportions'.
            train_indices: sample indices of the training set.
            test_indices: sample indices of the test set.

        Returns:
            dict with the accuracy, balanced accuracy and (for 'discrete'
            behavior) permutation test results.
        """
        if self.behavior_type == "discrete":
            results = self._single_behavior(denoiser, X_prep, B_prep, train_indices, test_indices)
        elif self.behavior_type == "proportions":
            results = self._multiple_behaviors(denoiser, X_prep, B_prep, train_indices, test_indices)
        elif self.behavior_type == "continuous":
            results = self._continuous_behavior(denoiser, X_prep, B_prep, train_indices, test_indices)
        else:
            raise NotImplementedError(f"Cross-decoding for behavior_type '{self.behavior_type}' is not implemented.")

        self.save(results)
        return results

    def score_from_arrays(
        self,
        X_original_train,
        X_original_test,
        X_denoised_train,
        X_denoised_test,
        B_train,
        B_test
    ) -> dict:
        """
        Compute the (discrete) cross-decoding scores from data that has
        already been denoised and split into train / test sets.

        The results are also saved to `save_path`, if one was given.

        Args:
            X_original_train, X_original_test: noisy activity, shape (n_samples, n_neurons).
            X_denoised_train, X_denoised_test: denoised activity, same shapes.
            B_train, B_test: discrete behavior labels, shape (n_samples,).

        Returns:
            dict with the accuracy, balanced accuracy and permutation test results.
        """
        results = self._compute(
            X_original_train, X_original_test,
            X_denoised_train, X_denoised_test,
            B_train, B_test
        )
        self.save(results)
        return results

    def save(self, results: dict):
        """Save `results` as JSON to `self.save_path` (no-op if it is None)."""
        if self.save_path is None:
            return
        directory = os.path.dirname(self.save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.save_path, "w") as f:
            # numpy / torch scalars are not JSON serializable: store them as floats
            json.dump(results, f, indent=4, default=float)

    @staticmethod
    def load(path: str) -> dict:
        """Load a results dictionary previously saved with `save`."""
        with open(path) as f:
            return json.load(f)

    # ============================================================
    # DISCRETE BEHAVIOR
    # ============================================================

    def _single_behavior(self, denoiser, X_prep, B_prep, train_indices, test_indices) -> dict:
        """Denoise `X_prep`, split it into train / test sets and run `_compute`."""
        assert train_indices is not None and test_indices is not None, "Train and test indices must be provided."

        # 1. Apply the denoiser and keep only the last time step of each window
        X_denoised, Y_denoised = denoiser.pipeline(X_prep)
        X_denoised_step = X_denoised[:, -1, :]
        X_noisy_step = X_prep[:, 0, -1, :]

        X_n_train, X_n_test = X_noisy_step[train_indices], X_noisy_step[test_indices]
        X_d_train, X_d_test = X_denoised_step[train_indices], X_denoised_step[test_indices]
        B_train, B_test = B_prep[train_indices], B_prep[test_indices]

        return self._compute(X_n_train, X_n_test, X_d_train, X_d_test, B_train, B_test)

    @staticmethod
    def _permutation_scores(
        B_perm,
        X_n_train, X_d_train,
        X_n_test, X_d_test,
        X_d_test_scaled_by_noisy, X_n_test_scaled_by_denoised,
        B_test_m, test_mask,
    ):
        """
        Run a single permutation: fit both SVMs on the shuffled labels
        `B_perm` and return the 4 balanced accuracies
        [noisy_noisy, denoised_denoised, noisy_denoised, denoised_noisy].
        Test predictions are restricted to `test_mask` (classes present in
        both train and test sets).
        """
        # Warning filters are not propagated to the joblib worker processes
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

            svm_n = SVC(kernel="linear").fit(X_n_train, B_perm)
            svm_d = SVC(kernel="linear").fit(X_d_train, B_perm)

            preds = (
                svm_n.predict(X_n_test),
                svm_d.predict(X_d_test),
                svm_n.predict(X_d_test_scaled_by_noisy),
                svm_d.predict(X_n_test_scaled_by_denoised),
            )
            return [
                balanced_accuracy_score(B_test_m, p[test_mask]) for p in preds
            ]

    def _compute(
        self,
        X_original_train,
        X_original_test,
        X_denoised_train,
        X_denoised_test,
        B_train,
        B_test
    ) -> dict:
        """
        Core discrete cross-decoding: standardize each dataset, fit a linear
        SVM on noisy and on denoised data, evaluate every (train, test)
        combination and run the permutation test on the balanced accuracy.
        """
        # 2. Raw train / test splits
        X_n_train_raw, X_n_test_raw = X_original_train, X_original_test
        X_d_train_raw, X_d_test_raw = X_denoised_train, X_denoised_test

        # 3. Independent standardization, with SEPARATE scaler instances
        scaler_noisy = StandardScaler()
        X_n_train = scaler_noisy.fit_transform(X_n_train_raw)
        X_n_test = scaler_noisy.transform(X_n_test_raw)

        scaler_denoised = StandardScaler()
        X_d_train = scaler_denoised.fit_transform(X_d_train_raw)
        X_d_test = scaler_denoised.transform(X_d_test_raw)

        # 4. Cross standardization FOR CROSS-DECODING: each test set is scaled
        #    with the scaler of the model that predicts it (aligns the spaces)
        X_d_test_scaled_by_noisy = scaler_noisy.transform(X_d_test_raw)
        X_n_test_scaled_by_denoised = scaler_denoised.transform(X_n_test_raw)

        # 5. Fit the SVMs
        svm_noisy = SVC(kernel='linear')
        svm_noisy.fit(X_n_train, B_train)

        svm_denoised = SVC(kernel='linear')
        svm_denoised.fit(X_d_train, B_train)

        # 6. Predictions (inputs scaled consistently with the SVM that predicts them)
        B_pred_noisy_noisy = svm_noisy.predict(X_n_test)
        B_pred_denoised_denoised = svm_denoised.predict(X_d_test)

        # Cross-decoding, scaled consistently with the predicting model
        B_pred_noisy_denoised = svm_noisy.predict(X_d_test_scaled_by_noisy)
        B_pred_denoised_noisy = svm_denoised.predict(X_n_test_scaled_by_denoised)

        # Train-on-train baseline
        B_pred_noisy_train = svm_noisy.predict(X_n_train)
        B_pred_denoised_train = svm_denoised.predict(X_d_train)

        # 7. Metrics calculation
        acc_noisy_noisy = accuracy_score(B_test, B_pred_noisy_noisy)
        acc_denoised_denoised = accuracy_score(B_test, B_pred_denoised_denoised)
        acc_noisy_denoised = accuracy_score(B_test, B_pred_noisy_denoised)
        acc_denoised_noisy = accuracy_score(B_test, B_pred_denoised_noisy)

        acc_noisy_train = accuracy_score(B_train, B_pred_noisy_train)
        acc_denoised_train = accuracy_score(B_train, B_pred_denoised_train)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

            # Compute the balanced accuracy scores only for the classes present in both the
            # test set and train set to avoid the warning about undefined metrics.
            common_classes = set(B_test) & set(B_train)
            if not common_classes:
                raise ValueError("No common classes between test and train sets.")

            B_test_filtered = [b for b in B_test if b in common_classes]
            B_pred_noisy_noisy_filtered = [b for i, b in enumerate(B_pred_noisy_noisy) if B_test[i] in common_classes]
            B_pred_denoised_denoised_filtered = [b for i, b in enumerate(B_pred_denoised_denoised) if B_test[i] in common_classes]
            B_pred_noisy_denoised_filtered = [b for i, b in enumerate(B_pred_noisy_denoised) if B_test[i] in common_classes]
            B_pred_denoised_noisy_filtered = [b for i, b in enumerate(B_pred_denoised_noisy) if B_test[i] in common_classes]

            B_train_filtered = [b for b in B_train if b in common_classes]
            B_pred_noisy_train_filtered = [b for i, b in enumerate(B_pred_noisy_train) if B_train[i] in common_classes]
            B_pred_denoised_train_filtered = [b for i, b in enumerate(B_pred_denoised_train) if B_train[i] in common_classes]

            bal_acc_noisy_noisy = balanced_accuracy_score(B_test_filtered, B_pred_noisy_noisy_filtered)
            bal_acc_denoised_denoised = balanced_accuracy_score(B_test_filtered, B_pred_denoised_denoised_filtered)
            bal_acc_noisy_denoised = balanced_accuracy_score(B_test_filtered, B_pred_noisy_denoised_filtered)
            bal_acc_denoised_noisy = balanced_accuracy_score(B_test_filtered, B_pred_denoised_noisy_filtered)

            bal_acc_noisy_train = balanced_accuracy_score(B_train_filtered, B_pred_noisy_train_filtered)
            bal_acc_denoised_train = balanced_accuracy_score(B_train_filtered, B_pred_denoised_train_filtered)

        # 8. Permutation test (shuffling of the training labels). The p-value is
        #    one-sided: fraction of null scores >= the observed one, with the +1 correction
        B_train_arr = np.asarray(B_train)
        B_test_arr = np.asarray(B_test)

        observed = {
            "noisy_noisy": bal_acc_noisy_noisy,
            "denoised_denoised": bal_acc_denoised_denoised,
            "noisy_denoised": bal_acc_noisy_denoised,
            "denoised_noisy": bal_acc_denoised_noisy,
        }

        perm_results = {}
        if self.n_permutations > 0:
            rng = np.random.default_rng(self.random_state)
            test_mask = np.isin(B_test_arr, list(common_classes))
            B_test_m = B_test_arr[test_mask]
            perms = [rng.permutation(B_train_arr) for _ in range(self.n_permutations)]

            null_arr = np.asarray(
                Parallel(n_jobs=self.n_jobs)(
                    delayed(self._permutation_scores)(
                        B_perm,
                        X_n_train, X_d_train,
                        X_n_test, X_d_test,
                        X_d_test_scaled_by_noisy, X_n_test_scaled_by_denoised,
                        B_test_m, test_mask,
                    )
                    for B_perm in tqdm(perms, desc="Permutation test", unit="perm")
                )
            )  # shape (n_permutations, 4)

            for i, c in enumerate(self.CONDITIONS):
                null = null_arr[:, i]
                perm_results[f"pval_{c}"] = (1 + np.sum(null >= observed[c])) / (self.n_permutations + 1)
                perm_results[f"null_mean_{c}"] = float(null.mean())
                perm_results[f"null_p95_{c}"] = float(np.percentile(null, 95))
        else:
            for c in self.CONDITIONS:
                perm_results[f"pval_{c}"] = np.nan
                perm_results[f"null_mean_{c}"] = np.nan
                perm_results[f"null_p95_{c}"] = np.nan

        return {
            "acc_noisy_noisy": acc_noisy_noisy,
            "acc_denoised_denoised": acc_denoised_denoised,
            "acc_noisy_denoised": acc_noisy_denoised,
            "acc_denoised_noisy": acc_denoised_noisy,
            "acc_noisy_train": acc_noisy_train,
            "acc_denoised_train": acc_denoised_train,
            "bal_acc_noisy_noisy": bal_acc_noisy_noisy,
            "bal_acc_denoised_denoised": bal_acc_denoised_denoised,
            "bal_acc_noisy_denoised": bal_acc_noisy_denoised,
            "bal_acc_denoised_noisy": bal_acc_denoised_noisy,
            "bal_acc_noisy_train": bal_acc_noisy_train,
            "bal_acc_denoised_train": bal_acc_denoised_train,
            **perm_results
        }

    # ============================================================
    # PROPORTIONS BEHAVIOR
    # ============================================================

    def _multiple_behaviors(self, denoiser, X_prep, B_prep, train_indices, test_indices) -> dict:
        """
        Cross-decoding for behavior proportions: the behaviors present at each
        sample are turned into a uniform distribution and a linear decoder
        (`SimpleMultiClassPredictor`) is trained on noisy and on denoised data.

        Note: here the ``acc_*`` entries are cross-entropy LOSSES (lower is
        better), the ``bal_acc_*`` entries are placeholders set to 0 and no
        permutation test is run.
        """
        assert train_indices is not None and test_indices is not None, "Train and test indices must be provided."

        # Count the number of behaviors (columns) in B_prep
        num_behaviors = B_prep.shape[1]
        print(f"[bold cyan]Detected {num_behaviors} behaviors for cross-decoding evaluation.[/bold cyan]")
        print(f"[bold cyan]Detected {B_prep.shape[0]} samples for cross-decoding evaluation.[/bold cyan]")
        print(f"[bold cyan]Train indices: {len(train_indices)}, Test indices: {len(test_indices)}[/bold cyan]")

        # Apply the denoiser and keep only the last time step of each window
        X_denoised, Y_denoised = denoiser.pipeline(X_prep)
        X_denoised_step = X_denoised[:, -1, :]
        X_noisy_step = X_prep[:, 0, -1, :]

        X_n_train_raw, X_n_test_raw = X_noisy_step[train_indices], X_noisy_step[test_indices]
        X_d_train_raw, X_d_test_raw = X_denoised_step[train_indices], X_denoised_step[test_indices]

        # Behaviors present at each sample -> uniform distribution over them
        B_binarized = (B_prep > 0).astype(int)
        B_distribution = B_binarized / B_binarized.sum(axis=1, keepdims=True)

        B_train = B_distribution[train_indices]
        B_test = B_distribution[test_indices]

        print(f"[bold cyan]Training set shape: {X_n_train_raw.shape}, {B_train.shape}[/bold cyan]")
        print(f"[bold cyan]Test set shape: {X_n_test_raw.shape}, {B_test.shape}[/bold cyan]")


        # Independent standardization, with SEPARATE scaler instances
        scaler_noisy = StandardScaler()
        X_n_train = scaler_noisy.fit_transform(X_n_train_raw)
        X_n_test = scaler_noisy.transform(X_n_test_raw)

        scaler_denoised = StandardScaler()
        X_d_train = scaler_denoised.fit_transform(X_d_train_raw)
        X_d_test = scaler_denoised.transform(X_d_test_raw)

        # Cross standardization: each test set is scaled with the scaler of the predicting model
        X_n_test_scaled_by_denoised = scaler_denoised.transform(X_n_test_raw)
        X_d_test_scaled_by_noisy = scaler_noisy.transform(X_d_test_raw)

        noisy_model = SimpleMultiClassPredictor(input_dim=X_n_train.shape[1], num_classes=num_behaviors)
        denoised_model = SimpleMultiClassPredictor(input_dim=X_d_train.shape[1], num_classes=num_behaviors)

        noisy_model.train_model(torch.tensor(X_n_train, dtype=torch.float32), torch.tensor(B_train, dtype=torch.float32), epochs=100, lr=0.01)
        denoised_model.train_model(torch.tensor(X_d_train, dtype=torch.float32), torch.tensor(B_train, dtype=torch.float32), epochs=100, lr=0.01)

        pred_noisy_noisy = noisy_model.predict(X_n_test)
        pred_noisy_denoised = noisy_model.predict(X_d_test_scaled_by_noisy)
        pred_denoised_denoised = denoised_model.predict(X_d_test)
        pred_denoised_noisy = denoised_model.predict(X_n_test_scaled_by_denoised)

        pred_noisy_noisy = pred_noisy_noisy.numpy()
        pred_noisy_denoised = pred_noisy_denoised.numpy()
        pred_denoised_denoised = pred_denoised_denoised.numpy()
        pred_denoised_noisy = pred_denoised_noisy.numpy()

        # Cross-entropy between the predicted logits and the target distributions
        loss = torch.nn.CrossEntropyLoss()

        acc_noisy_noisy = loss(torch.tensor(pred_noisy_noisy, dtype=torch.float32), torch.tensor(B_test, dtype=torch.float32))
        acc_noisy_denoised = loss(torch.tensor(pred_noisy_denoised, dtype=torch.float32), torch.tensor(B_test, dtype=torch.float32))
        acc_denoised_denoised = loss(torch.tensor(pred_denoised_denoised, dtype=torch.float32), torch.tensor(B_test, dtype=torch.float32))
        acc_denoised_noisy = loss(torch.tensor(pred_denoised_noisy, dtype=torch.float32), torch.tensor(B_test, dtype=torch.float32))


        acc_noisy_train = loss(torch.tensor(noisy_model.predict(X_n_train), dtype=torch.float32), torch.tensor(B_train, dtype=torch.float32))
        acc_denoised_train = loss(torch.tensor(denoised_model.predict(X_d_train), dtype=torch.float32), torch.tensor(B_train, dtype=torch.float32))

        bal_acc_noisy_noisy = 0
        bal_acc_noisy_denoised = 0
        bal_acc_denoised_denoised = 0
        bal_acc_denoised_noisy = 0
        bal_acc_noisy_train = 0
        bal_acc_denoised_train = 0

        return {
            "acc_noisy_noisy": acc_noisy_noisy,
            "acc_denoised_denoised": acc_denoised_denoised,
            "acc_noisy_denoised": acc_noisy_denoised,
            "acc_denoised_noisy": acc_denoised_noisy,
            "acc_noisy_train": acc_noisy_train,
            "acc_denoised_train": acc_denoised_train,
            "bal_acc_noisy_noisy": bal_acc_noisy_noisy,
            "bal_acc_denoised_denoised": bal_acc_denoised_denoised,
            "bal_acc_noisy_denoised": bal_acc_noisy_denoised,
            "bal_acc_denoised_noisy": bal_acc_denoised_noisy,
            "bal_acc_noisy_train": bal_acc_noisy_train,
            "bal_acc_denoised_train": bal_acc_denoised_train
        }

    # ============================================================
    # CONTINUOUS BEHAVIOR
    # ============================================================

    def _continuous_behavior(self, denoiser, X_prep, B_prep, train_indices, test_indices) -> dict:
        """
        Cross-decoding for continuous behavior: a linear regression decoder
        is trained on noisy and on denoised data, and each is evaluated on
        both test sets (same scheme as the discrete / proportions cases).

        ``B_prep`` has shape (n_samples,) for a single continuous variable
        or (n_samples, n_variables) for several.

        Returns ``r2_<cond>`` (coefficient of determination, averaged
        uniformly over the variables; higher is better) and ``mse_<cond>``
        (lower is better) for every condition; no permutation test is run.
        """
        assert train_indices is not None and test_indices is not None, "Train and test indices must be provided."

        B = np.asarray(B_prep, dtype=float).reshape(len(B_prep), -1)

        # Apply the denoiser and keep only the last time step of each window
        X_denoised, _ = denoiser.pipeline(X_prep)
        X_denoised_step = X_denoised[:, -1, :]
        X_noisy_step = X_prep[:, 0, -1, :]

        X_n_train_raw, X_n_test_raw = X_noisy_step[train_indices], X_noisy_step[test_indices]
        X_d_train_raw, X_d_test_raw = X_denoised_step[train_indices], X_denoised_step[test_indices]
        B_train, B_test = B[train_indices], B[test_indices]

        # Independent standardization, with SEPARATE scaler instances
        scaler_noisy = StandardScaler()
        X_n_train = scaler_noisy.fit_transform(X_n_train_raw)
        X_n_test = scaler_noisy.transform(X_n_test_raw)

        scaler_denoised = StandardScaler()
        X_d_train = scaler_denoised.fit_transform(X_d_train_raw)
        X_d_test = scaler_denoised.transform(X_d_test_raw)

        # Cross standardization: each test set is scaled with the scaler of the predicting model
        X_n_test_scaled_by_denoised = scaler_denoised.transform(X_n_test_raw)
        X_d_test_scaled_by_noisy = scaler_noisy.transform(X_d_test_raw)

        lr_noisy = LinearRegression().fit(X_n_train, B_train)
        lr_denoised = LinearRegression().fit(X_d_train, B_train)

        # condition -> (model, inputs, targets)
        conditions = {
            "noisy_noisy": (lr_noisy, X_n_test, B_test),
            "denoised_denoised": (lr_denoised, X_d_test, B_test),
            "noisy_denoised": (lr_noisy, X_d_test_scaled_by_noisy, B_test),
            "denoised_noisy": (lr_denoised, X_n_test_scaled_by_denoised, B_test),
            "noisy_train": (lr_noisy, X_n_train, B_train),
            "denoised_train": (lr_denoised, X_d_train, B_train),
        }

        results = {}
        for key, (model, X, B_true) in conditions.items():
            B_pred = model.predict(X)
            results[f"r2_{key}"] = r2_score(B_true, B_pred)
            results[f"mse_{key}"] = mean_squared_error(B_true, B_pred)
        return results

    # ============================================================
    # PRINTING
    # ============================================================

    @staticmethod
    def print_results(results: dict):
        """
        Print the cross-decoding results as formatted tables.

        Args:
            results: dictionary returned by `score` / `score_from_arrays`,
                containing the accuracy, balanced accuracy and permutation
                test scores.
        """

        def _is_nan(x) -> bool:
            return x is None or np.isnan(x)

        def _num(x, fmt=".4f", style=None) -> str:
            if _is_nan(x):
                return "[dim]n/a[/dim]"
            s = format(x, fmt)
            return f"[{style}]{s}[/{style}]" if style else s

        def _p_cell(p) -> tuple[str, str]:
            """Return (colored p-value, colored significance label)."""
            if _is_nan(p):
                return "[dim]n/a[/dim]", ""
            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            else:
                stars = "n.s."
            color = "green" if p < 0.05 else "red"
            return f"[{color}]{p:.4f}[/{color}]", f"[bold {color}]{stars}[/bold {color}]"

        console = Console()

        conditions = [
            ("noisy_noisy",       "Noisy",    "Noisy"),
            ("denoised_denoised", "Denoised", "Denoised"),
            ("noisy_denoised",    "Noisy",    "Denoised"),
            ("denoised_noisy",    "Denoised", "Noisy"),
        ]

        if "r2_noisy_noisy" in results:
            # Continuous behavior: regression scores, no permutation test
            reg = Table(
                title="Cross-Decoding Results (continuous behavior)",
                title_style="bold cyan",
                box=box.ROUNDED,
                header_style="bold white",
                caption="Linear regression decoder - R² higher is better, MSE lower is better",
                caption_style="dim",
            )
            reg.add_column("Train", justify="left", style="bold")
            reg.add_column("Test", justify="left", style="bold")
            reg.add_column("R²", justify="right", style="bold")
            reg.add_column("MSE", justify="right")

            rows = conditions + [("noisy_train", "Noisy", "Noisy (train)"), ("denoised_train", "Denoised", "Denoised (train)")]
            for i, (key, train, test) in enumerate(rows):
                reg.add_row(
                    train,
                    test,
                    _num(results[f"r2_{key}"]),
                    _num(results[f"mse_{key}"]),
                    end_section=(i in (1, 3)),
                )

            console.print()
            console.print(reg)
            console.print()
            return

        # ------------------------------------------------------------
        # Main table: scores + permutation test
        # ------------------------------------------------------------
        main = Table(
            title="Cross-Decoding Results",
            title_style="bold cyan",
            box=box.ROUNDED,
            header_style="bold white",
            caption="* p<0.05   ** p<0.01   *** p<0.001   n.s. = not significant   "
                    "(one-sided permutation test on balanced accuracy)",
            caption_style="dim",
        )
        main.add_column("Train", justify="left", style="bold")
        main.add_column("Test", justify="left", style="bold")
        main.add_column("Accuracy", justify="right")
        main.add_column("Bal. acc.", justify="right", style="bold")
        main.add_column("Null mean", justify="right", style="dim")
        main.add_column("Null p95", justify="right", style="dim")
        main.add_column("p-value", justify="right")
        main.add_column("Sig.", justify="center")

        for i, (key, train, test) in enumerate(conditions):
            p_str, stars = _p_cell(results[f"pval_{key}"])
            main.add_row(
                train,
                test,
                _num(results[f"acc_{key}"]),
                _num(results[f"bal_acc_{key}"]),
                _num(results[f"null_mean_{key}"]),
                _num(results[f"null_p95_{key}"]),
                p_str,
                stars,
                # separatore tra decoding "diretto" e cross-decoding
                end_section=(i == 1),
            )

        # ------------------------------------------------------------
        # Training baseline
        # ------------------------------------------------------------
        base = Table(
            title="Training Set Baseline",
            title_style="bold yellow",
            box=box.ROUNDED,
            header_style="bold white",
        )
        base.add_column("Model", justify="left", style="bold")
        base.add_column("Accuracy", justify="right")
        base.add_column("Bal. acc.", justify="right", style="bold")

        for label, key in (("Noisy", "noisy"), ("Denoised", "denoised")):
            base.add_row(
                label,
                _num(results[f"acc_{key}_train"]),
                _num(results[f"bal_acc_{key}_train"]),
            )

        console.print()
        console.print(main)
        console.print()
        console.print(base)
        console.print()


    def print_comparative_results(
        self,
        results: dict,
        base_results: dict,
        name: str = "Results",
        base_name: str = "Base",
        tol: float = 1e-4,
        show_stars: bool = False,
    ):
        """
        Print a side-by-side comparison of two cross-decoding result dictionaries.

        Args:
            results: first result dictionary (as returned by `score`).
            base_results: second result dictionary, used as the reference.
            name: label for `results`.
            base_name: label for `base_results`.
            tol: absolute difference under which two scores count as a tie.
            show_stars: if True, show permutation-test significance stars
                next to the balanced accuracy values.
        """

        def _is_nan(x) -> bool:
            return x is None or np.isnan(x)

        def _stars(p) -> str:
            if _is_nan(p):
                return ""
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return "n.s."

        def _cell(x, p=None) -> str:
            if _is_nan(x):
                return "[dim]n/a[/dim]"
            s = f"{x:.4f}"
            if show_stars and p is not None and not _is_nan(p):
                color = "green" if p < 0.05 else "red"
                s += f" [{color}]{_stars(p)}[/{color}]"
            return s

        def _delta(a, b) -> str:
            if _is_nan(a) or _is_nan(b):
                return "[dim]n/a[/dim]"
            d = a - b
            color = "dim" if abs(d) <= tol else ("green" if d > 0 else "red")
            return f"[{color}]{d:+.4f}[/{color}]"

        def _winner(a, b) -> str:
            if _is_nan(a) or _is_nan(b):
                return "[dim]n/a[/dim]"
            d = a - b
            if abs(d) <= tol:
                return "[dim]= tie[/dim]"
            return (
                f"[bold green]{name}[/bold green]" if d > 0
                else f"[bold magenta]{base_name}[/bold magenta]"
            )

        def _outcome(a, b) -> str | None:
            if _is_nan(a) or _is_nan(b):
                return None
            d = a - b
            return "tie" if abs(d) <= tol else ("first" if d > 0 else "second")

        console = Console()

        conditions = [
            ("noisy_noisy",       "Noisy → Noisy"),
            ("denoised_denoised", "Denoised → Denoised"),
            ("noisy_denoised",    "Noisy → Denoised"),
            ("denoised_noisy",    "Denoised → Noisy"),
        ]

        # ------------------------------------------------------------
        # Main comparison table
        # ------------------------------------------------------------
        caption = f"Δ = {name} - {base_name} (higher is better; |Δ| ≤ {tol:g} = tie)"
        if show_stars:
            caption += "   Stars: one-sided permutation test vs chance, per result"

        main = Table(
            title=f"Cross-Decoding Comparison: {name} vs {base_name}",
            title_style="bold cyan",
            box=box.ROUNDED,
            header_style="bold white",
            caption=caption,
            caption_style="dim",
        )
        main.add_column("Train → Test", justify="left", style="bold")
        main.add_column("Metric", justify="left", style="dim")
        main.add_column(name, justify="right")
        main.add_column(base_name, justify="right")
        main.add_column("Δ", justify="right")
        main.add_column("Winner", justify="center")

        wins = {"acc": [0, 0, 0], "bal_acc": [0, 0, 0]}  # first, second, tie
        idx = {"first": 0, "second": 1, "tie": 2}

        for key, label in conditions:
            for metric, metric_label in (("acc", "Accuracy"), ("bal_acc", "Bal. acc.")):
                a = results.get(f"{metric}_{key}")
                b = base_results.get(f"{metric}_{key}")

                if metric == "bal_acc":
                    cell_a = _cell(a, results.get(f"pval_{key}"))
                    cell_b = _cell(b, base_results.get(f"pval_{key}"))
                    row_label = ""
                else:
                    cell_a, cell_b = _cell(a), _cell(b)
                    row_label = label

                out = _outcome(a, b)
                if out is not None:
                    wins[metric][idx[out]] += 1

                main.add_row(
                    row_label,
                    metric_label,
                    cell_a,
                    cell_b,
                    _delta(a, b),
                    _winner(a, b),
                    end_section=(metric == "bal_acc"),
                )

        # ------------------------------------------------------------
        # Training baseline (no winner: higher train score = more fitting,
        # not necessarily better)
        # ------------------------------------------------------------
        base = Table(
            title="Training Set Baseline",
            title_style="bold yellow",
            box=box.ROUNDED,
            header_style="bold white",
            caption="No winner shown: a higher training score is not necessarily better",
            caption_style="dim",
        )
        base.add_column("Model", justify="left", style="bold")
        base.add_column("Metric", justify="left", style="dim")
        base.add_column(name, justify="right")
        base.add_column(base_name, justify="right")
        base.add_column("Δ", justify="right")

        for label, key in (("Noisy", "noisy"), ("Denoised", "denoised")):
            for metric, metric_label in (("acc", "Accuracy"), ("bal_acc", "Bal. acc.")):
                a = results.get(f"{metric}_{key}_train")
                b = base_results.get(f"{metric}_{key}_train")
                base.add_row(
                    label if metric == "acc" else "",
                    metric_label,
                    _cell(a),
                    _cell(b),
                    _delta(a, b),
                    end_section=(metric == "bal_acc"),
                )

        # ------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="dim", justify="right")
        summary.add_column()
        for metric, metric_label in (("bal_acc", "Balanced accuracy"), ("acc", "Accuracy")):
            w1, w2, t = wins[metric]
            total = w1 + w2 + t
            summary.add_row(
                metric_label,
                f"[bold green]{name} {w1}/{total}[/bold green]   "
                f"[bold magenta]{base_name} {w2}/{total}[/bold magenta]   "
                f"[dim]ties {t}[/dim]",
            )

        w1, w2, _ = wins["bal_acc"]
        if w1 > w2:
            verdict, color = f"{name} is better on balanced accuracy", "green"
        elif w2 > w1:
            verdict, color = f"{base_name} is better on balanced accuracy", "magenta"
        else:
            verdict, color = "No clear winner on balanced accuracy", "yellow"

        console.print()
        console.print(main)
        console.print()
        console.print(base)
        console.print()
        console.print(
            Panel(
                summary,
                title=f"[bold {color}]{verdict}[/bold {color}]",
                title_align="left",
                border_style=color,
                box=box.ROUNDED,
                padding=(1, 2),
                expand=False,
            )
        )
        console.print()



# ============================================================
# Backward-compatible functional API
#
# Thin wrappers around `CrossDecoder`, kept so that existing callers keep
# working unchanged.
# ============================================================

def cross_decoding_score(
    denoiser,
    X_prep,
    B_prep,
    train_indices,
    test_indices,
    behavior_type=None,
    n_null=1000,
    save_path=None
):
    """Functional wrapper around `CrossDecoder.score` (`n_null` = number of permutations)."""
    return CrossDecoder(behavior_type, n_permutations=n_null, save_path=save_path).score(
        denoiser, X_prep, B_prep, train_indices, test_indices
    )


def _compute_cross_decoding(
    X_original_train,
    X_original_test,
    X_denoised_train,
    X_denoised_test,
    B_train,
    B_test,
    n_permutations: int = 1000,
    random_state: int | None = None,
    n_jobs: int = -1
) -> dict:
    """Functional wrapper around `CrossDecoder.score_from_arrays` (discrete behavior)."""
    return CrossDecoder(
        "discrete", n_permutations=n_permutations, random_state=random_state, n_jobs=n_jobs
    ).score_from_arrays(
        X_original_train, X_original_test, X_denoised_train, X_denoised_test, B_train, B_test
    )


print_cross_decoding_results = CrossDecoder.print_results
