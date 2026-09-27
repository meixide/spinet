"""Stability selection that survives a numerically unstable subsample.

LassoNet's base training loop asserts when the training loss becomes NaN
(``lassonet/interfaces.py``, inside the ``closure`` passed to the optimizer).
During stability selection that assert propagates out of
``stability_selection`` and aborts the whole run, discarding every path that
had already been computed.

The failure is a property of the individual subsample, not of the data set:
``_stability_selection_path`` draws its half sample with ``random.shuffle``, so
simply drawing again is enough. This wrapper redraws instead of aborting and
keeps the return signature of ``BaseLassoNet.stability_selection``.
"""

import warnings

import torch
from lassonet.utils import selection_probability
from tqdm import tqdm

WRONG_THRESHOLD = 1 / 2


def stability_selection(model, X, y, n_models=20, max_retries=10):
    """Drop-in replacement for ``model.stability_selection(X, y, n_models)``.

    Parameters
    ----------
    model : BaseLassoNet
        Estimator to compute the paths with.
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target values.
    n_models : int
        Number of subsampled paths to average over.
    max_retries : int
        How many times a single subsample may be redrawn after a NaN loss
        before giving up. Retries are counted across the whole run.

    Returns
    -------
    Same five values as ``BaseLassoNet.stability_selection``:
    ``(oracle, order, wrong, paths, prob)``.
    """
    n_retries = 0

    def draw_path(lambda_seq=None):
        nonlocal n_retries
        for _ in range(max_retries + 1):
            try:
                return model._stability_selection_path(X, y, lambda_seq)
            except AssertionError:
                n_retries += 1
        raise RuntimeError(
            f"{max_retries + 1} consecutive subsamples produced a NaN loss. "
            "This is more than the occasional unstable draw, so something "
            "else is wrong: check that X is standardized and that y has no "
            "zero or non-finite times."
        )

    # the first path fixes the lambda grid that all the others reuse
    lambda_seq = [it.lambda_ for it in draw_path()]
    paths = [
        draw_path(lambda_seq)
        for _ in tqdm(range(n_models), desc="Stability selection")
    ]

    if n_retries:
        warnings.warn(
            f"redrew {n_retries} subsample(s) after a NaN loss "
            f"while computing {n_models + 1} paths",
            stacklevel=2,
        )

    prob, expected_wrong = selection_probability(paths)
    order = expected_wrong.indices
    wrong = expected_wrong.values

    if wrong[0] > WRONG_THRESHOLD:
        oracle = 0
    else:
        oracle = int(torch.argmax(wrong[1:] / wrong[:-1])) + 1

    return oracle, order, wrong, paths, prob
