# MED-MAGMA

The code to run MED-MAGMA is contained in `MED_MAGMA.py`.  Below is a minimal example of how to run it:

```{python}
from MED_MAGMA import quotient, EM_algorithm, sparse_quotient

# Let X be your dataset
X: np.ndarray

# Map it to the quotient space
# If X contains zeros, use `sparse_quotient` instead
Y = quotient(X)

# Run our algorithm; EM_results contains the learned graphs, magnitudes contains the learned "r" values if those are of interest
EM_results, magnitudes = EM_algorithm(Y, max_iter = 100, verbose=False, very_verbose=False, stop_tol=1e-7, regularization=0)
```

In the paper we mentioned briefly that one could turn the $\mathbf{z}^*$ estimation portion of our algorithm into a convex problem by using a simplex constraint rather than our geometric mean constraint (but that this would introduce bias into the algorithm that made it unsuitable).  If you wish to check this out for yourself, we have `MED_MAGMA_simplex.py` which has the same API as `MED_MAGMA.py`.  What you'll find is that it works for synthetic data, but on most real datasets it just learns an essentially constant matrix (which manifests from the aforementioned bias towards the simplex's corners - which dominates when signal-to-noise is lower).

## Package

This algorithm is available on PyPI [https://pypi.org/project/MED-MAGMA/1.0.0/](https://pypi.org/project/MED-MAGMA/1.0.0/).

### Dependencies

This code was written in Python 3.13 with NumPy 2.3 and SciPy 1.16.

The exact environment used to run the paper's experiments is given in `environment.yaml`, but to just run our algorithm only Python, NumPy, and SciPy are required.  (`MED_MAGAM_simplex` requires cvxpy as well).

## Paper

The figures for the paper are generated in the following:

* Figure 1 (Tail Dependence): `median-data-experiment.ipynb`
* Figure 2 (Synthetic Validation): `synthetic-experiments-paper.ipynb`
* Figure 3 (Real Validation): `all-real-data-all.ipynb`
* Figure 4 (UMAP): `median-data-experiment.ipynb`
* Figure 5 (Supplementary, Synthetic Validation): `synthetic-experiments-paper.ipynb`
* Figure 6 (Supplementary, Effect of Sparsity): `synthetic-experiments-paper.ipynb`

All further empirical values given in the paper are from `median-data-experiment.ipynb`, except for robustness using `Robin`; this is computed in `robin-on-median.rmd` (an R markdown file).

The conda environment used to generate these results is given in `environment.yaml`.