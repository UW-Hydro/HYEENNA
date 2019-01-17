# HYEENNA
Hydrologic Entropy Estimators based on Nearest Neighbor Approximations provides estimators for information theoretic
quantities as well as a series of algorithms and analysis tools implemented in pure python.

## Installation
For now, HYEENNA is only available to install from source.  To do so, clone HYEENNA with:

`git clone https://github.com/arbennett/HYEENNA.git`

Then navigate to the HYEENNA directory and install with:

`python setup.py install`

## Usage

HYEENNA provides nearest neighbor based estimators for
 * Shannon Entropy (Single and multivariate cases)
 * Mutual Information
 * Conditional Mutual Information
 * KL Divergence
 * Transfer Entropy
 * Conditional Transfer Entropy

## Examples

We provide several example notebooks in the `notebooks <https://github.com/UW-Hydro/HYEENNA/tree/master/notebooks>__` directory.

## Documentation

See the full documentation at hyeenna.readthedocs.io

## References
.. [0] Goria, M. N., Leonenko, N. N., Mergel, V. V., & Inverardi, P. L. N.
   (2005). A new class of random vector entropy estimators and its
   applications in testing statistical hypotheses. Journal of
   Nonparametric Statistics, 17(3), 277–297.
   https://doi.org/10.1080/104852504200026815

.. [1] - Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
   Estimating mutual information. Physical Review E - Statistical Physics,
   Plasmas, Fluids, and Related Interdisciplinary Topics, 69(6), 16.
   https://doi.org/10.1103/PhysRevE.69.066138

.. [2] - Vlachos, I., & Kugiumtzis, D. (2010).
   Non-uniform state space reconstruction and coupling detection.
   https://doi.org/10.1103/PhysRevE.82.016207

.. [3] - Wang, Q., Kulkarni, S. R., & Verdu, S. (2006). A Nearest-Neighbor
   Approach to Estimating Divergence between Continuous Random Vectors.
   In 2006 IEEE International Symposium on Information Theory.
   https://doi.org/10.1109/ISIT.2006.261842
