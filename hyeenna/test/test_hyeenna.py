import pytest

import hyeenna as hy
import numpy as np

np.random.seed(42)

uniform_dists = [3, 9]
pareto_alpha = [1, 1.5]
rayleigh_sigma = [1, 3]
normal_params = [(0, 1), (3, 6)]

k_counts = [15, 100, 250]
sample_size = [1000, 5000, 50000]
FULL_SAMPLE = np.max(sample_size)
ENTROPY_ERR_THRESH = 0.1


class Helpers:
    @staticmethod
    def test_entropy(full_sample, analytic, sample_size, neighbors):
        sub_sample = np.random.choice(full_sample, sample_size)
        test_entropy = hy.entropy(sub_sample, k=neighbors)
        percent_err = np.abs((test_entropy - analytic)/analytic)
        assert percent_err <= ENTROPY_ERR_THRESH


@pytest.fixture()
def helpers():
    return Helpers


# -------- entropy testing ---------------------------------------------------
# -------- ENTROPY TESTING ---------------------------------------------------
# -------- entropy testing ---------------------------------------------------


@pytest.fixture(params=k_counts)
def neighbors(request):
    return request.param


@pytest.fixture(params=sample_size)
def sample_size(request):
    return request.param


@pytest.fixture(params=uniform_dists)
def uniform(request):
    width = request.param
    sample = width * np.random.random(FULL_SAMPLE)
    h_analytic = np.log(width)
    return sample, h_analytic


@pytest.fixture(params=pareto_alpha)
def pareto(request):
    alpha = request.param
    sample = np.random.pareto(alpha, size=FULL_SAMPLE)
    h_analytic = np.log(1/alpha) + 1 + (1/alpha)
    return sample, h_analytic


@pytest.fixture(params=rayleigh_sigma)
def rayleigh(request):
    sigma = request.param
    sample = np.random.rayleigh(scale=sigma, size=FULL_SAMPLE)
    h_analytic = 1 + np.log(sigma/np.sqrt(2)) + (np.euler_gamma/2)
    return sample, h_analytic


@pytest.fixture(params=normal_params)
def normal(request):
    u, s = request.param
    sample = np.random.normal(loc=u, scale=s, size=FULL_SAMPLE)
    h_analytic = np.log(s * np.sqrt(2*np.pi*np.e))
    return sample, h_analytic


def test_uniform_entropy(helpers, uniform, sample_size, neighbors):
    helpers.test_entropy(uniform[0], uniform[1], sample_size, neighbors)


def test_pareto_entropy(helpers, pareto, sample_size, neighbors):
    helpers.test_entropy(pareto[0], pareto[1], sample_size, neighbors)


def test_rayleigh_entropy(helpers, rayleigh, sample_size, neighbors):
    helpers.test_entropy(rayleigh[0], rayleigh[1], sample_size, neighbors)


def test_normal_entropy(helpers, normal, sample_size, neighbors):
    helpers.test_entropy(normal[0], normal[1], sample_size, neighbors)


# -------- mutual information testing -----------------------------------------
# -------- MUTUAL INFORMATION TESTING -----------------------------------------
# -------- mutual information testing -----------------------------------------


# -------- kl divergence testing ----------------------------------------------
# -------- KL DIVERGENCE TESTING ----------------------------------------------
# -------- kl divergence testing ----------------------------------------------


# -------- transfer entropy testing -------------------------------------------
# -------- TRANSFER ENTROPY TESTING -------------------------------------------
# -------- transfer entropy testing -------------------------------------------


