import pytest

from sabc._src.proposals import DiffEvolution
from sabc._src.schedule import MultiEps, SingleEps


def test_single_eps_carries_v():
  assert SingleEps(v=2.0).v == 2.0
  assert SingleEps().algorithm == "single_eps"


def test_multi_eps_algorithm_name():
  assert MultiEps().algorithm == "multi_eps"


def test_single_eps_rejects_nonpositive_v():
  with pytest.raises(ValueError):
    SingleEps(v=0.0)


def test_diffevolution_gamma0_optional():
  assert DiffEvolution().gamma0 is None
  assert DiffEvolution(gamma0=0.5).gamma0 == 0.5
