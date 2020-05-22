import pytest

import gp
from gp.genome import ID_INPUT_NODE, ID_OUTPUT_NODE, ID_NON_CODING_GENE


def test_gradient_based_step_towards_maximum():
    torch = pytest.importorskip("torch")

    primitives = (gp.Parameter,)
    genome = gp.Genome(1, 1, 1, 1, 1, primitives)
    # f(x) = c
    genome.dna = [ID_INPUT_NODE, ID_NON_CODING_GENE, 0, 0, ID_OUTPUT_NODE, 1]
    ind = gp.individual.Individual(None, genome)

    def objective(f):
        x_dummy = torch.zeros((1, 1), dtype=torch.double)  # not used
        target_value = torch.ones((1, 1), dtype=torch.double)
        return torch.nn.MSELoss()(f(x_dummy), target_value)

    # test increase parameter value if too small
    ind.parameter_names_to_values["<p1>"] = 0.9
    gp.local_search.gradient_based(ind, objective, 0.05, 1)
    assert ind.parameter_names_to_values["<p1>"] == pytest.approx(0.91)

    # test decrease parameter value if too large
    ind.parameter_names_to_values["<p1>"] = 1.1
    gp.local_search.gradient_based(ind, objective, 0.05, 1)
    assert ind.parameter_names_to_values["<p1>"] == pytest.approx(1.09)

    # test no change of parameter value if at optimum
    ind.parameter_names_to_values["<p1>"] = 1.0
    gp.local_search.gradient_based(ind, objective, 0.05, 1)
    assert ind.parameter_names_to_values["<p1>"] == pytest.approx(1.0)
