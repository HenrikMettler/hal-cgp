import numpy as np
import pytest

import cgp
from cgp.genome import ID_INPUT_NODE, ID_OUTPUT_NODE, ID_NON_CODING_GENE
from pytest import fixture


def test_check_dna_consistency():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1, "levels_back": 1}

    primitives = (cgp.Add,)
    genome = cgp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
    )
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,
        0,
        1,
        ID_OUTPUT_NODE,
        0,
        ID_NON_CODING_GENE,
    ]

    # invalid length
    with pytest.raises(ValueError):
        genome.dna = [
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            0,
            ID_OUTPUT_NODE,
            ID_INPUT_NODE,
            ID_OUTPUT_NODE,
            0,
            ID_NON_CODING_GENE,
            0,
        ]

    # invalid function gene for input node
    with pytest.raises(ValueError):
        genome.dna = [
            0,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            0,
            ID_OUTPUT_NODE,
            0,
            ID_OUTPUT_NODE,
            0,
            ID_NON_CODING_GENE,
        ]

    # invalid input gene for input node
    with pytest.raises(ValueError):
        genome.dna = [
            ID_INPUT_NODE,
            0,
            ID_NON_CODING_GENE,
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            0,
            ID_OUTPUT_NODE,
            0,
            ID_OUTPUT_NODE,
            0,
            ID_NON_CODING_GENE,
        ]

    # invalid function gene for hidden node
    with pytest.raises(ValueError):
        genome.dna = [
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            2,
            0,
            1,
            ID_OUTPUT_NODE,
            0,
            ID_NON_CODING_GENE,
        ]

    # invalid input gene for hidden node
    with pytest.raises(ValueError):
        genome.dna = [
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            0,
            2,
            1,
            ID_OUTPUT_NODE,
            0,
            ID_NON_CODING_GENE,
        ]

    # invalid function gene for output node
    with pytest.raises(ValueError):
        genome.dna = [
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            0,
            0,
            1,
            0,
            0,
            ID_NON_CODING_GENE,
        ]

    # invalid input gene for input node
    with pytest.raises(ValueError):
        genome.dna = [
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            0,
            0,
            1,
            ID_OUTPUT_NODE,
            3,
            ID_NON_CODING_GENE,
        ]

    # invalid inactive input gene for output node
    with pytest.raises(ValueError):
        genome.dna = [
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            0,
            0,
            1,
            ID_OUTPUT_NODE,
            0,
            0,
        ]


def test_permissible_inputs():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 4, "n_rows": 3, "levels_back": 2}

    primitives = (cgp.Add,)
    genome = cgp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
    )
    genome.randomize(np.random)

    for input_idx in range(params["n_inputs"]):
        region_idx = input_idx
        with pytest.raises(AssertionError):
            genome._permissible_inputs(region_idx)

    expected_for_hidden = [
        [0, 1],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 5, 6, 7, 8, 9, 10],
    ]

    for column_idx in range(params["n_columns"]):
        region_idx = params["n_inputs"] + params["n_rows"] * column_idx
        assert expected_for_hidden[column_idx] == genome._permissible_inputs(region_idx)

    expected_for_output = list(range(params["n_inputs"] + params["n_rows"] * params["n_columns"]))

    for output_idx in range(params["n_outputs"]):
        region_idx = params["n_inputs"] + params["n_rows"] * params["n_columns"] + output_idx
        assert expected_for_output == genome._permissible_inputs(region_idx)


def test_region_iterators():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1, "levels_back": 1}

    primitives = (cgp.Add,)
    genome = cgp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
    )
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,
        0,
        1,
        ID_OUTPUT_NODE,
        0,
        ID_NON_CODING_GENE,
    ]

    for region_idx, region in genome.iter_input_regions():
        assert region == [ID_INPUT_NODE, ID_NON_CODING_GENE, ID_NON_CODING_GENE]

    for region_idx, region in genome.iter_hidden_regions():
        assert region == [0, 0, 1]

    for region_idx, region in genome.iter_output_regions():
        assert region == [ID_OUTPUT_NODE, 0, ID_NON_CODING_GENE]


def test_check_levels_back_consistency():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 4, "n_rows": 3, "levels_back": None}

    primitives = (cgp.Add,)

    params["levels_back"] = 0
    with pytest.raises(ValueError):
        cgp.Genome(
            params["n_inputs"],
            params["n_outputs"],
            params["n_columns"],
            params["n_rows"],
            params["levels_back"],
            primitives,
        )

    params["levels_back"] = params["n_columns"] + 1
    with pytest.raises(ValueError):
        cgp.Genome(
            params["n_inputs"],
            params["n_outputs"],
            params["n_columns"],
            params["n_rows"],
            params["levels_back"],
            primitives,
        )

    params["levels_back"] = params["n_columns"] - 1
    cgp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
    )


def test_catch_invalid_allele_in_inactive_region():
    primitives = (cgp.ConstantFloat,)
    genome = cgp.Genome(1, 1, 1, 1, 1, primitives)

    # should raise error: ConstantFloat node has no inputs, but silent
    # input gene should still specify valid input
    with pytest.raises(ValueError):
        genome.dna = [ID_INPUT_NODE, ID_NON_CODING_GENE, 0, ID_NON_CODING_GENE, ID_OUTPUT_NODE, 1]

    # correct
    genome.dna = [ID_INPUT_NODE, ID_NON_CODING_GENE, 0, 0, ID_OUTPUT_NODE, 1]


def test_individuals_have_different_genome(population_params, genome_params, ea_params):
    def objective(ind):
        ind.fitness = 1.0
        return ind

    pop = cgp.Population(**population_params, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(**ea_params)

    pop._generate_random_parent_population()

    ea.initialize_fitness_parents(pop, objective)

    ea.step(pop, objective)

    for i, parent_i in enumerate(pop._parents):
        for j, parent_j in enumerate(pop._parents):
            if i != j:
                assert parent_i is not parent_j
                assert parent_i.genome is not parent_j.genome
                assert parent_i.genome.dna is not parent_j.genome.dna


def test_is_gene_in_input_region(rng_seed):
    genome = cgp.Genome(2, 1, 2, 1, None, (cgp.Add,))
    rng = np.random.RandomState(rng_seed)
    genome.randomize(rng)

    assert genome._is_gene_in_input_region(0)
    assert not genome._is_gene_in_input_region(6)


def test_is_gene_in_hidden_region(rng_seed):
    genome = cgp.Genome(2, 1, 2, 1, None, (cgp.Add,))
    rng = np.random.RandomState(rng_seed)
    genome.randomize(rng)

    assert genome._is_gene_in_hidden_region(6)
    assert genome._is_gene_in_hidden_region(9)
    assert not genome._is_gene_in_hidden_region(5)
    assert not genome._is_gene_in_hidden_region(12)


def test_is_gene_in_output_region(rng_seed):
    genome = cgp.Genome(2, 1, 2, 1, None, (cgp.Add,))
    rng = np.random.RandomState(rng_seed)
    genome.randomize(rng)

    assert genome._is_gene_in_output_region(12)
    assert not genome._is_gene_in_output_region(11)


def test_mutation_rate(rng_seed, mutation_rate):
    n_input = 1
    n_output = 1
    n_column = 4
    n_row = 3
    genome = cgp.Genome(n_input, n_output, n_column, n_row, None, (cgp.Add, cgp.Sub))
    rng = np.random.RandomState(rng_seed)
    genome.randomize(rng)

    def count_n_immutable_genes(n_input, n_output, n_row):
        counter = n_input * 3  # 3 is the arity
        counter += n_output * 2  # 2 of 3 output genes can not be mutated
        if n_input == 1:
            counter += (
                n_row * 2
            )  # input gene addresses in the first hidden layer can't be mutated in that case
        return counter

    def count_mutations(prev_dna, new_dna):
        counter = 0
        for idx in range(len(prev_dna)):
            if prev_dna[idx] != new_dna[idx]:
                counter += 1
        return counter

    acceptable_error_interval = 0.05  # Todo: get comment for this value from Jakob and Max
    n = 1000
    n_immutable_genes = count_n_immutable_genes(n_input, n_output, n_row)
    n_mutations_expected = n * mutation_rate * (len(genome.dna) - n_immutable_genes)
    print(len(genome.dna), n_immutable_genes)
    n_mutations = 0
    for _ in range(n):
        temp_dna = genome.dna
        genome.mutate(mutation_rate, rng)
        current_n_mutations = count_mutations(temp_dna, genome.dna)
        n_mutations += current_n_mutations
    print(n_mutations, n_mutations_expected, len(genome.dna))
    assert abs(n_mutations_expected - n_mutations) < (
        n_mutations_expected * acceptable_error_interval
    )


def test_only_silent_mutations(genome_params, rng_seed):
    genome = cgp.Genome(**genome_params)
    rng = np.random.RandomState(rng_seed)
    genome.randomize(rng)

    only_silent_mutations_0 = genome.mutate(mutation_rate=0, rng=rng)
    assert only_silent_mutations_0

    only_silent_mutations_1 = genome.mutate(mutation_rate=1, rng=rng)
    assert not only_silent_mutations_1


# Overwrite global genome_params() from conftest,
# since tests below strongly depend on these variable values
@fixture
def genome_params():
    return {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 3,
        "n_rows": 3,
        "levels_back": 2,
        "primitives": (cgp.Add, cgp.Sub, cgp.ConstantFloat),
    }


def test_permissible_values_hidden(genome_params, rng_seed):
    genome = cgp.Genome(**genome_params)

    # test function gene
    gene_idx = 6  # function gene of first hidden gene
    gene = 0  #
    region_idx = 1000  # can be any number, since not touched for function gene
    permissible_function_gene_values = genome._determine_permissible_values_hidden(
        gene_idx, gene, region_idx
    )

    # test input gene
    gene_idx = 16  # first input gene in second hidden layer
    gene = 0
    region_idx = 5
    permissible_hidden_input_gene_values = genome._determine_permissible_values_hidden(
        gene_idx, gene, region_idx
    )

    assert permissible_function_gene_values == [
        1,
        2,
    ]  # function idx 1,2 (cgp.Sub, cgp.ConstantFloat)
    assert permissible_hidden_input_gene_values == [
        1,
        2,
        3,
        4,
    ]  # gene indices of input, 1st hidden layer


def test_permissible_values_output(genome_params, rng_seed):
    genome = cgp.Genome(**genome_params)

    gene_idx_0 = 33
    gene_idx_1 = 34
    gene_idx_2 = 35

    gene = 5  # possible input genes: 0-10

    permissible_values_0 = genome._determine_permissible_values_output(gene_idx_0, gene)
    permissible_values_1 = genome._determine_permissible_values_output(gene_idx_1, gene)
    permissible_values_2 = genome._determine_permissible_values_output(gene_idx_2, gene)
    print(permissible_values_1)
    assert permissible_values_0 == []
    assert permissible_values_1 == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]
    assert permissible_values_2 == []
