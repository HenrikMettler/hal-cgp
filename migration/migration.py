#import multiprocessing as mp
from typing import List

import numpy as np

import cgp

n_populations = 2
migration_interval = 10  # wild guess


def f_target(x: List):
    # Rosenbroeck function (https://en.wikipedia.org/wiki/Rosenbrock_function)
    a = 1
    b = 100
    return (a - x[0]) ** 2 + b(x[1] - x[0] ** 2) ** 2


def objective(individual):

    if not individual.fitness_is_None():
        return individual

    n_function_evaluations = 1000

    np.random.seed(1234)

    f = individual.to_func()
    loss = 0
    for x in np.random.uniform(-4, 4, n_function_evaluations):
        # the callable returned from `to_func` accepts and returns
        # lists; accordingly we need to pack the argument and unpack
        # the return value
        y = f([x])[0]
        loss += (f_target([x]) - y) ** 2

    individual.fitness = -loss / n_function_evaluations

    return individual


population_1_params = {"n_parents": 1, "seed": 8188211}
population_2_params = {"n_parents": 1, "seed": 8188212}

genome_params = {
    "n_inputs": 2,
    "n_outputs": 1,
    "n_columns": 12,
    "n_rows": 1,
    "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.ConstantFloat),
}

# Sole difference between the populations: mutation rate
# (poss. alternative: one with local ES search, other with gradient)
ea_params_1 = {"n_offsprings": 4, "mutation_rate": 0.01, "n_processes": 1}
ea_params_2 = {"n_offsprings": 4, "mutation_rate": 0.1, "n_processes": 1}

evolve_params = {"max_generations": migration_interval, "min_fitness": 0.0}
# max_generations is no longer the true max_gen, but instead mu, the migration interval

pop_1 = cgp.Population(**population_1_params, genome_params=genome_params)
pop_2 = cgp.Population(**population_2_params, genome_params=genome_params)

ea_1 = cgp.ea.MuPlusLambda(**ea_params_1)
ea_2 = cgp.ea.MuPlusLambda(**ea_params_2)

history = {}
history["fitness_champion"] = []


def recording_callback(pop):
    history["fitness_champion"].append(pop.champion.fitness)


cgp.evolve(pop_1, objective, ea_1, **evolve_params, print_progress=True, callback=recording_callback)
