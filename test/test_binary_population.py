import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '../')
import gp

params = {
    'seed': 1234,
    'n_parents': 5,
    'n_offsprings': 5,
    'genome_length': 10,
    'generations': 100,
    'n_breeding': 5,
    'tournament_size': 6,
    'mutation_rate': 0.1,
}


def test_binary_population():

    np.random.seed(params['seed'])

    target_sequence = str(np.random.randint(10 ** params['genome_length'])).zfill(params['genome_length'])

    def objective(individual):

        if individual.fitness is not None:
            return individual

        individual.fitness = np.sum(1 if individual.genome[i] == target_sequence[i] else 0
                                    for i in range(len(target_sequence)))
        return individual

    # create population object that will be evolved
    pop = gp.BinaryPopulation(
        params['n_parents'], params['n_offsprings'], params['genome_length'], params['n_breeding'],
        params['tournament_size'], params['mutation_rate'])

    # generate initial parent population of size N
    pop.generate_random_parent_population()

    # generate initial offspring population of size N
    pop.generate_random_offspring_population()

    # perform evolution
    history_fitness = []
    for i in range(params['generations']):

        # combine parent and offspring populations
        pop.create_combined_population()

        #  compute fitness for all objectives for all individuals
        pop.compute_fitness(objective)

        # sort population according to fitness & crowding distance
        pop.sort()

        # fill new parent population according to sorting
        pop.create_new_parent_population()

        # generate new offspring population from parent population
        pop.create_new_offspring_population()

        history_fitness.append(pop.fitness)

    for ind in pop.parents:
        assert(target_sequence == ind.genome)

    # plt.plot(np.mean(history_fitness, axis=1))
    # plt.show()
