import concurrent.futures
import numpy as np

from typing import Callable, List

from ..individual import Individual
from ..population import Population


class MuPlusLambda:
    """Generic (mu + lambda) evolution strategy based on Deb et al. (2002).

    Currently only uses a single objective.

    Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002).
    A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE
    transactions on evolutionary computation, 6(2), 182-197.
    """

    def __init__(
        self,
        n_offsprings: int,
        n_breeding: int,
        tournament_size: int,
        *,
        n_processes: int = 1,
        local_search: Callable[[Individual], None] = lambda combined: None
    ):
        """Init function

        Parameters
        ----------
        n_offsprings : int
            Number of offspring in each iteration.
        n_breeding : int
            Number of parents to use for breeding in each iteration.
        tournament_size : int
            Tournament size in each iteration.
        n_processes : int, optional
            Number of parallel processes to be used. If greater than 1,
            parallel evaluation of the objective is supported. Defaults to 1.
        local_search : Callable[[Individua], None], optional
            Called before each fitness evaluation with a joint list of
            offsprings and parents to optimize numeric leaf values of
            the graph. Defaults to identity function.

        """
        self.n_offsprings = n_offsprings

        if n_breeding < n_offsprings:
            raise ValueError(
                "size of breeding pool must be at least as large "
                "as the desired number of offsprings"
            )
        self.n_breeding = n_breeding

        self.tournament_size = tournament_size
        self.n_processes = n_processes
        self.local_search = local_search

    def initialize_fitness_parents(
        self, pop: Population, objective: Callable[[Individual], Individual]
    ) -> None:
        """Initialize the fitness of all parents in the given population.

        Parameters
        ----------
        pop : Population
            Population instance.
        objective : Callable[[gp.Individual], gp.Individual]
            An objective function used for the evolution. Needs to take an
            individual (Individual) as input parameter and return
            a modified individual (with updated fitness).
        """
        # TODO can we avoid this function? how should a population be
        # initialized?
        pop._parents = self._compute_fitness(pop.parents, objective)

    def step(self, pop: Population, objective: Callable[[Individual], Individual]) -> Population:
        """Perform one step in the evolution.

        Parameters
        ----------
        pop : Population
            Population instance.
        objective : Callable[[gp.Individual], gp.Individual]
            An objective function used for the evolution. Needs to take an
            individual (Individual) as input parameter and return
            a modified individual (with updated fitness).

        Returns
        ----------
        Population
            Modified population with new parents.
        """
        offsprings = self._create_new_offspring_generation(pop)

        # we want to prefer offsprings with the same fitness as their
        # parents as this allows accumulation of silent mutations;
        # since the built-in sort is stable (does not change the order
        # of elements that compare equal)
        # (https://docs.python.org/3.5/library/functions.html#sorted),
        # we can make sure that offsprings preceed parents with
        # identical fitness in the /sorted/ combined population by
        # concatenating the parent population to the offspring
        # population instead of the other way around
        combined = offsprings + pop.parents

        for ind in combined:
            self.local_search(ind)

        combined = self._compute_fitness(combined, objective)

        combined = self._sort(combined)

        pop.parents = self._create_new_parent_population(pop.n_parents, combined)

        return pop

    def _create_new_offspring_generation(self, pop: Population) -> List[Individual]:
        # fill breeding pool via tournament selection from parent
        # population
        breeding_pool: List[Individual] = []
        while len(breeding_pool) < self.n_breeding:
            tournament_pool = pop.rng.permutation(pop.parents)[: self.tournament_size]
            best_in_tournament = sorted(tournament_pool, key=lambda x: -x.fitness)[0]
            breeding_pool.append(best_in_tournament.clone())

        # create offsprings by applying crossover to breeding pool and mutating
        # resulting individuals
        offsprings = pop.crossover(breeding_pool, self.n_offsprings)
        offsprings = pop.mutate(offsprings)
        # TODO this call to pop to label individual is quite ugly, find a
        # better way to track ids of individuals; maybe in the EA?
        pop._label_new_individuals(offsprings)

        return offsprings

    def _compute_fitness(
        self, combined: List[Individual], objective: Callable[[Individual], Individual],
    ) -> List[Individual]:
        # computes fitness on all individuals, objective functions
        # should return immediately if fitness is not None
        if self.n_processes == 1:
            combined = list(map(objective, combined))
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                combined = list(executor.map(objective, combined))

        return combined

    def _sort(self, combined: List[Individual]) -> List[Individual]:
        # create copy of population
        combined_copy = [ind.clone() for ind in combined]

        # replace all nan by -inf to make sure they end up at the end
        # after sorting
        for ind in combined_copy:
            if np.isnan(ind.fitness):
                ind.fitness = -np.inf

        # get list of indices that sorts combined_copy ("argsort")
        combined_sorted_indices = [
            idx for (idx, _) in sorted(enumerate(combined_copy), key=lambda x: -x[1].fitness)
        ]

        # return sorted original list of individuals
        return [combined[idx] for idx in combined_sorted_indices]

    def _create_new_parent_population(
        self, n_parents: int, combined: List[Individual]
    ) -> List[Individual]:
        # create new parent population by picking the `n_parents` individuals
        # with the highest fitness
        parents: List[Individual] = []
        for i in range(n_parents):
            # note: unclear whether clone() is needed here; using
            # clone() avoids accidentally sharing references across
            # individuals, but might incur a performance penalty
            new_individual = combined[i].clone()

            # since this individual is genetically identical to its
            # parent, the identifier is the same
            new_individual.idx = combined[i].idx

            parents.append(new_individual)

        return parents