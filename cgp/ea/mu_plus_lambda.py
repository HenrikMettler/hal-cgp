import multiprocessing as mp
from typing import Callable, List, Union

import numpy as np

from ..individual import IndividualBase
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
        tournament_size: int,
        *,
        n_processes: int = 1,
        local_search: Callable[[IndividualBase], None] = lambda combined: None,
        k_local_search: Union[int, None] = None,
        reorder_genome: bool = False,
        hurdle_percentile: List = [0.0]
    ):
        """Init function

        Parameters
        ----------
        n_offsprings : int
            Number of offspring in each iteration.
        tournament_size : int
            Tournament size in each iteration.
        n_processes : int, optional
            Number of parallel processes to be used. If greater than 1,
            parallel evaluation of the objective is supported. Defaults to 1.
        local_search : Callable[[Individua], None], optional
            Called before each fitness evaluation with a joint list of
            offsprings and parents to optimize numeric leaf values of
            the graph. Defaults to identity function.
        k_local_search : int
            Number of individuals in the whole population (parents +
            offsprings) to apply local search to.
       reorder_genome : bool, optional
            Whether genome reordering should be applied.
            Reorder shuffles the genotype of an individual without changing its phenotype,
            thereby contributing to neutral drift through the genotypic search space.
            If True, reorder is applied to each parents genome at every generation
            before creating offsprings.
            Defaults to True.
        hurdle_percentile : List[float], optional
            Specifies which percentile of individuals passes the
            respective hurdle, i.e., is evaluated on the next
            objective when providing a list of objectives to be
            evaluated sequentially.
        """
        self.n_offsprings = n_offsprings

        self.tournament_size = tournament_size
        self.n_processes = n_processes
        self.local_search = local_search
        self.k_local_search = k_local_search
        self.reorder_genome = reorder_genome
        self.hurdle_percentile = hurdle_percentile

        self.process_pool: Union[None, "mp.pool.Pool"]
        if self.n_processes > 1:
            self.process_pool = mp.Pool(processes=self.n_processes)
        else:
            self.process_pool = None
        self.n_objective_calls: int = 0

    def __del__(self):
        if self.n_processes > 1:
            self.process_pool.close()

    def initialize_fitness_parents(
        self, pop: Population, objective: Callable[[IndividualBase], IndividualBase]
    ) -> None:
        """Initialize the fitness of all parents in the given population.

        Parameters
        ----------
        pop : Population
            Population instance.
        objective : Callable[[gp.IndividualBase], gp.IndividualBase]
            An objective function used for the evolution. Needs to take an
            individual (IndividualBase) as input parameter and return
            a modified individual (with updated fitness).
        """
        # TODO can we avoid this function? how should a population be
        # initialized?
        pop._parents = self._compute_fitness(pop.parents, objective, use_hurdles=False)

    def step(
        self, pop: Population, objective: Callable[[IndividualBase], IndividualBase],
    ) -> Population:
        """Perform one step in the evolution.

        Parameters
        ----------
        pop : Population
            Population instance.
        objective : Callable[[gp.IndividualBase], gp.IndividualBase]
            An objective function used for the evolution. Needs to take an
            individual (IndividualBase) as input parameter and return
            a modified individual (with updated fitness).

        Returns
        ----------
        Population
            Modified population with new parents.
        """

        if self.reorder_genome:
            pop.reorder_genome()

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

        # we follow a two-step process for selection of new parents:
        # we first determine the fitness for all individuals, then, if
        # applicable, we apply local search to the k_local_search
        # fittest individuals; after this we need to recompute the
        # fitness for all individuals for which parameters changed
        # during local search; finally we sort again by fitness, now
        # taking into account the effect of local search for
        # subsequent selection
        combined = self._compute_fitness(combined, objective)
        combined = self._sort(combined)

        n_total = self.n_offsprings + pop.n_parents
        k_local_search = n_total if self.k_local_search is None else self.k_local_search
        for idx in range(k_local_search):
            self.local_search(combined[idx])

        combined = self._compute_fitness(combined, objective)
        combined = self._sort(combined)

        pop.parents = self._create_new_parent_population(pop.n_parents, combined)

        return pop

    def _create_new_offspring_generation(self, pop: Population) -> List[IndividualBase]:
        # use tournament selection to randomly select individuals from
        # parent population

        if self.tournament_size > pop.n_parents:
            raise ValueError("tournament_size must be less or equal n_parents")

        offsprings: List[IndividualBase] = []
        while len(offsprings) < self.n_offsprings:
            tournament_pool = pop.rng.permutation(pop.parents)[: self.tournament_size]
            best_in_tournament = sorted(tournament_pool, reverse=True)[0]
            offsprings.append(best_in_tournament.clone())

        # mutate individuals to create offsprings
        offsprings = pop.mutate(offsprings)

        for ind in offsprings:
            ind.idx = pop.get_idx_for_new_individual()

        return offsprings

    def _compute_fitness(
        self,
        combined: List[IndividualBase],
        objective: Union[
            Callable[[IndividualBase], IndividualBase],
            List[Callable[[IndividualBase], IndividualBase]],
        ],
        use_hurdles=True,
    ) -> List[IndividualBase]:
        def compute_fitness_hurdle(ind_evaluating: List[IndividualBase]) -> float:
            return np.percentile(
                np.unique([ind.fitness_current_objective for ind in ind_evaluating]),
                self.hurdle_percentile[ind_evaluating[0].objective_idx] * 100,
            )

        self.update_n_objective_calls(combined)

        if callable(objective):
            objective = [objective]

        ind_evaluating = list(combined)
        ind_done_evaluating = []
        for obj_idx, obj in enumerate(objective):

            # prime individuals for receiving fitness for this objective
            for ind in ind_evaluating:
                ind.objective_idx = obj_idx

            if self.n_processes == 1:
                # don't use the process pool if running just a single
                # process to avoid any multiprocessing-associated overhead
                ind_evaluating = list(map(obj, ind_evaluating))
            else:
                assert isinstance(self.process_pool, mp.pool.Pool)
                ind_evaluating = self.process_pool.map(obj, ind_evaluating)

            fitness_hurdle = compute_fitness_hurdle(ind_evaluating)

            ind_evaluating_new = []
            for ind in ind_evaluating:
                assert isinstance(ind.fitness_current_objective, float)
                if not use_hurdles or ind.fitness_current_objective >= fitness_hurdle:
                    ind_evaluating_new.append(ind)
                else:
                    ind_done_evaluating.append(ind)
            ind_evaluating = ind_evaluating_new

        combined = ind_done_evaluating + [ind for ind in ind_evaluating]
        return combined

    def _sort(self, combined: List[IndividualBase]) -> List[IndividualBase]:
        return sorted(combined, reverse=True)

    def _create_new_parent_population(
        self, n_parents: int, combined: List[IndividualBase]
    ) -> List[IndividualBase]:
        """Create the new parent population by picking the first `n_parents`
        individuals from the combined population.

        """
        return combined[:n_parents]

    def update_n_objective_calls(self, combined: List[IndividualBase]) -> None:
        """Increase n_objective_calls by the number of individuals with fitness=None,
         i.e., for which the objective function will be evaluated.
        """
        for individual in combined:
            if individual.fitness_is_None():
                self.n_objective_calls += 1
