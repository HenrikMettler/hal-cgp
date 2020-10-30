__version__ = "0.2.0dev"
__maintainer__ = "Jakob Jordan, Maximilian Schmidt"
__author__ = "Happy Algorithms League"
__license__ = "GPLv3"
__description__ = "Cartesian genetic programming (CGP) in pure Python."
__url__ = "https://happy-algorithms-league.github.io/hal-cgp/"
__doc__ = f"{__description__} <{__url__}>"

from . import ea, local_search, utils
from .cartesian_graph import CartesianGraph
from .genome import Genome
from .hl_api import evolve
from .individual import IndividualMultiGenome, IndividualSingleGenome
from .node import OperatorNode
from .node_impl import Add, ConstantFloat, Div, Mul, Parameter, Pow, Sub
from .population import Population
from .utils import objective