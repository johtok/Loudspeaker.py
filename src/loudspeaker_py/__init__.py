"""Main package for Loudspeaker Py CFES framework."""

__version__ = "0.1.0"
__author__ = "Johannes Nørskov Toke"
__email__ = "johannestoke@gmail.com"

# Core modules
from . import models
from . import solvers  
from . import utils
from . import data
from . import visualization

__all__ = ["models", "solvers", "utils", "data", "visualization"]