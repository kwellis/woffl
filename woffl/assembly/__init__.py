"""Assembly Folder

The assembly folder is for the collection of functions and classes that are
used to analyze the entire Well Assembly. The assembly consists of an IPR, Jet Pump
Well Bore, Well Profile and Surface Constraints. Each assembly is unique and possesses
its own set of unique results to be analyzed.
"""

from .batchpump import BatchPump
from .network import WellNetwork
