import agentpy as ap
import numpy as np

class MarketStatistician(ap.Agent):

    def setup(self):
        self.utility = 0

    def U(self: ap.Agent, consumption: int) -> float:
        """Return the CARA utility of consumption."""
        return (-np.exp(-self.p.dorra * consumption))
