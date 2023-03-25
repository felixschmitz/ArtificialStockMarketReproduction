import agentpy as ap
import numpy as np

class MarketStatistician(ap.Agent):

    def setup(self):
        self.utility = 0
        if self.p.mode == 1:
            self.slope, self.intercept, self.pdVariance = self.model.hreeSlope, self.model.hreeIntercept, self.model.hreeVariance        

    def U(self: ap.Agent, consumption: int) -> float:
        '''Return the CARA utility of consumption.'''
        return (-np.exp(-self.p.dorra * consumption))
    
    def price_prediction(self: ap.Agent) -> float:
        if self.p.mode == 1:
            price = self.model.market_clearing_price(self.model.hreeSlope, self.model.hreeIntercept, self.model.hreeVariance)
            return self.slope * (price + self.model.dividend) + self.intercept, self.pdvariance
        