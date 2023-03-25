from agents import MarketStatistician as MS

import agentpy as ap
import numpy as np
import math

class ArtificialStockMarket(ap.Model):
    """def __init__(self):
        pass"""

    def setup(self: ap.Model):
        self.dividend = self.p.averageDividend
        self.hreeSlope, self.hreeIntercept, self.hreeVariance = self.hree_values()
        self.hreePrice = self.hree_price()
        self.agents = ap.AgentList(self, self.p.N, MS)

    def step(self: ap.Model):
        #print(self.agents.U(10))
        self.dividend = self.dividend_process()
        self.hreePrice = self.hree_price()
        self.marketPrice = self.market_clearing_price()
        self.document()

    def document(self: ap.Model):
        self.record(self.vars)

    def dividend_process(self: ap.Model) -> float:
        '''Returns dividend based on AR(1) process given average dividend, autoregressive parameter, and error variance.'''
        errorTerm = self.nprandom.normal(0, math.sqrt(self.p.errorVar))
        return self.p.averageDividend + self.p.autoregressiveParam * (self.dividend - self.p.averageDividend) + errorTerm
    
    def hree_values(self: ap.Model, a_min: float=0.7, a_max: float=1.2, b_min: float=-10, b_max: float=19.002):
        '''Returns homogeneous rational expectations equilibrium predictos values.'''
        return self.nprandom.uniform(a_min, a_max), self.nprandom.uniform(b_min, b_max), self.p.initialPriceDividendVariance
    
    def hree_price(self: ap.Model) -> float:
        '''Returns homogeneous rational expectation equilibrium price.'''
        f = self.p.autoregressiveParam / (1 + self.p.interestRate - self.p.autoregressiveParam)
        g = ((1 + f) * ((1 - self.p.autoregressiveParam) * self.p.averageDividend - self.p.dorra * self.p.errorVar)) / self.p.interestRate
        return f * self.dividend + g
    
    def market_clearing_price(self: ap.Model) -> float:
        '''Returns inductive market clearing price.'''
        numerator = self.dividend * (sum(self.agents.slope) / sum(self.agents.pdVariance)) + (sum(self.agents.intercept) / sum(self.agents.pdVariance)) - self.p.N * self.p.dorra
        denominator = ((1 + self.p.interestRate) * (1 / sum(self.agents.pdVariance)) - (sum(self.agents.slope) / sum(self.agents.pdVariance)))
        return numerator / denominator
