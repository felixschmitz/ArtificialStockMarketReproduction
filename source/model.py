from agents import MarketStatistician as MS
import agentpy as ap
import numpy as np
import math

class ArtificialStockMarket(ap.Model):
    """def __init__(self):
        pass"""

    def setup(self: ap.Model):
        self.agents = ap.AgentList(self, self.p.N, MS)
        self.dividend = self.p.averageDividend

    def step(self: ap.Model):
        #print(self.agents.U(10))
        self.dividend = self.dividend_process(self.dividend)
        self.document()

    def document(self: ap.Model):
        #self.record(['dividend'])
        self.record(self.vars)


    def dividend_process(self: ap.Model, prevDividend: float) -> float:
        '''Returns dividend based on AR(1) process given average dividend, autoregressive parameter, and error variance.'''
        errorTerm = np.random.normal(0, math.pow(self.p.errorVar, 2))
        return self.p.averageDividend + self.p.autoregressiveParam * (prevDividend - self.p.averageDividend) + errorTerm
