import agentpy as ap
import numpy as np


class MarketStatistician(ap.Agent):
    def setup(self):
        self.currCash = self.model.p.initialCash
        self.prevCash = self.model.p.initialCash
        self.stocksOwned = 1
        self.optimalStockOwned = 1
        self.demand = self.optimalStockOwned - self.stocksOwned
        self.wealth = self.wealthCalc()
        self.expectedWealth = self.budgetConstraint()
        self.utility = self.U(wealthExpect=self.expectedWealth)
        self.rules = self.createRules()
        """if self.p.mode == 1:
            self.slope, self.intercept, self.pdVariance = (
                self.model.hreeSlope,
                self.model.hreeIntercept,
                self.model.hreeVariance,
            )"""

    def createRules(self: ap.Agent) -> dict:
        d = {}
        constantCondition = {11: 1, 12: 0}
        for i in range(1, self.model.p.M + 1):
            variableCondition = {
                i: (1 if j < 10 else 0 if 10 <= j < 20 else None)
                for i, j in enumerate(
                    self.model.nprandom.integers(0, 100, 10).tolist(), 1
                )
            }

            d[i] = {
                "condition": variableCondition | constantCondition,
                "activationIndicator": 0,
                "activationCount": 0,
                "a": self.model.nprandom.uniform(0.7, 1.2),
                "b": self.model.nprandom.uniform(-10, 19.002),
                "fitness": self.p.M,
                "accuracy": 4.0,
            }
        return d

    def step(self: ap.Agent):
        self.wealth = self.wealthCalc()

    def wealthCalc(self: ap.Agent):
        return self.prevCash - self.model.currentPrice * self.stocksOwned

    def U(self: ap.Agent, wealthExpect: int) -> float:
        """Return the CARA utility of expected Wealth."""
        return -np.exp(-self.p.dorra * wealthExpect)

    def budgetConstraint(self: ap.Agent) -> float:
        return self.stocksOwned * (futurePrice + futureDividend) + (
            1 + self.model.p.interestRate
        ) * (self.wealth - self.model.currentPrice * self.stocksOwned)

    def expectationFormation(self: ap.Agent):
        # To Do
        return  # expected price plus dividend

    def optimalStockAmount(self: ap.Agent):
        return (
            self.expectationFormation()
            - self.model.currentPrice * (1 + self.model.p.interestRate)
        ) / (self.p.dorra * self.model.varPriceDividend)

    def priceDerivative(self: ap.Agent):
        return (
            self.a * (1 + self.model.dividend) + self.b - 1 - self.model.p.interestRate
        ) / (self.p.dorra * self.model.varPriceDividend)

    """def price_prediction(self: ap.Agent) -> float:
        if self.p.mode == 1:
            price = self.model.market_clearing_price(
                self.model.hreeSlope, self.model.hreeIntercept, self.model.hreeVariance
            )
            return (
                self.slope * (price + self.model.dividend) + self.intercept,
                self.pdvariance,
            )"""
