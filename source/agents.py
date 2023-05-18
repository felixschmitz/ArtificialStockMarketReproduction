import agentpy as ap
import numpy as np


class MarketStatistician(ap.Agent):
    def setup(self):
        """setup function initializing and declaring class specific variables"""
        self.rules = self.createRules()
        self.currCash = self.model.p.initialCash
        self.prevCash = self.model.p.initialCash
        self.stocksOwned = 1
        self.optimalStockOwned = 1
        self.demand = self.optimalStockOwned - self.stocksOwned
        self.wealth = self.currCash
        self.expectedWealth = self.budgetConstraint()
        self.utility = self.utilityFunction()

        """if self.p.mode == 1:
            self.slope, self.intercept, self.pdVariance = (
                self.model.hreeSlope,
                self.model.hreeIntercept,
                self.model.hreeVariance,
            )"""

    def createRules(self: ap.Agent) -> dict:
        """creating dict of predictive bitstring rules with their respective predictors and observatory meassures"""
        d = {}
        constantConditions = {11: 1, 12: 0}
        for i in range(1, self.model.p.M + 1):
            variableConditions = {
                i: (1 if j < 10 else 0 if 10 <= j < 20 else None)
                for i, j in enumerate(
                    self.model.nprandom.integers(0, 100, 10).tolist(), 1
                )
            }

            d[i] = {
                "condition": variableConditions | constantConditions,
                "activationIndicator": 0,
                "activationCount": 0,
                "a": self.model.nprandom.uniform(0.7, 1.2),
                "b": self.model.nprandom.uniform(-10, 19.002),
                "fitness": self.p.M,
                "accuracy": 4.0,
            }
        return d

    def step(self: ap.Agent):
        """agent centered timeline followed at each timestep"""
        self.wealth = self.wealthCalc()
        self.demand = self.optimalStockAmount() - self.stocksOwned
        self.update()
        self.document()

    def document(self: ap.Agent):
        """documenting relevant variables of agents"""
        self.record(["demand", "wealth", "utility"])

    def update(self: ap.Agent):
        """updating central variables of agents"""
        self.utility = self.utilityFunction()
        self.wealth = self.currCash * self.model.p.interestRate + self.wealthCalc()

    def wealthCalc(self: ap.Agent) -> float:
        """returning current wealth level"""
        return self.prevCash - self.model.price * self.stocksOwned

    def utilityFunction(self: ap.Agent) -> float:
        """returning the CARA utility of expected wealth."""
        return -np.exp(-self.p.dorra * self.budgetConstraint())

    def budgetConstraint(self: ap.Agent) -> float:
        """returning the expected wealth"""
        return self.optimalStockAmount() * (self.expectationFormation()) + (
            1 + self.model.p.interestRate
        ) * (self.wealth - self.model.price * self.optimalStockAmount())

    def expectationFormation(self: ap.Agent) -> float:
        """returning combined expected price plus dividend based on activated rule"""
        return self.rules.get(1).get("a") * (
            self.model.price + self.model.dividend
        ) + self.rules.get(1).get("b")

    def optimalStockAmount(self: ap.Agent) -> float:
        """returning the current optimal amount of stocks to be held"""
        return (
            self.expectationFormation()
            - self.model.price * (1 + self.model.p.interestRate)
        ) / (self.p.dorra * self.model.varPriceDividend)

    def priceDerivative(self: ap.Agent) -> float:
        """returning the partial derivative of the optimal demand with respect to the price"""
        return (
            self.a * (1 + self.model.dividend) + self.b - 1 - self.model.p.interestRate
        ) / (self.p.dorra * self.model.varPriceDividend)

    """def price_predictionHREE(self: ap.Agent) -> float:
        if self.p.mode == 1:
            price = self.model.market_clearing_price(
                self.model.hreeSlope, self.model.hreeIntercept, self.model.hreeVariance
            )
            return (
                self.slope * (price + self.model.dividend) + self.intercept,
                self.pdvariance,
            )"""
