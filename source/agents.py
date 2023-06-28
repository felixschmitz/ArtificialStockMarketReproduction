import agentpy as ap
import numpy as np


class MarketStatistician(ap.Agent):
    def setup(self):
        """setup function initializing and declaring class specific variables"""
        self.cash = self.model.p.initialCash
        self.rules = self.initializeRules(numRules=self.model.p.M)
        self.currentRule, self.activeRules = self.activateRules()
        self.forecast = self.expectationFormation()
        self.position = 1
        self.optimalStockOwned = 1
        self.demand = self.optimalStockOwned - self.position
        self.slope = 0

    def step(self: ap.Agent):
        """agent centered timeline followed at each timestep"""
        self.currentRule, self.activeRules = self.activateRules()
        self.forecast = self.expectationFormation()
        self.settingDemandAndSlope()
        self.constrainDemand()

    def update(self: ap.Agent):
        """updating central variables of agents"""
        self.errorVarianceUpdate()
        # self.utility = self.utilityFunction()

        gArandint = self.model.nprandom.integers(1000)
        gACondition = ((self.model.p.forecastAdaptation) & (gArandint < 4)) | (
            (not self.model.p.forecastAdaptation) & (gArandint == 0)
        )
        # fast: forecastAdaptataion == True, gArandint < 4, theta = 1/75, p(crossover) = 0.1
        # slow: forecastAdaptation == False, gArandint == 0, theta = 1/150, p(crossover) = 0.3
        if gACondition:
            self.geneticAlgorithm()

        # self.wealth = self.wealthCalc()
        """self.currCash = (
            self.currCash * (1 + self.model.p.interestRate)
            + self.model.dividend * self.position
        )"""

    def document(self: ap.Agent):
        """documenting relevant variables of agents"""
        # self.record(["demand", "wealth", "utility"])
        self.record(["demand"])
        self.record("pdExpectation", self.expectationFormation())

    def createRule(self: ap.Agent, geneticAlgo: bool = False) -> dict:
        """creating dict of predictive bitstring rule with respective predictor and observatory meassures"""
        constantConditions = {11: 1, 12: 0}
        if geneticAlgo:
            pass
        else:
            variableConditions = {
                i: (1 if j < 10 else 0 if 10 <= j < 20 else None)
                for i, j in enumerate(
                    self.model.nprandom.integers(0, 100, 10).tolist(), 1
                )
            }
            return {
                "condition": variableConditions | constantConditions,
                "activationIndicator": 0,
                "activationCount": 0,
                "a": self.model.nprandom.uniform(0.7, 1.2),
                "b": self.model.nprandom.uniform(-10, 19.002),
                "fitness": self.model.p.M,
                "accuracy": self.model.p.initialPredictorVariance,
                "errorVariance": self.model.p.initialPredictorVariance,
            }

    def initializeRules(self: ap.Agent, numRules: int) -> dict:
        """initializing dict of rules with respective predictive bitstring rules"""
        d = {}
        for i in range(1, numRules + 1):
            d[i] = self.createRule()
        return d

    def activateRules(self: ap.Agent) -> tuple[int, list]:
        """activating the rules matching the models worldState and returning a list reflecting the keys"""
        activeRuleKeys = []
        currentRuleKey = 0
        for ruleID in self.rules:
            for ruleBitID, ruleBit in enumerate(
                self.rules[ruleID]["condition"].values(), 1
            ):
                if ruleBit is None:
                    pass
                elif ruleBit != self.model.worldState[ruleBitID]:
                    break
                if ruleBitID == 11:
                    self.rules[ruleID]["activationIndicator"] = 1
                    self.rules[ruleID]["activationCount"] += 1
                    activeRuleKeys.append(ruleID)
                    if (currentRuleKey == 0) or (
                        self.rules[currentRuleKey]["accuracy"]
                        > self.rules[ruleID]["accuracy"]
                    ):
                        currentRuleKey = ruleID

        if currentRuleKey == 0:
            activeRuleKeys = [0]
            weights = [rule["fitness"] for rule in self.rules.values()]
            self.rules[0] = {
                "a": np.average(
                    [rule["a"] for rule in self.rules.values()], weights=weights
                ),
                "b": np.average(
                    [rule["b"] for rule in self.rules.values()], weights=weights
                ),
                "fitness": self.model.p.M,
                "accuracy": self.model.p.initialPredictorVariance,
            }
        return currentRuleKey, activeRuleKeys

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
