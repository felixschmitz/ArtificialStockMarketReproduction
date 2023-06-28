import agentpy as ap
import numpy as np
import scipy
import math


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

    def settingDemandAndSlope(self: ap.Agent):
        """setting the demand and the slope of the demand function"""
        if self.forecast >= 0:
            self.demand = -(
                (
                    (self.model.price * (1 + self.model.dividend) - self.forecast)
                    / (
                        self.model.p.dorra
                        * self.rules.get(self.currentRule).get("accuracy")
                    )
                )
                + self.position
            )
            self.slope = (
                self.rules.get(self.currentRule).get("a") - (1 + self.model.dividend)
            ) / (self.model.p.dorra * self.rules.get(self.currentRule).get("accuracy"))
        else:
            self.forecast = 0
            self.demand = -(self.model.price * (1 + self.model.dividend)) / (
                (self.model.p.dorra * self.rules.get(self.currentRule).get("accuracy"))
                + self.position
            )
            self.slope = -(1 + self.model.dividend) / (
                self.model.p.dorra * self.rules.get(self.currentRule).get("accuracy")
            )

        # restricting the demand within the range of the maximum bid
        if self.demand > self.model.p.maxBid:
            self.demand = self.model.p.maxBid
            self.slope = 0
        elif self.demand < -self.model.p.maxBid:
            self.demand = -self.model.p.maxBid
            self.slope = 0

    def getDemandAndSlope(self: ap.Agent) -> tuple[float, float]:
        """returning the demand and the slope of the demand function"""
        return self.demand, self.slope

    def constrainDemand(self: ap.Agent):
        """constraining the demand to the maximum bid and the minimum holding"""
        if self.demand > 0:
            if self.demand * self.model.price > self.cash - self.model.p.minCash:
                if self.cash - self.model.p.minCash > 0:
                    self.demand = (self.cash - self.model.p.minCash) / self.model.price
                    self.slope = -self.demand / self.model.price
                else:
                    self.demand = 0
                    self.slope = 0

        elif (self.demand < 0) & (
            self.demand + self.position < self.model.p.minHolding
        ):
            self.demand = self.model.p.minHolding - self.position
            self.slope = 0

    def expectationFormation(self: ap.Agent) -> float:
        """returning combined expected price plus dividend based on activated rule"""
        return self.rules.get(self.currentRule).get("a") * (
            self.model.price + self.model.dividend
        ) + self.rules.get(self.currentRule).get("b")

    def errorVarianceUpdate(self: ap.Agent) -> None:
        """updating the errorVariance of the predictors"""
        for ruleID in self.activeRules:
            if ruleID != 0:
                self.rules[ruleID]["errorVariance"] = (
                    1 - self.model.theta
                ) * self.rules[ruleID]["errorVariance"] + self.model.theta * math.pow(
                    self.model.price + self.model.dividend - self.forecast, 2
                )
                self.rules[ruleID]["activationIndicator"] = 0

    def geneticAlgorithmPreparation(self: ap.Agent, ruleID: str):
        """preparing the rules for the genetic algorithm by updating accuracy and fitness"""
        self.rules[ruleID]["accuracy"] = self.rules[ruleID]["errorVariance"]
        s = sum(value == None for value in self.rules[ruleID]["condition"].values())
        self.rules[ruleID]["fitness"] = (
            self.model.p.M - self.rules[ruleID]["accuracy"] - (self.model.p.C * s)
        )

    def geneticAlgorithm(self: ap.Agent):
        """performing the genetic algorithm"""
        print(f"\nactivated at timestep: {self.model.t} for agent: {self.id}")
        rulesToBeReplaced = list(
            dict(
                sorted(self.rules.items(), key=lambda item: item[1]["errorVariance"])
            ).keys()
        )[-20:]
        parentRules = set(self.rules)
        print(
            f"rules to be replaced: {rulesToBeReplaced}, with accuracies: {[self.rules[i]['errorVariance'] for i in rulesToBeReplaced]}"
        )
        for ruleID in self.rules.keys():
            if ruleID != 0:
                self.geneticAlgorithmPreparation(ruleID=ruleID)
        # print([self.rules[i]["accuracy"] for i in self.activeRules])
