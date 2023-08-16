import agentpy as ap
import numpy as np
from operator import countOf
import math
from ast import literal_eval


class MarketStatistician(ap.Agent):
    def setup(self):
        """setup function initializing and declaring class specific variables"""
        self.cash = self.model.p.initialCash
        self.rules = self.initializeRules(numRules=self.model.p.M)
        self.currentRule, self.activeRules = self.activateRules()
        self.forecast = self.expectationFormation()
        self.position = 1
        self.wealth = self.cash + self.position * self.model.price
        self.demand = 1
        self.slope = 0
        self.utility = self.utilityCalc()

    def step(self: ap.Agent):
        """agent centered timeline followed at each timestep"""
        self.prevActiveRules = self.activeRules.copy()
        self.currentRule, self.activeRules = self.activateRules()

    def specialistSteps(self: ap.Agent):
        self.forecast = self.expectationFormation()
        self.demand, self.slope = self.demandAndSlopeCalc()

    def update(self: ap.Agent):
        """updating central variables of agents"""
        self.errorVarianceUpdate()
        self.prevForecastUpdate()
        self.gARandom = (
            self.model.randomGenerator()
        )  # should the random generators be initialized during setup?
        gARandInt = self.gARandom.integers(1000)
        gACondition = ((self.model.p.forecastAdaptation) & (gARandInt < 4)) | (
            (not self.model.p.forecastAdaptation) & (gARandInt == 0)
        )
        if gACondition:
            self.geneticAlgorithm()

        self.utility = self.utilityCalc()

        # cash calculation with taxation based on Ehrentreich (2008) to prevent wealth explosion
        self.cash -= (self.demand - self.position) * self.model.price
        self.position = self.demand
        self.cash = self.cash + self.position * (
            self.model.dividend - self.model.p.interestRate * self.model.price
        )
        self.wealth = self.cash + self.position * self.model.price

    def document(self: ap.Agent):
        """documenting relevant variables of agents"""
        self.record(
            "bitsUsed",
            sum(
                [
                    4
                    - [
                        self.rules[rule]["condition"].get(i) for i in range(7, 11)
                    ].count(None)
                    for rule in self.rules
                ]
            ),
        )
        self.record(["forecast", "demand", "cash", "wealth", "position", "utility"])

    def end(self: ap.Agent):
        self.record(["rules"])

    def initializeRules(self: ap.Agent, numRules: int) -> dict:
        """initializing dict of rules with respective predictive bitstring rules"""
        if self.model.p.mode == 3 and self.model.t == 0:
            if self.model._run_id[0] != None:
                # importing rules from pre-trained model for each type of forecastAdaptation (experiment)
                d = literal_eval(
                    self.model.importedDataDict["variables"]["MarketStatistician"]
                    .loc[self.model._run_id[0]]
                    .loc[self.id]
                    .iloc[-1]
                    .rules
                )
                pt, dt = (
                    self.model.importedDataDict["variables"]["ArtificialStockMarket"]
                    .loc[self.model._run_id[0]]
                    .iloc[-1][["price", "dividend"]]
                )
                self.addPrevForecast(d, pt, dt)
            else:
                # importing rules from pre-trained model (single model)
                d = literal_eval(
                    self.model.importedDataDict["variables"]["MarketStatistician"]
                    .loc[self.id]
                    .iloc[-1]
                    .rules
                )
        else:
            d = {}
            for i in range(1, numRules + 1):
                d[i] = self.createRule()
        return d

    def addPrevForecast(self: ap.Agent, d: dict, pt: float, dt: float):
        for predictor in d:
            d[predictor]["prevForecast"] = (
                d[predictor]["a"] * (pt + dt) + d[predictor]["b"]
            )

    def createRule(self: ap.Agent) -> dict:
        """creating dict of predictive bitstring rule with respective predictor and observatory meassures"""
        self.ruleRandom = self.model.randomGenerator()
        constantConditions = {11: 1, 12: 0}
        variableConditions = {
            i: (1 if j < 10 else 0 if 10 <= j < 20 else None)
            for i, j in enumerate(self.ruleRandom.integers(0, 100, 10).tolist(), 1)
        }
        return {
            "condition": variableConditions | constantConditions,
            "activationIndicator": 0,
            "activationCount": 0,
            "a": self.ruleRandom.uniform(0.7, 1.2)
            if self.model.p.mode != 1
            else self.model.p.hreeA,
            "b": self.ruleRandom.uniform(-10, 19.002)
            if self.model.p.mode != 1
            else self.model.p.hreeB,
            "fitness": self.model.p.M,
            "accuracy": self.model.p.initialPredictorVariance,
            "errorVariance": self.model.p.initialPredictorVariance,
            "prevForecast": 110,
        }

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
                        self.rules[currentRuleKey]["errorVariance"]
                        > self.rules[ruleID]["errorVariance"]
                    ):
                        currentRuleKey = ruleID

        if currentRuleKey == 0:
            activeRuleKeys = [0]
            weights = [rule["fitness"] for rule in self.rules.values()]
            self.rules[0] = {
                "condition": {
                    i + 1: (None if i < 10 else True if i == 10 else False)
                    for i in range(12)
                },
                "activationIndicator": 1,
                "activationCount": 0,
                "a": np.average(
                    [rule["a"] for rule in self.rules.values()], weights=weights
                ),
                "b": np.average(
                    [rule["b"] for rule in self.rules.values()], weights=weights
                ),
                "fitness": np.average(
                    [rule["fitness"] for rule in self.rules.values()], weights=weights
                ),  # self.model.p.M,
                "accuracy": np.average(
                    [rule["accuracy"] for rule in self.rules.values()], weights=weights
                ),  # self.model.p.initialPredictorVariance,
                "errorVariance": np.average(
                    [rule["errorVariance"] for rule in self.rules.values()],
                    weights=weights,
                ),  # self.model.p.initialPredictorVariance,
            }
        return currentRuleKey, activeRuleKeys

    def utilityCalc(self: ap.Agent) -> float:
        """calculating the utility of the agent"""
        return -(np.exp(-self.model.p.dorra * (self.demand - self.position)))

    def expectationFormation(self: ap.Agent) -> float:
        """returning combined expected price plus dividend based on activated rule"""
        return self.rules.get(self.currentRule).get("a") * (
            self.model.price + self.model.dividend
        ) + self.rules.get(self.currentRule).get("b")

    def prevExpectationFormation(self: ap.Agent, ruleID: int) -> float:
        """returning combined expected price plus dividend based on activated rule"""
        return self.rules.get(ruleID).get("a") * (
            self.model.price + self.model.dividend
        ) + self.rules.get(ruleID).get("b")

    def demandAndSlopeCalc(self: ap.Agent) -> float:
        """getting the demand and slope from the demand function"""
        demand = (
            self.forecast - self.model.price * (1 + self.model.p.interestRate)
        ) / (self.model.p.dorra * self.rules.get(self.currentRule).get("accuracy"))
        slope = self.rules.get(self.currentRule).get("a") - (
            1 + self.model.p.interestRate
        ) / (self.model.p.dorra * self.rules.get(self.currentRule).get("accuracy"))
        demand = self.constrainDemand(demand)
        return demand, slope

    def constrainDemand(self: ap.Agent, demand: float) -> float:
        """constraining the demand to the maximum bid and the minimum holding"""
        if (demand > 0) & ((demand * self.model.price) > (self.cash)):
            if self.cash > 0:
                demand = (self.cash) / self.model.price
            else:
                demand = 0
        elif (demand < 0) & (demand + self.position < self.model.p.minHolding):
            demand = self.model.p.minHolding
        return demand

    def errorVarianceUpdate(self: ap.Agent) -> None:
        """updating the errorVariance of the predictors"""
        for ruleID in self.prevActiveRules:
            if ruleID != 0:
                self.rules[ruleID]["errorVariance"] = (
                    1 - self.model.theta
                ) * self.rules[ruleID]["errorVariance"] + self.model.theta * math.pow(
                    self.model.price
                    + self.model.dividend
                    - self.rules[ruleID]["prevForecast"],
                    2,
                )

    def prevForecastUpdate(self: ap.Agent) -> None:
        """updating the previous forecast of the predictors"""
        for ruleID in self.activeRules:
            if ruleID != 0:
                self.rules[ruleID]["prevForecast"] = self.prevExpectationFormation(
                    ruleID=ruleID
                )

    def crossoverRule(
        self: ap.Agent, ruleID: int, parentRuleID1: int, parentRuleID2: int
    ) -> dict:
        """crossover of two parent rules"""
        self.crossoverRandom = self.model.randomGenerator()
        # crossover on bitstring level
        constantConditions = {11: 1, 12: 0}
        variableConditions = {
            idx + 1: (self.rules[parentID]["condition"][idx + 1])
            for idx, parentID in enumerate(
                self.crossoverRandom.choice(
                    [parentRuleID1, parentRuleID2], 10
                ).tolist(),
            )
        }
        # setting new bitstring after uniform crossover
        self.rules[ruleID]["condition"] = variableConditions | constantConditions

        # crossover on predictive vector level
        # 0: crossover component-wise,
        # 1: linear combination,
        # 2: complete selection of one predictive vector
        crossoverProcedure = self.crossoverRandom.integers(3)
        if crossoverProcedure == 0:
            self.rules[ruleID]["a"], self.rules[ruleID]["b"] = [
                self.rules[parentID][("a" if not idx else "b")]
                for idx, parentID in enumerate(
                    self.crossoverRandom.choice(
                        [parentRuleID1, parentRuleID2], 2
                    ).tolist()
                )
            ]

        elif crossoverProcedure == 1:
            # linear combination of parent predictive vectors
            (p1A, p1B), (p2A, p2B) = [
                (a, 1 - a) for a in self.crossoverRandom.uniform(0, 1, 2)
            ]
            self.rules[ruleID]["a"], self.rules[ruleID]["b"] = (
                self.rules[parentRuleID1]["a"] * p1A
                + self.rules[parentRuleID2]["a"] * p2A,
                self.rules[parentRuleID1]["b"] * p1B
                + self.rules[parentRuleID2]["b"] * p2B,
            )

        elif crossoverProcedure == 2:
            # complete selection of one of the parents' predictive vector
            parentID = self.crossoverRandom.choice([parentRuleID1, parentRuleID2])
            self.rules[ruleID]["a"], self.rules[ruleID]["b"] = (
                self.rules[parentID]["a"],
                self.rules[parentID]["b"],
            )

    def mutatingBit(
        self: ap.Agent, predictiveBit: bool, bit: bool | None | float
    ) -> bool | None | float:
        """mutating a single bit of a predictive vector or bitstring"""
        self.mutationRandom = self.model.randomGenerator()
        if predictiveBit:
            # mutating a predictive vector
            # is this ok or do I need to make sure they stay in the specific ranges?
            return bit * self.mutationRandom.uniform(0.95, 1.05)
        else:
            # mutating a bitstring
            s = {True, False, None}
            s.remove(bit)
            return self.mutationRandom.choice(list(s))

    def geneticAlgorithmPreparation(self: ap.Agent, ruleID: str):
        """preparing the rules for the genetic algorithm by updating accuracy and fitness"""
        self.rules[ruleID]["accuracy"] = self.rules[ruleID]["errorVariance"]
        s = sum(value != None for value in self.rules[ruleID]["condition"].values())
        self.rules[ruleID]["fitness"] = (
            # self.model.p.M - self.rules[ruleID]["accuracy"] - (self.model.p.C * s)
            1e9
            # - self.rules[ruleID]["accuracy"]
            - self.rules[ruleID]["errorVariance"]
            - (self.model.p.C * s)
        )

    def geneticAlgorithm(self: ap.Agent):
        """performing the genetic algorithm"""
        # print(f"\nactivated at timestep: {self.model.t} for agent: {self.id}")
        rulesToBeReplaced = set(
            list(
                dict(
                    sorted(
                        self.rules.items(), key=lambda item: item[1]["errorVariance"]
                    )
                ).keys()
            )[-20:]
        )

        parentRules = set(self.rules) - {0} - rulesToBeReplaced
        # fitness values and normalized fitness values (used as probabilities) of the parent rules
        fitnessValues = [self.rules[i]["fitness"] for i in list(parentRules)]
        normalizedFitnessValues = [float(i) / sum(fitnessValues) for i in fitnessValues]
        for ruleID in self.rules.keys():
            if ruleID != 0:
                self.geneticAlgorithmPreparation(ruleID=ruleID)
            if ruleID in rulesToBeReplaced:
                # initializing new rule
                self.rules[ruleID] = self.createRule()

                crossoverRandInt = self.gARandom.integers(100)
                crossoverCondition = (
                    (self.model.p.forecastAdaptation) & (crossoverRandInt < 30)
                ) | (
                    (not self.model.p.forecastAdaptation) & (crossoverRandInt < 10)
                )  # see page 13?
                # crossover takes place with probability 0.3 in the fast adaptation case
                # and 0.1 in the slow adaptation case
                if crossoverCondition:
                    # parents for crossover are chosen based on fitness
                    p1, p2 = self.gARandom.choice(
                        list(parentRules),
                        2,
                        p=normalizedFitnessValues,
                    )
                    # performing crossover
                    self.crossoverRule(
                        ruleID=ruleID, parentRuleID1=p1, parentRuleID2=p2
                    )

                # mutation takes place with probability 0.03 per bit
                for predictor in ["a", "b"]:
                    if self.gARandom.random() < 0.03:
                        self.rules[ruleID][predictor] = self.mutatingBit(
                            predictiveBit=True, bit=self.rules[ruleID][predictor]
                        )
                for key in range(1, 11):
                    if self.gARandom.random() < 0.03:
                        self.rules[ruleID]["condition"][key] = self.mutatingBit(
                            predictiveBit=False,
                            bit=self.rules[ruleID]["condition"][key],
                        )
