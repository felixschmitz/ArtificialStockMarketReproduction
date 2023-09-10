from agents import MarketStatistician as MS

import agentpy as ap
import numpy as np
import math

np.seterr("raise")


class ArtificialStockMarket(ap.Model):
    def setup(self: ap.Model):
        """setup function initializing and declaring class specific variables"""
        if self.p.mode > 1:
            self.importedDataDict = self.readDataDict(dataDictPath=self.p.importPath)
        self.hreeRandom = self.randomGenerator()
        self.dividendRandom = self.randomGenerator()
        self.theta = 1 / 75 if self.p.forecastAdaptation else 1 / 150
        self.dividend = self.p.averageDividend
        self.price = 80
        self.hreePrice = self.hreePriceCalc()
        self.f, self.g = self.p.hreeA, self.p.hreeB
        self.document()
        self.worldState = self.worldInformation()
        self.agents = ap.AgentList(self, self.p.N, MS)

    def randomGenerator(self: ap.Model) -> np.random.Generator:
        """returning a random generator"""
        seed = self.model.random.getrandbits(self.p.seed)
        return np.random.default_rng(seed=seed)

    def step(self: ap.Model):
        """model centered timeline followed at each timestep"""
        self.dividend = self.dividend_process()
        self.worldState = self.worldInformation()
        self.agents.step()  # activating world state matching predictors
        self.specialistPriceCalc()
        self.agents.update()
        self.agents.document()
        self.document()

    def document(self: ap.Model):
        """documenting relevant variables of the model"""
        if self.t > 0:
            self.record("avgForecast", np.average(self.agents.forecast))
            self.record("avgDemand", np.average(self.agents.demand))
            self.record("avgWealth", np.average(self.agents.wealth))
            self.record("avgPosition", np.average(self.agents.position))
            self.record("avgBitsUsed", np.average(self.agents.log.get("bitsUsed")))
            self.record("avgUtility", np.average(self.agents.log.get("utility")))
            self.record("anayliticalMarketClearingPrice", self.marketClearingPrice())
        self.hreePrice = self.hreePriceCalc()
        self.record("hreeForecast", self.hreeForecastCalc())
        self.record(
            [
                "dividend",
                "price",
                "hreePrice",
            ]
        )
        self.record("pd", self.price + self.dividend)

        self.update()
        self.record(["varPriceDividend"])

    def update(self: ap.Model):
        """updating central variables of the model"""
        self.varPriceDividend = np.var(
            np.array(self.log.get("dividend") + np.array(self.log.get("price")))
        )

    def end(self: ap.Model):
        self.agents.end()

    def readDataDict(self: ap.Agent, dataDictPath: str) -> dict:
        """reading data from dict"""
        exp_name, exp_id = dataDictPath.rsplit("_", 1)
        path, exp_name = exp_name.rsplit("/", 1)
        return ap.DataDict.load(
            exp_name=exp_name, exp_id=exp_id, path=path, display=False
        )

    def specialistPriceCalc(self: ap.Model):
        trialsSpecialist = 0
        while trialsSpecialist < self.p.trialsSpecialist:
            trialsSpecialist += 1
            self.agents.specialistSteps()
            sumDemand = sum(self.agents.demand)
            sumSlope = sum(self.agents.slope)
            demandDifference = sumDemand - self.p.N
            if abs(demandDifference) < self.p.epsilon:
                break
            if sumSlope != 0:
                self.price -= demandDifference / sumSlope
            else:
                self.price *= 1 + 0.0005 * demandDifference
            self.price = (
                self.p.minPrice
                if self.price < self.p.minPrice
                else (self.p.maxPrice if self.price > self.p.maxPrice else self.price)
            )

    def dividend_process(self: ap.Model) -> float:
        """returning current dividend based on AR(1) process"""
        errorTerm = self.dividendRandom.normal(0, math.sqrt(self.p.errorVar))
        return (
            self.p.averageDividend
            + self.p.autoregressiveParam * (self.dividend - self.p.averageDividend)
            + errorTerm
        )

    def worldInformation(self: ap.Model) -> dict:
        """returning the fundamental, technical, and constant state of the world"""
        fundamentalValues = [0.25, 0.5, 0.75, 0.875, 1.0, 1.25]
        technicalValues = [self.technicalMA(periods) for periods in [5, 10, 100, 500]]
        constantConditions = {11: 1, 12: 0}

        fundamentalConditions = {
            idx: (self.price * self.p.interestRate / self.dividend) > fundamental
            for idx, fundamental in zip(range(1, 7), fundamentalValues)
        }
        technicalConditions = {
            idx: self.price > MA for idx, MA in zip(range(7, 11), technicalValues)
        }

        return fundamentalConditions | technicalConditions | constantConditions

    def technicalMA(self: ap.Model, periodMA) -> float:
        """returning the moving average for a certain period (input)"""
        return np.average(self.log.get("price")[-periodMA:])

    def marketClearingPrice(self: ap.Model):
        """Returns inductive market clearing price. Delete?"""
        return (
            sum(
                [
                    (
                        self.dividend * agent.rules.get(agent.currentRule).get("a")
                        + agent.rules.get(agent.currentRule).get("b")
                    )
                    / agent.rules.get(agent.currentRule).get("accuracy")
                    for agent in self.agents
                ]
            )
            - self.p.N * self.p.dorra
        ) / (
            sum(
                [
                    (
                        (
                            (1 + self.p.interestRate)
                            - agent.rules.get(agent.currentRule).get("a")
                        )
                        / agent.rules.get(agent.currentRule).get("accuracy")
                    )
                    for agent in self.agents
                ]
            )
        )

    def hreePriceCalc(self: ap.Model) -> float:
        """Returns homogeneous rational expectation equilibrium price for current period."""
        """f = self.p.autoregressiveParam / (
            1 + self.p.interestRate - self.p.autoregressiveParam
        )
        g = (
            (1 + f)
            * (
                (1 - self.p.autoregressiveParam) * self.p.averageDividend
                - self.p.dorra * self.p.errorVar
            )
        ) / self.p.interestRate
        return f * self.dividend + g"""
        self.f = self.p.autoregressiveParam / (
            1 + self.p.interestRate - self.p.autoregressiveParam
        )
        self.g = (
            (1 + self.f) * (1 - self.p.autoregressiveParam) * self.p.averageDividend
            - self.p.dorra * (math.pow(1 + self.f, 2)) * self.p.errorVar
        ) / self.p.interestRate
        return self.f * self.dividend + self.g

    def hreeForecastCalc(self: ap.Model) -> float:
        """Returns homogeneous raional expactation equilibrium forecast of next periods price plus dividend."""
        """return (1 + self.p.interestRate) * self.hreePrice + (
            (self.p.dorra * (1 + self.p.interestRate) * self.p.errorVar)
            / (1 + self.p.interestRate - self.p.autoregressiveParam)
        )"""
        return self.p.hreeA * (self.dividend + self.price) + self.p.hreeB
