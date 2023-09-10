from agents import MarketStatistician as MS

import agentpy as ap
import numpy as np
import math

np.seterr("raise")


class ArtificialStockMarket(ap.Model):
    def setup(self: ap.Model):
        """setup function initializing and declaring class specific variables"""
        if self.p.mode == 3:
            # importing data from previous experiment based on specified path
            self.importedDataDict = self.readDataDict(dataDictPath=self.p.importPath)
        self.hreeRandom = (
            self.randomGenerator()
        )  # initializing random generator for h.r.e.e. processes
        self.dividendRandom = (
            self.randomGenerator()
        )  # initializing random generator for dividend process
        self.theta = (
            1 / 75 if self.p.forecastAdaptation else 1 / 150
        )  # specifying theta for given regimes based on forecastAdaptation
        self.dividend = self.p.averageDividend  # calculating initial dividend value
        self.price = 80  # specifying initial price value
        self.hreePrice = (
            self.hreePriceCalc()
        )  # calculating initial h.r.e.e. price value
        self.document()
        self.worldState = (
            self.worldInformation()
        )  # observing state of the world based on fundamental, technical, and constant conditions
        self.agents = ap.AgentList(self, self.p.N, MS)  # initializing agents

    def randomGenerator(self: ap.Model) -> np.random.Generator:
        """returning a random generator"""
        seed = self.model.random.getrandbits(self.p.seed)
        return np.random.default_rng(seed=seed)

    def step(self: ap.Model):
        """model centered timeline followed at each timestep"""
        self.dividend = (
            self.dividend_process()
        )  # calculating dividend for current period
        self.worldState = (
            self.worldInformation()
        )  # observing state of the world based on fundamental, technical, and constant conditions
        self.agents.step()  # activating world state matching predictors of agents
        self.specialistPriceCalc()  # iterative specialist price determination for current period
        self.agents.update()  # updating agent specific variables
        self.agents.document()  # documenting agent specific variables
        self.document()  # documenting model specific variables

    def document(self: ap.Model):
        """documenting relevant variables of the model"""
        if self.t > 0:
            # documenting variables for all timesteps except the first one
            self.record("avgForecast", np.average(self.agents.forecast))
            self.record("avgDemand", np.average(self.agents.demand))
            self.record("avgWealth", np.average(self.agents.wealth))
            self.record("avgPosition", np.average(self.agents.position))
            self.record(
                "avgBitsUsed",
                np.average(
                    [sublist[-1] for sublist in self.agents.log.get("bitsUsed")]
                ),
            )
            self.record(
                "sumBitsUsed",
                sum(sublist[-1] for sublist in self.agents.log.get("bitsUsed")),
            )
            if self.t > 1:
                self.record(
                    "aggregatedVolume",
                    sum(
                        abs(sublist[-2] - sublist[-1])
                        for sublist in self.agents.log.get("position")
                    ),
                )
        self.hreePrice = (
            self.hreePriceCalc()
        )  # calculating h.r.e.e. price for current period
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
        # updating variance of price plus dividend up until current period
        self.varPriceDividend = np.var(
            np.array(self.log.get("dividend") + np.array(self.log.get("price")))
        )

    def end(self: ap.Model):
        """end function for model"""
        self.agents.end()

    def readDataDict(self: ap.Agent, dataDictPath: str) -> dict:
        """reading data from dict of previous experiment"""
        exp_name, exp_id = dataDictPath.rsplit("_", 1)
        path, exp_name = exp_name.rsplit("/", 1)
        return ap.DataDict.load(
            exp_name=exp_name, exp_id=exp_id, path=path, display=False
        )

    def specialistPriceCalc(self: ap.Model):
        trialsSpecialist = 0
        while trialsSpecialist < self.p.trialsSpecialist:
            trialsSpecialist += 1
            self.agents.specialistSteps()  # activating specialist steps of agents
            sumDemand = sum(
                self.agents.demand
            )  # calculating sum of demand of all agents
            sumSlope = sum(self.agents.slope)  # calculating sum of slope of all agents
            demandDifference = (
                sumDemand - self.p.N
            )  # calculating difference between sum of demand and number of shares of the asset
            if abs(demandDifference) < self.p.epsilon:
                # if difference between sum of demand and number of shares of the asset is smaller than epsilon, break
                break
            if sumSlope != 0:
                # if sum of slope is not zero, update price based on demandDifference and sum of slope
                self.price -= demandDifference / sumSlope
            else:
                # if sum of slope is zero, update price based on the demandDifference and a factor
                self.price *= 1 + 0.0005 * demandDifference
            # if price is smaller than minPrice, set price to minPrice, if price is larger than maxPrice, set price to maxPrice, else keep price
            self.price = (
                self.p.minPrice
                if self.price < self.p.minPrice
                else (self.p.maxPrice if self.price > self.p.maxPrice else self.price)
            )

    def dividend_process(self: ap.Model) -> float:
        """returning current dividend based on AR(1) process"""
        errorTerm = self.dividendRandom.normal(0, math.sqrt(self.p.errorVar))
        # calculating the dividend based on AR(1) process specified in the paper
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

    def hreePriceCalc(self: ap.Model) -> float:
        """Returns h.r.e.e. price for current period. Formula is based on Ehrentreich2008 p.102f."""
        self.f = self.p.autoregressiveParam / (
            1 + self.p.interestRate - self.p.autoregressiveParam
        )
        self.g = (
            (1 + self.f) * (1 - self.p.autoregressiveParam) * self.p.averageDividend
            - self.p.dorra * (math.pow(1 + self.f, 2)) * self.p.errorVar
        ) / self.p.interestRate
        return self.f * self.dividend + self.g

    def hreeForecastCalc(self: ap.Model) -> float:
        """Returns h.r.e.e. forecast of next periods price plus dividend."""
        return self.p.hreeA * (self.dividend + self.hreePrice) + self.p.hreeB
