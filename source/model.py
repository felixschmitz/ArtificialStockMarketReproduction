from agents import MarketStatistician as MS

import agentpy as ap
import numpy as np
import math

np.seterr("raise")


class ArtificialStockMarket(ap.Model):
    def setup(self: ap.Model):
        """setup function initializing and declaring class specific variables"""
        self.theta = 1 / 75 if self.p.forecastAdaptation else 1 / 150
        self.dividend = self.p.averageDividend
        self.price = 100
        self.hreeSlope, self.hreeIntercept = self.hreeValues()
        self.hreePrice = self.hreePriceCalc()
        self.document()
        self.worldState = self.worldInformation()
        self.agents = ap.AgentList(self, self.p.N, MS)

    def step(self: ap.Model):
        """model centered timeline followed at each timestep"""
        self.dividend = self.dividend_process()
        self.worldState = self.worldInformation()
        self.agents.step()
        self.price = self.marketClearingPrice()
        self.hreePrice = self.hreePriceCalc()
        self.agents.update()
        self.agents.document()
        self.document()

    def document(self: ap.Model):
        """documenting relevant variables of the model"""
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

    def dividend_process(self: ap.Model) -> float:
        """returning current dividend based on AR(1) process"""
        errorTerm = self.nprandom.normal(0, math.sqrt(self.p.errorVar))
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
        """Returns inductive market clearing price."""
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

    def hreeValues(
        self: ap.Model,
        a_min: float = 0.7,
        a_max: float = 1.2,
        b_min: float = -10,
        b_max: float = 19.002,
    ):
        """Returns homogeneous rational expectations equilibrium predictor values."""
        return (
            self.nprandom.uniform(a_min, a_max),
            self.nprandom.uniform(b_min, b_max),
            # self.p.initialPredictorVariance,
        )

    def hreePriceCalc(self: ap.Model) -> float:
        """Returns homogeneous rational expectation equilibrium price."""
        f = self.p.autoregressiveParam / (
            1 + self.p.interestRate - self.p.autoregressiveParam
        )
        g = (
            (1 + f)
            * (
                (1 - self.p.autoregressiveParam) * self.p.averageDividend
                - self.p.dorra * self.p.errorVar
            )
        ) / self.p.interestRate
        return f * self.dividend + g

    def hreeForecastCalc(self: ap.Model) -> float:
        """Returns homogeneous raional expactation equilibrium of next periods price"""
        return (1 + self.p.interestRate) * self.price + (
            (self.p.dorra * (2 + self.interestRate) * self.p.errorVar)
            / (1 + self.p.interestRate - self.p.autoregressiveParam)
        )
