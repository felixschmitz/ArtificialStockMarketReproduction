from agents import MarketStatistician as MS

import agentpy as ap
import numpy as np
import math


class ArtificialStockMarket(ap.Model):
    """
    def __init__(self, parameters=None, _run_id=None, **kwargs):
        super().__init__(parameters, _run_id, **kwargs)
    """

    def setup(self: ap.Model):
        """setup function initializing and declaring class specific variables"""
        self.dividend = self.p.averageDividend
        self.price = 100
        self.varPriceDividend = 4.0
        self.document()
        self.agents = ap.AgentList(self, self.p.N, MS)

        # self.hreeSlope, self.hreeIntercept, self.hreeVariance = self.hree_values()
        # self.hreePrice = self.hree_price()

    def step(self: ap.Model):
        """model centered timeline followed at each timestep"""
        self.dividend = self.dividend_process()
        self.agents.step()

        """
        self.hreePrice = self.hree_price()
        self.marketPrice = self.market_clearing_price()
        """
        self.document()

    def document(self: ap.Model):
        """documenting relevant variables of the model"""
        self.record(
            [
                "dividend",
                "price",
            ]
        )
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

    def specialist(self: ap.Model):
        """If the specialist is not able to find a market clearing price in the first place,
        an iterative process is started in which new trial prices are announced and agents update their
        effective demands and partial derivatives accordingly. If complete market clearing is not reached
        within a specified number of trials, one side of the market will be rationed.
        """
        pass

    """def hree_values(
        self: ap.Model,
        a_min: float = 0.7,
        a_max: float = 1.2,
        b_min: float = -10,
        b_max: float = 19.002,
    ):
        ""Returns homogeneous rational expectations equilibrium predictor values.""
        return (
            self.nprandom.uniform(a_min, a_max),
            self.nprandom.uniform(b_min, b_max),
            self.p.initialPriceDividendVariance,
        )

    def hree_price(self: ap.Model) -> float:
        ""Returns homogeneous rational expectation equilibrium price.""
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

    def market_clearing_price(self: ap.Model) -> float:
        ""Returns inductive market clearing price.""
        numerator = (
            self.dividend * (sum(self.agents.slope) / sum(self.agents.pdVariance))
            + (sum(self.agents.intercept) / sum(self.agents.pdVariance))
            - self.p.N * self.p.dorra
        )
        denominator = (1 + self.p.interestRate) * (1 / sum(self.agents.pdVariance)) - (
            sum(self.agents.slope) / sum(self.agents.pdVariance)
        )
        return numerator / denominator"""
