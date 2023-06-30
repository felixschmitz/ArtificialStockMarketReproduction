from model_params import parameters
from model import ArtificialStockMarket as ASM
from agents import MarketStatistician as MS
from viz_helper import lineplot, errLineplot

import agentpy as ap
import matplotlib.pyplot as plt


def runningExperiment(params: dict = parameters, model: ap.Model = ASM) -> ap.DataDict:
    """running an agentpy experiment with extended params"""
    expParams = params
    expParams.update({"forecast_adaptation": ap.Values(0, 1)})
    expSample = ap.Sample(expParams, randomize=False)
    exp = ap.Experiment(model, expSample, iterations=1, record=True, randomize=False)
    expResults = exp.run(n_jobs=-1)
    return expResults


def runningModel(params: dict = parameters, model: ap.Model = ASM) -> ap.DataDict:
    """running the agentpy model with basic params"""
    m = model(params)
    results = m.run()
    return results


if __name__ == "__main__":
    # experiment_results = runningExperiment()
    # print(experimentResults['info'])
    modelResults = runningModel()

    # data = modelResults["variables"]["MarketStatistician"][["pdExpectation"]]
    data = modelResults["variables"]["ArtificialStockMarket"].loc[
        parameters.get("steps") * 3 / 4 :, ["hreePrice", "price"]
    ]
    fig = lineplot(data)

    """data = modelResults["variables"]["ArtificialStockMarket"][
        ["pd", "varPriceDividend"]
    ]
    fig = errLineplot(data=data, y="pd", err="varPriceDividend")"""
    plt.show()
