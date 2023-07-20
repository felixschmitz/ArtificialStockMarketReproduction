from model_params import parameters
from model import ArtificialStockMarket as ASM
from agents import MarketStatistician as MS
from viz_helper import lineplot, errLineplot

import agentpy as ap
import matplotlib.pyplot as plt
import time


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
    steps = int(parameters.get("steps"))
    experimentResults = runningExperiment()

    r = str(input("Saving experiment results to file (T/F): "))
    if "t" in r.lower():
        experimentResults.save(
            exp_name=f"ASM_{steps}",
            exp_id=time.strftime("%d%m%Y-%H%M%S"),
            path="results",
            display=True,
        )

    # modelResults = runningModel()

    """# data = modelResults["variables"]["MarketStatistician"][["position"]]
    vars = (
        ["price", "hreePrice"]
        if parameters.get("mode") == 1 or parameters.get("mode") == 2
        else ["pd", "avgForecast"]
    )
    # ["avgForecast", "hreeForecast"]
    data = modelResults["variables"]["ArtificialStockMarket"][vars]

    fig = lineplot(data)"""

    """r = str(input("Saving model results to file (T/F): "))
    # r = "f"
    if "t" in r.lower():
        modelResults.save(
            exp_name=f"ASM_{steps}",
            exp_id=time.strftime("%d%m%Y-%H%M%S"),
            path="results",
            display=True,
        )"""

    """data = modelResults["variables"]["ArtificialStockMarket"][
        ["pd", "varPriceDividend"]
    ]
    fig = errLineplot(data=data, y="pd", err="varPriceDividend")"""
    plt.show()
