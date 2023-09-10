from model_params import parameters
from model import ArtificialStockMarket as ASM
from agents import MarketStatistician as MS

import agentpy as ap
import glob
from datetime import datetime
import math


def runningExperiment(params: dict = parameters, model: ap.Model = ASM) -> ap.DataDict:
    """running an agentpy experiment with extended params"""
    expParams = params
    expParams.update({"forecastAdaptation": ap.Values(0, 1)})
    expSample = ap.Sample(expParams, randomize=False)
    exp = ap.Experiment(model, expSample, iterations=1, record=True, randomize=False)
    expResults = exp.run(n_jobs=-1, verbose=50)
    return expResults


def runningSplitExperiment(
    params: dict = parameters, model: ap.Model = ASM
) -> ap.DataDict:
    """running an agentpy experiment with extended params in batches"""
    stepsize = math.ceil(params.get("steps") / params.get("batches"))
    expParams = params
    expParams.update({"forecastAdaptation": ap.Values(0, 1)})
    batches = math.ceil(params.get("steps") / stepsize)
    for batch in range(batches):
        print(f"Batch {batch+1} of {batches}")
        if batch != 0:
            # all batches except the first one innate the rules from the previous experiment
            importPath = getLatestExperimentPath()
            expParams.update({"importPath": importPath})
        expParams.update(
            {
                "steps": stepsize,
                "seed": expParams.get("seed") + batch,
                "mode": params.get("mode") if batch == 0 else 3,
            }
        )
        expSample = ap.Sample(expParams, randomize=False)
        exp = ap.Experiment(
            model, expSample, iterations=1, record=True, randomize=False
        )
        expResults = exp.run(n_jobs=-1)
        if batch != batches - 1:
            # all batches except the last one save the results to file with their batch size step size
            expResults.save(
                exp_name=f"ASM_{stepsize}",
                exp_id=datetime.now().strftime("%d%m%Y-%H%M%S"),
                path="results",
                display=True,
            )
        else:
            # the last batch results get returned
            return expResults


def getLatestExperimentPath() -> str:
    """returning the latest experiment path"""
    list_of_dirs = glob.glob("results/*/")
    list_of_dirs = [dir for dir in list_of_dirs if "ASM" in dir]
    list_of_timedeltas = [0] * (len(list_of_dirs))
    t = datetime.now()
    for idx, dir in enumerate(list_of_dirs):
        list_of_timedeltas[idx] = abs(
            t - datetime.strptime(dir.rsplit("_", 1)[1], "%d%m%Y-%H%M%S/")
        )
    return list_of_dirs[list_of_timedeltas.index(min(list_of_timedeltas))]


def runningModel(params: dict = parameters, model: ap.Model = ASM) -> ap.DataDict:
    """running the agentpy model with basic params"""
    m = model(params)
    results = m.run()
    return results


if __name__ == "__main__":
    steps = int(parameters.get("steps"))
    if parameters.get("experimentSplit"):
        experimentResults = runningSplitExperiment()
    else:
        experimentResults = runningExperiment()
    r = str(input("Saving experiment results to file (T/F): "))
    if "t" in r.lower():
        experimentResults.save(
            exp_name=f"ASM_{steps}",
            exp_id=datetime.now().strftime("%d%m%Y-%H%M%S"),
            path="results",
            display=True,
        )

    """
    modelResults = runningModel()
    r = str(input("Saving model results to file (T/F): "))
    if "t" in r.lower():
        modelResults.save(
            exp_name=f"ASM_{steps}",
            exp_id=datetime.now().strftime("%d%m%Y-%H%M%S"),
            path="results",
            display=True,
        )
    """
