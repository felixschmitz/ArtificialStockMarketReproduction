from model_params import parameters
from model import MyModel
from agents import MyAgent
import agentpy as ap

def running_model(parameters: dict, model):
    exp_params = parameters
    exp_params.update({'forecast_adaptation': ap.Values(0, 1)})
    exp_sample = ap.Sample(exp_params, randomize=False)
    exp = ap.Experiment(MyModel, exp_sample, iterations=1, record=True, randomize=False)
    exp_results = exp.run(n_jobs=-1)
    return exp_results
