from model_params import parameters
from model import ArtificialStockMarket as ASM
from agents import MyAgent
import agentpy as ap

def running_experiment(params: dict=parameters, model: ap.Model=ASM) -> ap.DataDict:
    exp_params = params
    exp_params.update({'forecast_adaptation': ap.Values(0, 1)})
    exp_sample = ap.Sample(exp_params, randomize=False)
    exp = ap.Experiment(model, exp_sample, iterations=1, record=True, randomize=False)
    exp_results = exp.run(n_jobs=-1)
    return exp_results

def running_model(params: dict=parameters, model: ap.Model=ASM) -> ap.DataDict:
    m = model(params)
    results = m.run()
    return results


if __name__ == '__main__':
    #experiment_results = running_experiment()
    #print(experiment_results['info'])
    model_results = running_model()
    print(model_results['info'])
