parameters={
    'seed':40,
    'N':25, # num of agents & num of assets
    'steps':10000, # num of steps/iterations by the model
    'average_dividend':10, # \bar{d}
    'autoregressive_parameter':0.95, # in the paper rho
    'error_variance':0.0743,
    'dorra':0.5, #degree of relative risk aversion
    'interest_rate':0.1,
    'C': 0.005, # cost levied for specificity
    'initial_price_dividend_variance':4.0,
    'M': 100, # number of predictors per agent
    'forecast_adaptation': 1, # binary with 0 for slow and 1 for fast
    'initial_cash':20000, # initial cash of each agent in the bank
    'trials_specialist':10, # trials per timestep for market clearing
}