parameters={
    'seed':40,
    'mode': 1, # with standard mode [0], diagnostics test hree [1], and diagnostics test adaptation [2]
    # 'dividendSequence': None, # for mode 2; diagnostics test with given dividend sequence
    # 'priceSeries': None, # for mode 2; diagnostics test with given price series
    'N':1, # num of agents & num of assets
    'steps':1000, # num of steps/iterations by the model
    'averageDividend':10, # \bar{d}
    'autoregressiveParam':0.95, # in the paper rho
    'errorVar':0.0743,
    'dorra':0.5, #degree of relative risk aversion
    'interestRate':0.1,
    'C': 0.005, # cost levied for specificity
    'initialPriceDividendVariance':4.0,
    'M': 100, # number of predictors per agent
    'forecastAdaptation': 1, # binary with 0 for slow and 1 for fast
    # 'initial_cash':20000, # initial cash of each agent in the bank
    # 'trialsSpecialist':10, # trials per timestep for market clearing
}