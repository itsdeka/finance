#!/usr/bin/env python
# coding: utf-8

from pandas_datareader import data
from pandas.tseries.offsets import CustomBusinessMonthBegin
from pandas.tseries.holiday import USFederalHolidayCalendar
import statsmodels.api as sm
import numpy as np


def get_expected_returns(database):
    expected_returns = []

    for i in range(0, len(database[symbols[0]]['monthly_log_returns'])):
        expected_return = 0
        for symbol in symbols:
            if symbol != '^GSPC':
                expected_return += database[symbol]['monthly_log_returns'].iloc[i] * database[symbol]['allocation']
        expected_returns.append(expected_return)
        print(
            f"{database[symbols[0]]['monthly_log_returns'].index[i]} | Expected return: {round(expected_return * 100, 2)}%")

    return expected_returns

bmth_us = CustomBusinessMonthBegin(calendar=USFederalHolidayCalendar())

start_date = '2015-01-01'
end_date = '2021-04-13'

# +++STEP 1: FORM THE PORTFOLIO

# Load the desired data
symbols = ['^GSPC', 'AMJ', 'QABA', 'DIA', 'AGG']
database = {}

for symbol in symbols:
    database[symbol] = {'data': data.DataReader(symbol, 'yahoo', start_date, end_date)}
    print(f'{symbol} data loaded')

# +++STEP 2: CONVERT DAILY ADJUSTED PRICES TO MONTHLY LOG RETURNS

for symbol in symbols:
    monthly = database[symbol]['data'].resample(bmth_us).mean()
    database[symbol]['monthly_log_returns'] = np.log(monthly['Adj Close'] / monthly['Adj Close'].shift(1))

# STEP 3: CONSTRUCT A PORTFOLIO and CALCULATE PORTFOLIO RETURNS FOR EACH MONTHS

while True:
    total_allocation = 0
    for symbol in symbols:
        if symbol != '^GSPC':
            answer = None
            while not answer:
                try:
                    answer = float(input(f'Please insert the weight for {symbol} (example 0.25): '))
                except ValueError:
                    answer = -1
                database[symbol]['allocation'] = answer
                if answer < 0 or answer > 1:
                    print('Please insert a valid weight between 0 and 1')
                    answer = None
            total_allocation += float(answer)
    if total_allocation == 1.0:
        expected_returns = get_expected_returns(database)
        X = database['^GSPC']['monthly_log_returns'].dropna()
        y = expected_returns[1:]
        X1 = sm.add_constant(X)
        model = sm.OLS(y, X1)
        results = model.fit()
        print(results.summary())
        print(f"The beta is {results.params[1]}")
        input('Press enter to start again...')
    else:
        print('The weights you chose do not sum up to 1. Please, try again. ')
