import pandas as pd
import statsmodels.api as sm
import Input_Data

rates_data = Input_Data.main()

#OLS Regression to estimate k, theta, sigma

def calibrate_params(rates_data):
    rates_down = rates_data.iloc[1:].reset_index(drop=True)
    rates = rates_data.iloc[:-1]
    data = pd.concat([rates, rates_down], axis=1)
    data.columns = ['rates', 'rates_t-1']
    x = sm.add_constant(data['rates_t-1'])
    y = data['rates']
    result_OLS = sm.OLS(y, x).fit()
    print(result_OLS.summary())

def main():
    calibrate_params(rates_data)

if __name__ == "__main__":
    main()