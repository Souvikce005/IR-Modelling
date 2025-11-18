import pandas as pd
import numpy as np

def main():
    curve_historical_data = pd.read_excel('SOFR.xlsx')
    rates = curve_historical_data['Rate (%)']/100
    return rates

if __name__ == "__main__":
    main()


