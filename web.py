from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import random

app = Flask(__name__)

# Path to your dataset
DATASET_PATH = './dataset1.xlsx'  # Update to your actual file path

def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path)
    df = df.rename(columns={'Unnamed: 0': 'date'})
    newdf = df.drop('date', axis=1)
    return newdf

def calculate_annualized_returns(newdf, years=5):
    w_df = newdf
    w_df_subtracted = w_df - w_df.iloc[0]
    df_subtracted = w_df_subtracted.iloc[1:]
    percentage_returns = df_subtracted.iloc[-1] / newdf.iloc[0]
    annualized_returns = percentage_returns * 100 * years
    return annualized_returns.to_dict()

def calculate_daily_returns(df):
    returns = df.pct_change().dropna()
    return returns

def calculate_average_annual_return(returns, num_trading_days_per_year=252):
    average_daily_return = returns.mean()
    average_annual_return_percentage = average_daily_return * 100 * num_trading_days_per_year
    return average_annual_return_percentage

def calculate_variance(returns):
    variance = returns.var() * 5
    return variance.to_dict()

def calculate_sharpe_ratio(returns, rf=0.02):
    excess_returns = returns.mean() - rf
    std_dev = returns.std()
    sharpe_ratio = excess_returns / std_dev
    return sharpe_ratio

def generate_random_portfolios(matching_stocks, other_stocks, num_portfolios=100, min_portfolio_size=1, max_portfolio_size=30):
    def generate_random_portfolio(matching, other, min_size, max_size):
        portfolio_size = random.randint(min_size, max_size)
        if portfolio_size < len(matching):
            portfolio = random.sample(matching, portfolio_size)
        else:
            portfolio = matching.copy()
            num_other_stocks = portfolio_size - len(matching)
            if num_other_stocks > 0:
                portfolio.extend(random.sample(other, min(num_other_stocks, len(other))))
        return portfolio

    return [generate_random_portfolio(matching_stocks, other_stocks, min_portfolio_size, max_portfolio_size) for _ in range(num_portfolios)]

def calculate_portfolio_metrics(newdf, random_portfolios):
    expected_returns_global = {}
    expected_risk_global = {}
    expected_sharpe_ratio_global = {}
    optimal_portfolio = None
    highest_sharpe_ratio = -np.inf
    
    for portfolio in random_portfolios:
        portfolio_tuple = tuple(portfolio)
        portfolio_len = len(portfolio)
        
        concatenated_data = pd.concat([newdf[stock] for stock in portfolio], axis=1)
        log_returns_local = np.log(concatenated_data / concatenated_data.shift(1)).dropna()
        
        w = np.random.random(portfolio_len)
        w /= np.sum(w)
        
        mean_log_return = log_returns_local.mean()
        sigma = log_returns_local.cov()
        
        expected_return = np.sum(mean_log_return * w)
        expected_risk = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
        expected_sharpe_ratio = expected_return / expected_risk
        
        expected_returns_global[portfolio_tuple] = expected_return
        expected_risk_global[portfolio_tuple] = expected_risk
        expected_sharpe_ratio_global[portfolio_tuple] = expected_sharpe_ratio
        
        if expected_sharpe_ratio > highest_sharpe_ratio:
            highest_sharpe_ratio = expected_sharpe_ratio
            optimal_portfolio = portfolio_tuple
    
    return expected_returns_global, expected_risk_global, expected_sharpe_ratio_global, optimal_portfolio

def plot_portfolios(expected_returns_global, expected_risk_global, expected_sharpe_ratio_global, optimal_portfolio):
    fig, ax = plt.subplots(figsize=(8, 6))
    max_portfolio = max(expected_sharpe_ratio_global, key=expected_sharpe_ratio_global.get)
    scatter = ax.scatter(
        expected_risk_global.values(),
        expected_returns_global.values(),
        c=list(expected_sharpe_ratio_global.values()),
        cmap='viridis'
    )
    ax.set_xlabel('Expected Risk')
    ax.set_ylabel('Expected Returns')
    fig.colorbar(scatter, ax=ax, label='Expected Sharpe Ratio')
    ax.scatter(expected_risk_global[max_portfolio], expected_returns_global[max_portfolio], c='red', label='Max Sharpe Ratio')
    
    # Plot the optimal portfolio
    if optimal_portfolio:
        optimal_risk = expected_risk_global[optimal_portfolio]
        optimal_return = expected_returns_global[optimal_portfolio]
        ax.scatter(optimal_risk, optimal_return, c='blue', marker='x', s=100, label='Optimal Portfolio')
    
    ax.legend()
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close(fig)
    
    return plot_url

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/robo_advisor')
def robo_advisor():
    return render_template('robo_advisor.html')

@app.route('/result', methods=['POST'])
def result():
    amount = float(request.form['amount'])
    period = request.form['period']
    risk = request.form['risk']
    
    # Load the dataset
    newdf = load_and_prepare_data(DATASET_PATH)
    
    # Perform calculations
    annualized_returns = calculate_annualized_returns(newdf)
    returns = calculate_daily_returns(newdf)
    average_annual_return_percentage = calculate_average_annual_return(returns)
    variance = calculate_variance(returns)
    sharpe_ratio = calculate_sharpe_ratio(returns)
    
    # Generate random portfolios and calculate metrics
    matching_stocks = list(annualized_returns.keys())
    other_stocks = list(variance.keys())
    random_portfolios = generate_random_portfolios(matching_stocks, other_stocks)
    expected_returns_global, expected_risk_global, expected_sharpe_ratio_global, optimal_portfolio = calculate_portfolio_metrics(newdf, random_portfolios)
    
    # Calculate portfolio weights for the optimal portfolio
    investments = {}
    if optimal_portfolio:
        portfolio_data = pd.concat([newdf[stock] for stock in optimal_portfolio], axis=1)
        log_returns_local = np.log(portfolio_data / portfolio_data.shift(1)).dropna()
        portfolio_len = len(optimal_portfolio)
        weights = np.random.random(portfolio_len)
        weights /= np.sum(weights)
        total_investment = amount
        investments = {stock: weight * total_investment for stock, weight in zip(optimal_portfolio, weights)}
    
    # Generate plot
    plot_url = plot_portfolios(expected_returns_global, expected_risk_global, expected_sharpe_ratio_global, optimal_portfolio)
    
    return render_template('result.html', amount=amount, period=period, risk=risk, plot_url=plot_url, optimal_portfolio=optimal_portfolio, investments=investments)

if __name__ == '__main__':
    app.run(debug=True)
