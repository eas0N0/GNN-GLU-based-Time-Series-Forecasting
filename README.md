# Introduction

Time series forecasting plays a vital role in a wide range of real-world applications, from economics and finance to 
transportation. Predicting future trends based on historical data can help us make better decisions. For example, if we
can get accurate forecasts of the stock and the commodity markets, we can make considerable profits from the forecasts.
Moreover, if we can forecast the trend of COVID-19 new cases, we can allocate medical resources to prevent the epidemic
outbreak in advance.

Multivariate time series data contains time series from multiple interlinked data. In addition to forecasting the trend
based on historical temporal patterns within each time series, we can utilize the correlations between different time series
to improve the accuracy of forecasting. For instance, as shown in Figure 1, the prices of crude oil and gasoline are highly
correlated and the rise in crude oil may indicate a rise in the price of gasoline in the near future. The observation motivates
that utilising the information from both time series would help us better forecast the trend of each one.


# Methodology

The illustration of our model is shown in the following figure. Given multivariate time-series X as input, we first generate the
correlations among different time series and build the graph G. Then we put G into the following three layers. Firstly, the **Spectral Layer** contains a Graph Fourier Transform, an Intra-series Layer, a Spectral Graph Convolution, and an
inverse Graph Fourier Transform. It mainly captures the inter-series relationship. Secondly, the **Intra-series Layer**
performs Discrete Fourier Transform, 1D Convolution, GLU, and inverse Discrete Fourier Transform. It mainly learns
the intra-series features. Thirdly, the **Temporal Layer** applies the GLU and Fully-connected layer (FC), making the
prediction. The loss function of the GLU-GNN forecasting network can be written as:

![model](https://github.com/eas0N0/eas0N0.github.io/assets/129197157/182b0f63-1cff-48fa-b49f-70fee4878727)

# Experiments

**Motivation** &emsp; The financial market’s volatility is inherently complex and nonlinear, making it challenging to rely solely
on a trader’s expertise and instinct for analysis and decision-making. Therefore, an intelligent, scientific, and efficient
method for guiding stock trading is essential. In this study, we present a hybrid GNN-GLU model for predicting stock
prices in China’s A-share market.

**Data Sources and Preprocessing** &emsp; We obtained daily individual stock data for the China A-share market from CSMAR
and company filing data covering the period from December 31, 2019, to March 31, 2023. We classified the stocks into
six industry categories: Finance, Properties, Industrials, Conglomerates, Commerce, and Utilities. We selected from the
stocks with top 30% market capitalization to ensure model stability.
We chose the closing price on a given trading day as the input for the model. In our experiment, we selected nine stocks
with historical data of 500 trading days. We trained the GNN-GLU model based on these data and used a rolling strategy
with a window size of 30 to forecast the next 10 days’ prices. More specifically, after we forecast the price of day t, we
combined the forecasting value with the prices of the past 29 days to predict the price of day t+1.

**Results and Discussion** &emsp; Our experiment results yielded three main findings. First, our hybrid GNN-GLU model
outperforms other methods in terms of accuracy indicators, including Mean Absolute Error (MAE), Root Mean Square
Error (RMSE), and Mean Absolute Percentage Error (MAPE). Second, our model can roughly capture the trend of stock
price (shown in Figures 5, 6, 7, and 8), thus exhibiting potential profitability in financial applications. Through numerous
experiments, we constructed various trading strategies that generate positive returns, the details of which are provided in
the appendix. Third, our model demonstrates high interpretability connected with reality. The illustration of the matrix
\mathcal{W}  in Figure 3 reflects the strength of the relationship between different stocks, which can help people better understand
the dynamics of the stock market.

![res1](https://github.com/eas0N0/eas0N0.github.io/assets/129197157/2d7f4417-6f02-4628-b2d7-ba94af2628d4)
![res2](https://github.com/eas0N0/eas0N0.github.io/assets/129197157/8c9d1402-c644-4a9a-8d11-f5a9ef42049c)
![res3](https://github.com/eas0N0/eas0N0.github.io/assets/129197157/d87afc81-f77e-4c72-9589-8cc66c0e0f4d)

# Conclusion
In this project, we propose a new forecasting model GNN-GLU to take advantage of inter-series correlations and temporal
dependencies by shuttling between the temporal domain and spectral domain and capturing corresponding features.
We also proposed three approaches to generate the correlation matrix, which is proven to have a significant impact on
performance. The results show that GNN-GLU outperforms other forecasting models in the database of the stock market
and COVID-19. Moreover, GNN-GLU exhibits strong interpretability regarding the generation and capture of correlation
among time series. There are three main directions for further work. First, We could try more methods in the correlation
feature extraction process. Second, we may employ residual connections to stack more blocks in our model to improve
forecasting accuracy. Third, we will look for its application to more real-world scenarios and try some large-scale datasets.

