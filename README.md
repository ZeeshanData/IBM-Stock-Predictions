ABSTRACT 
In this project I attempt to implement machine learning approach to predict stock prices. Machine learning is effectively implemented in forecasting stock prices. The objective of this predict the stock prices in order to make more informed and accurate investment decisions. We propose a stock price prediction system that integrates mathematical functions, machine learning, and other external factors for the purpose of achieving better stock prediction accuracy and issuing profitable trades.
There are two types of stocks. You may know of intraday trading by the commonly used term "day trading." Interday traders hold securities positions from at least one day to the next and often for several days to weeks or months. LSTMs are very powerful in sequence prediction problems because they’re able to store past information. This is important in our case because the previous price of a stock is crucial in predicting its future price. While predicting the actual price of a stock is an uphill climb, we can build a model that will predict whether the price will go up or down.


CHAPTER 1 
INTRODUCTION 
The financial market is a dynamic and composite system where people can buy and sell currencies, stocks, equities and derivatives over virtual platforms supported by brokers. The stock market allows investors to own shares of public companies through trading either by exchange or over the counter markets. This market has given investors the chance of gaining money and having a prosperous life through investing small initial amounts of money, low risk compared to the risk of opening new business or the need of high salary career. Stock markets are affected by many factors causing the uncertainty and high volatility in the market. Although humans can take orders and submit them to the market, automated trading systems (ATS) that are operated by the implementation of computer programs can perform better and with higher momentum in submitting orders than any human. However, to evaluate and control the performance of ATSs, the implementation of risk strategies and safety measures applied based on human judgements are required. Many factors are incorporated and considered when developing an ATS, for instance, trading strategy to be adopted, complex mathematical functions that reflect the state of a specific stock, machine learning algorithms that enable the prediction of the future stock value, and specific news related to the stock being analyzed. 
Time-series prediction is a common technique widely used in many real-world applications such as weather forecasting and financial market prediction. It uses the continuous data in a period of time to predict the result in the next time unit. Many time- series prediction algorithms have shown their effectiveness in practice. The most common algorithms now are based on Recurrent Neural Networks (RNN), as well as its special type - Long-short Term Memory (LSTM) and Gated Recurrent Unit (GRU). Stock market is a typical area that presents time-series data and many research study on it and proposed various models. In this project, LSTM model is used to predict the stock price. 


1.1 MOTIVATION FOR WORK 
Businesses primarily run over customer’s satisfaction, customer reviews about their products. Shifts in sentiment on social media have been shown to correlate with shifts in stock markets. Identifying customer grievances thereby resolving them leads to customer satisfaction as well as trustworthiness of an organization. Hence there is a necessity of an un biased automated system to classify customer reviews regarding any problem. In today’s environment where we’re justifiably suffering from data overload (although this does not mean better or deeper insights), companies might have mountains of customer feedback collected; but for mere humans, it’s still impossible to analyze it manually without any sort of error or bias. Oftentimes, companies with the best intentions find themselves in an insights vacuum. You know you need insights to inform your decision making and you know that you’re lacking them, but don’t know how best to get them. Sentiment analysis provides some answers into what the most important issues are, from the perspective of customers, at least. Because sentiment analysis can be automated, decisions can be made based on a significant amount of data rather than plain intuition. 
1.2 PROBLEM STATEMENT 
Time Series forecasting & modelling plays an important role in data analysis. Time series analysis is a specialized branch of statistics used extensively in fields such as Econometrics & Operation Research. Time Series is being widely used in analytics & data science. Stock prices are volatile in nature and price depends on various factors. The main aim of this project is to predict stock prices using Long short term memory (LSTM). 

CHAPTER 2 
LITERATURE SURVEY 
"What other people think” has always been an important piece of information for most of us during the decision-making process. The Internet and the Web have now (among other things) made it possible to find out about the opinions and experiences of those in the vast pool of people that are neither our personal acquaintances nor well-known professional critics — that is, people we have never heard of. And conversely, more and more people are making their opinions available to strangers via the Internet. The interest that individual users show in online opinions about products and services, and the potential influence such opinions wield, is something that is driving force for this area of interest. And there are many challenges involved in this process which needs to be walked all over in order to attain proper outcomes out of them. In this survey we analysed basic methodology that usually happens in this process and measures that are to be taken to overcome the challenges being faced. 
CHAPTER 3 
METHODOLOGY 
3.1 PROPOSED SYSTEMS 
The prediction methods can be roughly divided into two categories, statistical methods and artificial intelligence methods. Statistical methods include logistic regression model, ARCH model, etc. Artificial intelligence methods include multi-layer perceptron, convolutional neural network, naive Bayes network, back propagation network, single-layer LSTM, support vector machine, recurrent neural network, etc. They used Long short-term memory network (LSTM). 


Long short-term memory network: 
Long short-term memory network (LSTM) is a particular form of recurrent neural network (RNN). 
Working of LSTM: 
LSTM is a special network structure with three “gate” structures. Three gates are placed in an LSTM unit, called input gate, forgetting gate and output gate. While information enters the LSTM’s network, it can be selected by rules. Only the information conforms to the algorithm will be left, and the information that does not conform will be forgotten through the forgetting gate. 
The experimental data in this paper are the actual historical data downloaded from the Internet. Three data sets were used in the experiments. It is needed to find an optimization algorithm that requires less resources and has faster convergence speed. 
•	Used Long Short-term Memory (LSTM) with embedded layer and the LSTM neural network with automatic encoder. 
•	LSTM is used instead of RNN to avoid exploding and vanishing gradients. 
•	In this project python is used to train the model, MATLAB is used to reduce 
dimensions of the input. MySQL is used as a dataset to store and retrieve data. 
•	The historical stock data table contains the information of opening price, the highest 
price, lowest price, closing price, transaction date, volume and so on. 
•	The accuracy of this LSTM model used in this project is 57%. 
CHAPTER 4 
EXPERIMENT ANALYSIS 
This project can run on commodity hardware. We ran entire project on an Intel I5 processor with 8 GB Ram, 2 GB Nvidia Graphic Processor, It also has 2 
cores which runs at 1.7 GHz, 2.1 GHz respectively. First part of the is training phase which takes 10-15 mins of time and the second part is testing part which only takes few seconds to make predictions and calculate accuracy. 


CODE 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from joblib import Parallel, delayed
from collections import deque 
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import tensorflow as tf
from keras.models import Sequential
from keras.layers import BatchNormalization, LSTM
from keras.layers.core import Dropout, Dense

%matplotlib inline
plt.rcParams["figure.figsize"] = (20,10)
plt.rcParams["figure.autolayout"] = True
plt.rcParams['font.size'] = '20'


data = pd.read_csv('/Users/zee/Documents/2 Year IBM Stock Data.csv', index_col='time')
data.index = pd.to_datetime(data.index, format='%m/%d/%Y %H:%M')

print(data[:10])
data.head()

data.info()
data.describe()
data = data.resample('H').mean().dropna()
data.head()
data.isna().sum()
s1 = data.last('4D')

fig = go.Figure(data=[go.Candlestick(x=sample.index,
                open=sample['open'],
                high=sample['high'],
                low=sample['low'],
                close=sample['close'])])

fig.show()

close_C = df['close']

close_C.rolling(window=24).mean().plot(label='Mean - 24hrs')
close_C.rolling(window=24*60).mean().plot(label='Mean - 60days')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Mean close')
plt.legend();

data_change = data['close'].copy().pct_change().dropna()

area_size = np.pi * 20

plt.figure(figsize=(8, 4))
plt.scatter(data_change.mean(), data_change.std(), s=area)
plt.xlabel('Expected return')
plt.ylabel('Risk')

plt.annotate('IBM', xy=(data_change.mean(), data_change.std()), xytext=(40, 35), textcoords='offset points', ha='right', va='bottom', 
                arrowprops=dict(arrowstyle='-', color='blue', connectionstyle='arc3,rad=-0.3'))

data_copy = data.iloc[:, 3:4].copy()
data_copy.head()
data_copy['target'] = data_copy['close'].shift(periods=-10)
data_copy.head()

def bootstrap(df, freq, shift_max=0, n_samples=1, replace=False,
                       n_jobs=1):

    leap_mask = (~((df.index.month == 2) & (df.index.day == 29)))
    new_df = df.copy() #df without leap years
    new_df = new_df[leap_mask]       
    parallel = Parallel(n_jobs=n_jobs)
    
    def bootstrap(i):
        """
        Returns
        -------
        df_res : pd.DataFrame
            seasonally bootstrapped data
        """
        random.seed(i)
                
        if replace == True:
            index = new_df.index
            blocks = create_blocks(new_df, freq, shift_max)
            #convert list of blocks into array of blocks in order to use indexing
            b = np.empty(len(blocks), dtype=object)
            b[:] = blocks
            blocks = b
            lengths = [len(block) for block in blocks]
            ndays = np.bincount(lengths).argmax()
            blocks_to_shuffle = np.where(lengths == ndays)[0]
            new_indices = np.copy(blocks_to_shuffle)
            new_indices = np.random.choice(new_indices, len(new_indices),
                                           replace=True)
            blocks[blocks_to_shuffle]=blocks[new_indices]
                
        else:
            index = df.index
            blocks = create_blocks(df, freq, shift_max)
            random.shuffle(blocks)
        
                
        df_res =  pd.concat(blocks)

        if isinstance(df_res, pd.DataFrame):
            df_res.set_index(index, inplace=True)
            
        else:
            df_res.index = index
            
        return df_res
    
    
    return parallel(
            delayed(create_seasonal_bootstrap)(i) for i in range(n_samples)
            )


def create_blocks(df, freq, shift_max=0):
  
    shift = random.randint(-shift_max, shift_max)
    df = df.shift(shift)
    #we retrieve groups through the groupby method            
    blocks = deque([gp[1] for gp in df.groupby(pd.Grouper(freq=freq))])
        
    return blocks


bootstrap_samples = seasonal_bootstrap(df=df_copy,
                                       freq='M',
                                       shift_max=0,
                                       n_samples=6,
                                       replace=True,
                                       n_jobs=1)

def split_data(df, split_fraction):
    split_size = round(len(df)*split_fraction)
    train = df[:split_size]
    test = df[split_size:]
    return train, test
train_split, test_split = split_data(bootstrap_samples[1], 0.7)

print('train_split.shape', train_split.shape)
print('test_split.shape', test_split.shape)

scaler_train = MinMaxScaler(feature_range=(0,1))
scaler_train = scaler_train.fit(train_split)

scaled_train = scaler_train.transform(train_split)
scaled_test = scaler_train.transform(test_split)

print('scaled train', scaled_train.shape)
print('scaled test', scaled_test.shape)

def split_sequence(sequence, n_steps):
    X, y = list(), list()

    for i in range( len(sequence) ):
        end_ix = i + n_steps

        if end_ix > len(sequence)-1:
            break

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X)[:,:,:-1], np.array(y)[:,-1]



X_train, Y_train = split_sequence(scaled_train, 24)
X_test, Y_test = split_sequence(scaled_test, 24)

print(X_train.shape)
print(Y_train.shape)
print("")
print(X_test.shape)
print(Y_test.shape)

def lstm_model(X, Y_train, nb_neuron_dense, epoch):
    tf.keras.backend.clear_session()

    model = Sequential()
    model.add(LSTM(nb_neuron_dense, return_sequences=True, input_dim=n_input))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=32))
    model.add(Dropout(0.2))

    model.add(Dense(units=1, activation='linear'))

    model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])

    history = model.fit(X, Y_train,
                        batch_size=16, epochs=epoch, verbose=1)
    
    return history, model

n_input = X_train.shape[1] * X_train.shape[2]
x_train_reshape = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
x_test_reshape = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

def rescale_pred(X_test, Y_test_predicted, Y_test):
    concat_predict = np.concatenate( (X_test[:,0,:], Y_test_predicted), 1 )
    concat_real = np.concatenate( (X_test[:,0,:], Y_test.reshape( Y_test.shape[0], 1 )), 1 )

    Y_test_predicted_rescaled = scaler_train.inverse_transform(concat_predict)
    Y_test_real_rescaled = scaler_train.inverse_transform(concat_real)

    total = pd.concat([pd.DataFrame(Y_test_real_rescaled).iloc[:,-1],
                       pd.DataFrame(Y_test_predicted_rescaled).iloc[:,-1]],1)
    total.columns = ['Real', 'Predicted']
    
    return total
def error(df_actual, df_predicted):
    mse = mean_squared_error(df_actual, df_predicted, squared=True)
    rmse  = mean_squared_error(df_actual, df_predicted, squared=False)
    mae =  mean_absolute_error(df_actual, df_predicted)

    errorS = pd.DataFrame([mse, rmse, mae], index=['MSE','RMSE','MAE'])
    
    return errorS
def plot_val_loss(history):
    plt.plot(pd.DataFrame(history.history)['loss'], label = 'loss')
    plt.legend()
    plt.title("Number epoch: " +  str(len(pd.DataFrame(history.history))))
    return pd.DataFrame(history.history)
history, model = lstm_model(x_train_reshape, Y_train,
                                nb_neuron_dense=128,
                                epoch = 20)

plot_val_loss(history)
Y_test_predicted = model.predict(x_test_reshape, verbose=0)

concat_predict = np.concatenate( (X_test[:,0,:], Y_test_predicted), 1 )
Y_test_predicted_rescaled = scaler_train.inverse_transform(concat_predict)

concat_real = np.concatenate( (X_test[:,0,:], Y_test.reshape(Y_test.shape[0],1)), 1 )
Y_test_real_rescaled = scaler_train.inverse_transform(concat_real)

df_predicted = pd.concat([ pd.DataFrame( Y_test_predicted_rescaled[:,-1] ), pd.DataFrame( Y_test_real_rescaled[:,-1] ) ], 1)
df_predicted.columns = ["Predicted",'Real']
df_predicted.index = test_split[:-24].index
df_predicted = df_predicted.dropna()

train_split['target'].plot(label='Train split')
df_predicted['Predicted'].plot(label='Predicted')
df_predicted['Real'].plot(label='Real')
plt.legend();


errorS = pd.DataFrame(error(df_predicted['Real'], df_predicted['Predicted'])).T
errorS.columns=['MSE','RMSE','MAE']
errorS.plot.bar()
print(errorS)
plt.show()
