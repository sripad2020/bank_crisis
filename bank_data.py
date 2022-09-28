import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('finaldataset_1.csv')
print(data)
print(data.columns)
print('-----NaN------')
print(data.isna().sum())
print('-----------------')
print(data.info())
data['tradeshare']=data['tradeshare'].fillna(data['tradeshare'].median())
data['expgrowth']=data['expgrowth'].fillna(data['expgrowth'].median())
data['expgrowthTRIM']=data['expgrowthTRIM'].fillna(data['expgrowthTRIM'].median())
data['BANK']=data['BANK'].fillna(data['BANK'].mean())
data['TWIN']=data['TWIN'].fillna(data['TWIN'].mean())
data['RZ']=data['RZ'].fillna(data['RZ'].mean())
data['FL']=data['FL'].fillna(data['FL'].mean())
data['TANG']=data['TANG'].fillna(data['TANG'].mean())
data['ofagdp']=data['ofagdp'].fillna(data['ofagdp'].mean())
data['pcrdbofgdp']=data['pcrdbofgdp'].fillna(data['pcrdbofgdp'].mean())
data['stmktcap']=data['stmktcap'].fillna(data['stmktcap'].mean())
data['herf']=data['herf'].fillna(data['herf'].mean())
data['intout']=data['intout'].fillna(data['intout'].mean())
data['n']=data['n'].fillna(data['n'].mean())
data['homogeneity']=data['homogeneity'].fillna(data['homogeneity'].mean())
data['rd']=data['rd'].fillna(data['rd'].median())
data['caplab']=data['caplab'].fillna(data['caplab'].mean())
data['rznoncrisis']=data['rznoncrisis'].fillna(data['rznoncrisis'].mean())
data['RZyoung']=data['RZyoung'].fillna(data['RZyoung'].mean())
data['CCC']=data['CCC'].fillna(data['CCC'].mean())
data['INVSA']=data['INVSA'].fillna(data['INVSA'].mean())
data['policytot']=data['policytot'].fillna(data['policytot'].mean())
data['debtrelief']=data['debtrelief'].fillna(data['debtrelief'].mean())
data['recaps']=data['recaps'].fillna(data['recaps'].mean())
data['forbb']=data['forbb'].fillna(data['forbb'].mean())
data['forba']=data['forba'].fillna(data['forba'].mean())
data['liqsup']=data['liqsup'].fillna(data['liqsup'].median())
data['blanguar']=data['blanguar'].fillna(data['blanguar'].mean())
data['stmktcap']=data['stmktcap'].fillna(data['stmktcap'].mean())
data['pcrdbofgdp']=data['pcrdbofgdp'].fillna(data['pcrdbofgdp'].mean())

'''print(data[['year','product','tradevalue','tradeshare','expgrowth','expgrowthTRIM','BANK'
                            ,'BANK_W3','TWIN','RZ','FL','TANG','ofagdp','pcrdbofgdp','stmktcap',
                            'RecessionAbroad','GDPgrAbroad','durables','loss','loss2','GDPcap','developed',
                            'developing','blanguar','liqsup','forba','forba','forbb','recaps','debtrelief',
                            'policytot','GDPgr','INVSA','CCC','RZyoung','rznoncrisis',
                            'caplab','rd','homogeneity','n','herf','intout','contcrisis']].count())'''
#calculating recesssion

#trade value
#trade share
#product
#exp growth
#exp growth
#Fl
#developed
#developing
#liqsup
'''print(data['developed'].value_counts())
print(data['developing'].value_counts())
print(data['liqsup'].value_counts())'''
import  keras
from keras.models import Sequential
from keras.layers import Dense
import keras.activations,keras.metrics,keras.losses

from sklearn.model_selection import train_test_split
#x=data[['product','tradevalue','tradeshare','expgrowth','expgrowthTRIM','FL','developed','developing','liqsup']]
y=data['recession']
from sklearn.feature_selection import SelectKBest
feature_selection=SelectKBest(k=15)
feature_selection.fit(data[['year','product','tradevalue','tradeshare','expgrowth','expgrowthTRIM','BANK'
                            ,'BANK_W3','TWIN','RZ','FL','TANG','ofagdp','pcrdbofgdp','stmktcap',
                            'RecessionAbroad','GDPgrAbroad','durables','loss','loss2','GDPcap','developed',
                            'developing','blanguar','liqsup','forba','forba','forbb','recaps','debtrelief',
                            'policytot','GDPgr','INVSA','CCC','RZyoung','rznoncrisis',
                            'caplab','rd','homogeneity','n','herf','intout','contcrisis']],data['recession'])
print(feature_selection.pvalues_)

x=data[feature_selection.get_feature_names_out()]
x_train,x_test,y_train,y_test=train_test_split(x,y)

from sklearn.linear_model import Lasso,Ridge,LinearRegression
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
prediction=dt.predict(x_test)
print(dt.score(x_test,y_test))
from sklearn.metrics import r2_score
r2=r2_score(y_test,prediction)
print(r2)
import numpy as np
from sklearn.metrics import mean_squared_error
af=mean_squared_error(y_test,prediction)
print(af)
abc=np.sqrt(af)
print(abc)

#Forecasting the tradevalue
from statsmodels.tsa.holtwinters import ExponentialSmoothing
smoothing=ExponentialSmoothing(data['tradevalue']).fit()
forecast=smoothing.forecast(steps=35)
pred=smoothing.predict(34)
plt.plot(pred)
plt.plot(forecast,marker='o')
plt.show()
plt.plot(data['tradevalue'].sample(35))
plt.plot(forecast,marker='o')
plt.show()
#developed or not developed
features=SelectKBest(k=10)
features.fit(data[['year','product','tradevalue','tradeshare','expgrowth','expgrowthTRIM','BANK'
                            ,'BANK_W3','TWIN','RZ','FL','TANG','ofagdp','pcrdbofgdp','stmktcap',
                            'RecessionAbroad','GDPgrAbroad','durables','loss','loss2','GDPcap','developed',
                            'developing','blanguar','liqsup','forba','forba','forbb','recaps','debtrelief',
                            'policytot','GDPgr','INVSA','CCC','RZyoung','rznoncrisis',
                            'caplab','rd','homogeneity','n','herf','intout','contcrisis']],data['recession'])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data[features.get_feature_names_out()],data['recession'])
model=Sequential()
model.add(Dense(input_dim=data[features.get_feature_names_out()].shape[1],units=x_train.shape[1],activation=keras.activations.relu))
model.add(Dense(input_dim=data[features.get_feature_names_out()].shape[1],units=x_train.shape[1],activation=keras.activations.relu))
model.add(Dense(input_dim=data[features.get_feature_names_out()].shape[1],units=x_train.shape[1],activation=keras.activations.relu))
model.add(Dense(input_dim=data[features.get_feature_names_out()].shape[1],units=x_train.shape[1],activation=keras.activations.relu))
model.add(Dense(units=1,activation=keras.activations.relu))
model.compile(optimizer='adam',metrics=['mse'],loss=keras.losses.mean_squared_error)
model.fit(x_train,y_train,batch_size=10,epochs=25)
pred=model.predict(x_test)
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
mse=mean_squared_error(y_test,pred)
print(mse)
rmse=np.sqrt(mean_squared_error(y_test,pred))
print(rmse)
ma=r2_score(y_test,pred)
print(ma)

print(data['developed'])
#developed or not developed
features=SelectKBest(k=10)
features.fit(data[['year','product','tradevalue','tradeshare','expgrowth','expgrowthTRIM','BANK'
                            ,'BANK_W3','TWIN','RZ','FL','TANG','ofagdp','pcrdbofgdp','stmktcap',
                            'RecessionAbroad','GDPgrAbroad','durables','loss','loss2','GDPcap','recession',
                            'developing','blanguar','liqsup','forba','forba','forbb','recaps','debtrelief',
                            'policytot','GDPgr','INVSA','CCC','RZyoung','rznoncrisis',
                            'caplab','rd','homogeneity','n','herf','intout','contcrisis']],data['developed'])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data[features.get_feature_names_out()],data['developed'].astype(dtype='float32'))
y_train=np.asarray(y_train).reshape((-1,1))
y_test=np.asarray(y_test).reshape((-1,1))
model=Sequential()
model.add(Dense(input_dim=data[features.get_feature_names_out()].shape[1],units=x_train.shape[1],activation=keras.activations.relu))
model.add(Dense(input_dim=data[features.get_feature_names_out()].shape[1],units=x_train.shape[1],activation=keras.activations.relu))
model.add(Dense(input_dim=data[features.get_feature_names_out()].shape[1],units=x_train.shape[1],activation=keras.activations.relu))
model.add(Dense(input_dim=data[features.get_feature_names_out()].shape[1],units=x_train.shape[1],activation=keras.activations.relu))
model.add(Dense(units=1,activation=keras.activations.sigmoid))
model.compile(optimizer='adam',metrics=['accuracy'],loss=keras.losses.binary_crossentropy)
model.fit(x_train,y_train,batch_size=30,epochs=25)
prediction=model.predict(x_test)
print(model.evaluate)