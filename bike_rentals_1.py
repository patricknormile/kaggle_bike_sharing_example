# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 18:16:02 2020

@author: patno_000
"""

"""
Kaggle Comp - Bike rentals 

first submission did not do great, got a score of 0.78292

second submission
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime 

test = pd.read_csv('Kaggle/bike_rentals/test.csv')
train = pd.read_csv('Kaggle/bike_rentals/train.csv')

train['datetime'] = pd.to_datetime(train['datetime'], format='%Y-%m-%d %H:%M:%S')
test['datetime'] = pd.to_datetime(test['datetime'], format='%Y-%m-%d %H:%M:%S')

train['dayofweek'] = train['datetime'].dt.dayofweek
test['dayofweek'] = test['datetime'].dt.dayofweek

train['hour'] = train['datetime'].dt.hour
test['hour'] = test['datetime'].dt.hour

print(train.describe(include='all'))
print(test.describe(include='all'))



corr = train.loc[:,~train.columns.isin(['datetime'])].corr()
f, ax = plt.subplots(figsize=(20,14))
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

X_train_0 = train.loc[:,~train.columns.isin(['datetime','count'])]
y_train_0 = train.loc[:,['count']]
X_test_0 = test.loc[:,~test.columns.isin(['datetime'])]
#### goal, try lasso, ridge, decision tree regressor, random forrest, boosted trees

for col in train.columns:
    if col != 'datetime':
        plt.hist(train[col])
        plt.title(col)
        plt.show()


sns.boxplot(data=y_train_0)
### lots of outliers

def bxplts(data, count = 'count'):
    f, ax = plt.subplots(nrows=2, ncols=2)
    sns.boxplot(data=data, y=count, ax=ax[0][0])
    sns.boxplot(data=data, y=count,x='season', ax=ax[0][1])
    sns.boxplot(data=data, y=count, x='hour', ax=ax[1][0])
    sns.boxplot(data=data, y=count, x='workingday', ax=ax[1][1])

bxplts(train, 'count')
 




##### lasso regression
##### do this first to see what the most important features are        
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lass = Lasso(alpha=0.1)
lass.fit(X_train_0, y_train_0)

dict(zip(list(lass.coef_), list(X_train_0.columns)))

X_train_1 = X_train_0.loc[:,~X_train_0.columns.isin(['casual','registered'])]
X_test_1 = X_test_0.loc[:,~X_test_0.columns.isin(['casual','registered'])]
lass2 = Lasso(alpha=0.9)
lass2.fit(X_train_1, y_train_0)

#plt.plot(pd.DataFrame.from_dict(dict(zip(list(lass2.coef_), list(X_train_1.columns))), orient='index', columns=['col','coef']))
x = pd.DataFrame.from_dict(dict(zip(list(lass2.coef_), list(X_train_1.columns))), orient='index', columns=['col'])
x['coef'] = x.index

x.plot(x='col', y='coef', kind='bar')
#besides casual/registered, season and hour would be best predictors
# 
parms = {'alpha' : [*np.arange(0.1,1,0.1)],
         'fit_intercept' : [True, False]}

gridLass = GridSearchCV(estimator = lass2,
                  param_grid = parms,
                  scoring='neg_mean_squared_error',
                  refit = True,
                  cv = 5)
gridLass.fit(X_train_1, y_train_0)
lassB = gridLass.best_estimator_
gridLass.best_score_
gridLass.cv_results_
lassB.coef_
lassB.alpha
lassB.score
dir(lassB)
#


# gradient boosted trees

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
#stochastic
SGBM = GradientBoostingRegressor(max_depth=4, 
            subsample=0.9,
            max_features=0.75,
            n_estimators=200,                                
            random_state=2)

#not stochastic
GBM = GradientBoostingRegressor(n_estimators=200, 
            max_depth=4,
            random_state=2)

SGBM.fit(X_train_1, y_train_0.values.ravel())

params = {
            'max_depth' : [2,3,4,5,6,7,8,9,10],
            'n_estimators' : [100,200,400,600],
            'alpha' : [*np.arange(0.1,1,0.1)]
        }

RsCV = RandomizedSearchCV(estimator = SGBM,
                      param_distributions = params,
                      scoring = 'neg_mean_squared_error',
                      refit = True,
                      n_jobs = -1,
                      cv = 5,
                      verbose = 0)


RsCV.fit(X_train_1, y_train_0.values.ravel())
RsCV.cv_results_
SGBM_best = RsCV.best_estimator_
SGBM_best.get_params()

y_pred = SGBM_best.predict(X_test_1)
len(y_pred )


dts = test['datetime']
y_pred
y_pred_S = pd.Series(y_pred, name='count').clip(lower=0)
SGBM_out = pd.concat([dts, y_pred_S], axis=1)

SGBM_out.to_csv('Kaggle/bike_rentals/submission1.csv', index=False,
                header=True)



###################
"""

This got a score of about 0.78, so I can do much better.

next want to turn weather and seasons into dummy variables
normalize humidity and temp and see if that is a little better
"""

#log transform count
train['countlog'] = np.log(train['count'] +1)
y_train_1 = train.loc[:,['countlog']]
bxplts(train, 'countlog')
# far fewer outliers, mostly lower.

def IQR(array):
    x = np.percentile(array,[25, 75], axis=0)
    x0 = x[0]
    x1 = x[1]
    return x1 - x0

IQR1 = IQR(train['count'])
IQR2 = IQR(train['countlog'])



OL1 = train[(train['count'] > 1.5 * IQR1 + np.percentile(train['count'], 75)) |
            (train['count'] < - 1.5 * IQR1 + np.percentile(train['count'], 25))]['count']

OL2 = train[(train['countlog'] > 1.5 * IQR2 + np.percentile(train['countlog'], 75)) |
            (train['countlog'] < - 1.5 * IQR2 + np.percentile(train['countlog'], 25))]['countlog']


print(OL1.count(), OL2.count())
#roughly a third the outliers

#from sklearn.preprocessing import MinMaxScaler
#going to create my own minmax scaler 
def minmaxscl(df, columns):
    for col in columns:
        mn = np.min(df[col])
        mx = np.max(df[col])
        scl = (df[col] -  mn) / (mx - mn)
        df[col] = scl

colscale = ['temp','atemp', 'humidity', 'windspeed']
minmaxscl(train, colscale)
#minmaxscl(test, colscale)
catcols = ['season','weather','dayofweek','hour']


for col in catcols :
    train = train.join(pd.get_dummies(train[col], prefix = col))    


train.head()

X_train_2 = train.loc[:,~train.columns.isin(['datetime','count','countlog','casual','registered']+catcols)]

SGBM = GradientBoostingRegressor(max_depth=4, 
            subsample=0.9,
            max_features=0.75,
            n_estimators=200,                                
            random_state=2)

params = {
            'max_depth' : [2,3,4,5,6,7,8,9,10],
            'n_estimators' : [100,200,400,600],
            'alpha' : [*np.arange(0.1,1,0.1)]
        }

RsCV = RandomizedSearchCV(estimator = SGBM,
                      param_distributions = params,
                      scoring = 'neg_mean_squared_error',
                      refit = True,
                      n_jobs = -1,
                      cv = 5,
                      verbose = 0)


RsCV.fit(X_train_2, y_train_1.values.ravel())
RsCV.cv_results_
SGBM_best = RsCV.best_estimator_
SGBM_best.get_params()



#need to make test data same transformations as train data
minmaxscl(test, colscale)

for col in catcols :
    test = test.join(pd.get_dummies(test[col], prefix = col))    

X_train_2.columns
X_test_1.columns
X_test_0.columns
train.columns
X_test_2 = test.loc[:, ~test.columns.isin(['datetime','dayofweek','hour','count','countlog','casual','registered']+catcols)]
y_pred_2 = SGBM_best.predict(X_test_2)
 
predictions = np.exp(y_pred_2 )-1

#floor at 0
y_pred_S2 = pd.Series(predictions, name='count')
SGBM_out2 = pd.concat([dts, y_pred_S2], axis=1)

SGBM_out2.to_csv('Kaggle/bike_rentals/submission2.csv', index=False,
                header=True)

"""
score is much better at 0.45644
still room for improvement
"""



