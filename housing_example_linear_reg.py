import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

#Read raw csv file without headers and comma delimiter
df=pd.read_csv('housing.data',delim_whitespace=True,header=None)
#print(df)

#give names for each column
col_name=['CRIM','ZIN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PIRATIO','B','LSTAT','MEDV']
df.columns=col_name

#plot and analyse all datas
##sns.pairplot(df,size=1.5)
##plt.show()

#plot and analyse the interested columns saperately
col_study=['CRIM','ZIN','INDUS','CHAS','MEDV']
sns.pairplot(df[col_study],size=1.5)
plt.show()

#correlarion
##print(df.corr())

#correlation heat map
plt.figure(figsize=(16,10))
sns.heatmap(df.corr(),annot=True)
plt.show()

#correlation heat map for specified columns
plt.figure(figsize=(16,10))
sns.heatmap(df[col_study].corr(),annot=True)
plt.show()


#UNIVARIATE LINEAR REGRESSION USING 'RM' VS 'MEDV' ('RM'=Maximum correlated value)
X=df['RM'].values
X=X.reshape(-1,1)
y=df['MEDV'].values
model=LinearRegression()
model.fit(X,y)
print(model.coef_,model.intercept_)
plt.figure(figsize=(12,10));
sns.regplot(X,y);
plt.xlabel('average number of rooms per dwelling')
plt.ylabel('Mediamn value of owner-occupied homes in $1000');
plt.show();
sns.jointplot(x='RM',y='MEDV',data=df, kind='reg', size=10);
plt.show();

#UNIVARIATE LINEAR REGRESSION USING 'LSTAT' VS 'MEDV' ('LSTAT'=Minimum correlated value)
X=df['LSTAT'].values
X=X.reshape(-1,1)
y=df['MEDV'].values
model=LinearRegression()
model.fit(X,y)
print(model.coef_,model.intercept_)
plt.figure(figsize=(12,10));
sns.regplot(X,y);
plt.xlabel('average number of LSTAT')
plt.ylabel('Mediamn value of owner-occupied homes in $1000');
plt.show();
sns.jointplot(x='LSTAT',y='MEDV',data=df, kind='reg', size=10);
plt.show();
