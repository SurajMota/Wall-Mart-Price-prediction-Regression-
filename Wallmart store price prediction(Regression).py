# Importing Liabries
import numpy as np
import pandas as pd

# Importing data from csv file
Features_Data=pd.read_csv("features.csv")
Features_Data.head()

Features_Data.info()
Features_Data.shape
Features_Data.isnull().sum()

#Checking the missing value in Percentage
na_percentage=((Features_Data.isnull().sum()/Features_Data.shape[0])*100).sort_values(ascending=False)
na_percentage

#Data Wrangling
Features_Data.fillna(value={'MarkDown1':0,'MarkDown2':0,'MarkDown3':0,'MarkDown4':0,'MarkDown5':0},inplace=True)
Features_Data.interpolate(inplace=True)


Store_Data=pd.read_csv("stores.csv")
Store_Data.head()

#Joining the tables
Data1 = Features_Data.join(Store_Data.set_index("Store"), on="Store")
Data1.head()

Train_Data=pd.read_csv("train.csv")
Train_Data.head()

Data2=Data1.join(Train_Data[["Store","Date","Dept","Weekly_Sales"]].set_index(["Store","Date"]),on=["Store","Date"])
Data2


z=Data2["Weekly_Sales"].isnull()
Data2[z]


Data2.dropna(inplace=True)
Data2.isnull().sum()



#Feature Engineering
import seaborn as sns
corr = Data2.corr()
corr = np.abs(corr)   #abs if its negative impact or positive impact show both
sns.set(rc={'figure.figsize':(20,15)})
hm =sns.heatmap(corr, annot=True)
hm


corr["Weekly_Sales"].sort_values()


#We can do scalling but there was no impact of it so I avoid it.
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#scaler = sc.fit(X)   #through Fit() it will learn
#x = scaler.transform(X_train)  #through transform() apply
#x = scaler.transform(X_test)

#Train Test split
from sklearn.model_selection import train_test_split
Data2Train,Data2Test=train_test_split(Data2,test_size=0.30,random_state=88)
Data2Train.shape,Data2Test.shape

# Tking Dependent and Independent for the prediction
y=Data2Train["Weekly_Sales"]
x=Data2Train.drop(["Weekly_Sales","Date","Type"],axis=1)

#Data visualtzation
%matplotlib notebook
import matplotlib.pyplot as plt

plt.figure()
plt.scatter(Data2["Temperature"],Data2["Weekly_Sales"],color="green")
plt.xlabel("Temperature",color="red")
plt.ylabel("Weekly_Sales",color="red")
plt.title("Relation between Temperature and Weekly_Sales",color="black")

plt.figure()
plt.scatter(Data2["CPI"],Data2["Weekly_Sales"],color="yellow")
plt.xlabel("CPI",color="red")
plt.ylabel("Weekly_Sales",color="red")
plt.title("Relation between CPI and Weekly_Sales",color="black")

plt.figure()
plt.scatter(Data2["Fuel_Price"],Data2["Weekly_Sales"],color="orange")
plt.xlabel("Fuel_Price",color="red")
plt.ylabel("Weekly_Sales",color="red")
plt.title("Relation between Fuel_Price and Weekly_Sales",color="black")

#Linear Regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x,y)
reg.score(x,y)
test=Data2Test.drop(["Weekly_Sales","Date","Type"],axis=1)
test.shape
Pred_test=reg.predict(test)
Pred_test
k=Data2Test["Weekly_Sales"]

from sklearn import metrics
metrics.r2_score(k,Pred_test)



#Decision Tree Algo
from sklearn.tree import DecisionTreeRegressor
model3=DecisionTreeRegressor(random_state=42,max_depth=10)
y=Data2Train["Weekly_Sales"]
x=Data2Train.drop(["Weekly_Sales","Date","Type"],axis=1)
model3.fit(x,y)
model3.score(x,y)
Pred_test3=model3.predict(test)
Pred_test3
c=Data2Test["Weekly_Sales"]

from sklearn import metrics
metrics.r2_score(c,Pred_test3)



#Random Forest Algo
from sklearn.ensemble import RandomForestRegressor
model1=RandomForestRegressor(n_estimators=200,random_state=42,oob_score=True,max_depth=10)
model1.fit(x,y)
model1.score(x,y)
Pred_test1=model1.predict(test)
Pred_test1
c=Data2Test["Weekly_Sales"]

from sklearn import metrics
metrics.r2_score(c,Pred_test1)

#As we can clearly see Random Forest shows best Result of accuracy 87 percent on test data. So we will select model1