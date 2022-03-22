# -*- coding: utf-8 -*-
"""
Created on Sat May 15 21:06:28 2021

@author: justi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

#%% Q1

alldata = pd.read_csv("middleSchoolData.csv", delimiter=",")
alldata2 = alldata.to_numpy()


apps=alldata["applications"]
accp=alldata["acceptances"]

correlation = np.corrcoef(apps,accp)[0,1]

slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(apps,accp)


plt.plot(apps,accp, "o")
plt.plot(apps, slope1*apps + intercept1)
plt.xlabel('Applications')
plt.ylabel('Acceptances')
#Seems like applications are a better predictor than application rate

#%% Q2

size=alldata["school_size"]

rate=np.empty([594,])

for i in range(594):
    ratec = apps[i]/size[i]
    rate[i] = ratec    


accp2 = accp.to_numpy()

com = np.vstack((rate,accp2))
com = np.transpose(com)
whereNAN = np.isnan(com)
NANcoord = np.where(whereNAN==True)
com = np.delete(com,NANcoord[0],0)
correlation2 = np.corrcoef(com[:,0],com[:,1])

slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(com[:,0],com[:,1])


plt.plot(com[:,0],com[:,1], "o")
plt.plot (com[:,0], slope2*com[:,0] + intercept2)
plt.xlabel('Application rate')
plt.ylabel('Acceptances')

#%% Q3

arate = np.empty([594,])

for i in range(594):
    aratec = accp[i]/size[i]
    arate[i] = aratec    


maxrate = np.nanmax(arate)
maxindex = np.where(arate == maxrate)[0]

index0 = []
for i in range(594):
    index0.append(i+1) 

index0 = np.array(index0)


plt.bar(index0,arate, width=5)
plt.xlabel('Index')
plt.ylabel('Acceptance Rate (NOT ODDS)')
plt.show()
#ROW 306 in Excel sheet, ____ applied students got accepted.
#Odds are for getting to not getting in, _____

#%% Q4

#Do some PCA here.

combined2 =  alldata2[:,[11,12,13,14,15,16,21,22,23]]

float_arr = np.vstack(combined2[:, :]).astype(np.float)

whereNAN = np.isnan(float_arr)
NANcoord = np.where(whereNAN==True)
combined2 = np.delete(float_arr,NANcoord[0],0)

percieved = combined2[:,[0,1,2,3,4,5]]
perform =  combined2[:,[6,7,8]]

zpercieve = stats.zscore(percieved)

zperform = stats.zscore(perform)

#Sanity check code
meancheck = np.mean(zperform,axis=0)

pca = PCA()
pca.fit(zpercieve)
eigen = pca.explained_variance_
loadings = pca.components_
rotate = pca.fit_transform(zpercieve)
covar = eigen/sum(eigen)*100

#Scree plots.

plt.plot(eigen)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([0,5],[1,1],color='red',linewidth=1) # Kaiser criterion line

normalizedweight1 = eigen/np.sum(eigen)

#%% PCA1 part 2
whichPrincipalComponent = 0 

plt.bar(np.linspace(1,6,6),np.transpose(loadings)[:,whichPrincipalComponent])
plt.xlabel('Rating')
plt.ylabel('Loading')

#Factor 0 is how badly the students think of the school.

#%% PCA2

pca2 = PCA()
pca2.fit(zperform)
eigen2 = pca2.explained_variance_
loadings2 = pca2.components_
rotate2 = pca2.fit_transform(zperform)
covar2 = eigen2/sum(eigen2)*100


plt.plot(eigen2)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([0,3],[1,1],color='red',linewidth=1) # Kaiser criterion line

normalizedweight2 = eigen2/np.sum(eigen2)


#%% PCA 2 part 2

whichPrincipalComponent = 0 

plt.bar(np.linspace(1,3,3),np.transpose(loadings2)[:,whichPrincipalComponent])
plt.xlabel('Rating')
plt.ylabel('Loading')

#Component 0 is how well students performed overall.

#%% Corelating the factors.


PCAcorrelation = np.corrcoef(rotate[:,0],rotate2[:,0])

PCAcorrelation2 = np.corrcoef(rotate,rotate2,rowvar=False)

slope4, intercept4, r_value4, p_value4, std_err4 = stats.linregress(rotate[:,0],rotate2[:,0])

plt.plot(rotate[:,0],rotate2[:,0], "o")
plt.xlabel('How Negatively Students Think of the School ')
plt.ylabel('Overall Student Performance')


    
#%% Question 5

#High poverty vs. Non high poverty effects on acceptance rate.

poverty = alldata2[:,18]

float_pov = np.vstack(poverty).astype(np.float)

float_pov = np.transpose(float_pov)


com5 = np.vstack((arate,float_pov))
com5 = np.transpose(com5)
whereNAN = np.isnan(com5)
NANcoord = np.where(whereNAN==True)
com5 = np.delete(com5,NANcoord[0],0)

sort_pov = com5[com5[:,1].argsort()]

ttest = stats.ttest_ind(sort_pov[0:177][0],sort_pov[178:534][0])
print(ttest)

#This is not evidence, fail to reject null hypothesis.

#%% Question 6

#Acceptances are skewed, need to zscore the data first most likely.
#PLEASE NOTE THAT THIS DOES NOT INCLUDE CHARTER SCHOOLS, all charter
#school data on per_pupil_spending is removed. The analysis
#should not be not be interpreted to include charter schools.

resources = alldata["per_pupil_spending"]
resources2 = resources.to_numpy()

com6 = np.vstack((resources2,accp2))
com6 = np.transpose(com6)
whereNAN = np.isnan(com6)
NANcoord = np.where(whereNAN==True)
com6 = np.delete(com6,NANcoord[0],0)

zaccp = stats.zscore(com6[:,1])

correlation6 = np.corrcoef(com6[:,0],com6[:,1])


slope6, intercept6, r_value6, p_value6, std_err6 = stats.linregress(com6[:,0],com6[:,1])


plt.plot(com6[:,0],com6[:,1], "o")
plt.plot(com6[:,0], slope6*com6[:,0] + intercept6)
plt.xlabel('Per Pupil Spending')
plt.ylabel('Numer of Acceptances')
    

#%% Question 7

sort_accp = np.sort(accp)[::-1]

total_accp = np.sum(accp)


count = 0
schools = 0

for i in range(593):
    count += sort_accp[i]
    schools += 1
    if count>=(total_accp*0.9):
        print(i)
        print(count)
        print(schools)
        break

#123 schools out of 594

index = []
for i in range(594):
    index.append(i+1) 

index = np.array(index)

plt.bar(index,sort_accp, width=5)
plt.xlabel('Schools')
plt.ylabel('Accepted Students')
plt.show()


#%% Question 8

# 2c. Model: All factors

combined8 =  alldata2[:,2:21]
float8 = np.vstack(combined8).astype(np.float)

whereNAN = np.isnan(float8)
NANcoord = np.where(whereNAN==True)
float8 = np.delete(float8,NANcoord[0],0)

arate2 = np.empty([451,])

for i in range(451):
    aratec2 = float8[i][1]/float8[i][18]
    arate2[i] = aratec2    


accp3=float8[:,1]
 
zfloat8 = stats.zscore(float8, axis=0)
from sklearn import linear_model 

X = (zfloat8[:,2:19])
Y = arate2


regr = linear_model.LinearRegression() 
regr.fit(X,Y) 
rSqr = regr.score(X,Y) 
betas = regr.coef_ 
yInt = regr.intercept_  



y_hat = betas[0]*zfloat8[:,2] + betas[1]*zfloat8[:,3] + betas[2]*zfloat8[:,4] + betas[3]*zfloat8[:,5] + betas[4]*zfloat8[:,6] + betas[5]*zfloat8[:,7] + betas[6]*zfloat8[:,8] + betas[7]*zfloat8[:,9] + betas[8]*zfloat8[:,10] + betas[9]*zfloat8[:,11] + betas[10]*zfloat8[:,12] + betas[11]*zfloat8[:,13] + betas[12]*zfloat8[:,14] + betas[13]*zfloat8[:,15] + betas[14]*zfloat8[:,16] + betas[15]*zfloat8[:,17] + betas[16]*zfloat8[:,18] + yInt
plt.plot(regr.predict(X),arate2,'o',markersize=.75)
plt.xlabel('Model Prediction') 
plt.ylabel('Per Student chance of Acceptance')  
plt.title("Model for HSPHS Student Acceptance") 

#Credit for this code below goes to https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
import statsmodels.api as sm

X2 = sm.add_constant(X)
est = sm.OLS(Y, X2)
est2 = est.fit()
print(est2.summary())

#%% Question 8 part 2

combined82 =  alldata2[:,2:24]
float82 = np.vstack(combined82).astype(np.float)

whereNAN = np.isnan(float82)
NANcoord = np.where(whereNAN==True)
float82 = np.delete(float82,NANcoord[0],0)

zperform2 = stats.zscore(float82[:,[19,20,21]])

#Do a PCA on Objective measures of achievement again
pca3 = PCA()
pca3.fit(zperform2)
eigen3 = pca3.explained_variance_
loadings3 = pca3.components_
rotate3 = pca3.fit_transform(zperform2)
covar3 = eigen2/sum(eigen3)*100

#%% Eigenvalues

plt.plot(eigen3)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([0,3],[1,1],color='red',linewidth=1) # Kaiser criterion line

normalizedweight2 = eigen2/np.sum(eigen2)

#%% Loadings

whichPrincipalComponent = 0 # Try a few possibilities (at least 1,2,3)

plt.bar(np.linspace(1,3,3),np.transpose(loadings3)[:,whichPrincipalComponent])
plt.xlabel('Rating')
plt.ylabel('Loading')


#%% Rest of question
from sklearn import linear_model 

X2 = (float82[:,2:19])
Y2 = rotate3[:,0]

regr2 = linear_model.LinearRegression() 
regr2.fit(X2,Y2) 
rSqr2 = regr2.score(X2,Y2) 
betas2 = regr2.coef_ 
yInt2 = regr2.intercept_  


y_hat2 = betas2[0]*float82[:,2] + betas2[1]*float82[:,3] + betas2[2]*float82[:,4] + betas2[3]*float82[:,5] + betas2[4]*float82[:,6] + betas2[5]*float82[:,7] + betas2[6]*float82[:,8] + betas2[7]*float82[:,9] + betas2[8]*float82[:,10] + betas2[9]*float82[:,11] + betas2[10]*float82[:,12] + betas2[11]*float82[:,13] + betas2[12]*float82[:,14] + betas2[13]*float82[:,15] + betas2[14]*float82[:,16] + betas2[15]*float82[:,17] + betas2[16]*float82[:,18] + yInt
plt.plot(regr2.predict(X2),rotate3[:,0],'o',markersize=.75) # y_hat, income
plt.xlabel('Model prediction') 
plt.ylabel('Student Performance')  

X3 = sm.add_constant(X2)
est = sm.OLS(Y2, X3)
est2 = est.fit()
print(est2.summary())


