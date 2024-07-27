rm(list=ls(all=TRUE))
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

library(forecast)
library(ggplot2) 
library(dplyr)
library(lubridate)
library(reshape2)
library(feasts)
library(fpp3)
library(urca)
library(trend)
library(randtests)

set.seed(1)


data<-read.table(file="dataset/electricity.csv",header=TRUE,sep=";",dec=",")
data

str(data)
glimpse(data)

dts  <-ts(data[,5], start = c(2017,9),frequency =12)
dts

g<-autoplot(dts, xlab="year",ylab="Electricity Consumption (KWh)")+
  theme_bw() 
g

ggAcf(data[, 5])

ggPacf(data[, 5])



#################################ADF######################################

#H0: non-stationary
#Ha: stationary

args(ur.df)   
adf_test<-ur.df(y=dts,lags=5,type="trend",selectlags="AIC")
adf_test@testreg

adf_test@cval 
summary(adf_test)@teststat

# |-2.083| > |-3.45|, 
# Not reject H0, 
# the data has a unit root and is non-stationary


#################################KPSS######################################

args(ur.kpss)

kpss_test<-ur.kpss(dts,type="tau",lags="short")
summary(kpss_test) 

# |0.1613| > |0.146| 
# the data has a unit root and is non-stationary


############################Run###########################################

args(runs.test)

runs_test<-runs.test(dts)
runs_test

# |-5.6985| > |5%|
# the data is nonrandomness and has trend

############################Man-Kendall###################################

mankendall_test<-mk.test(dts)
mankendall_test

# |0.002437| < |5%|
# has no trend.

##############################Cox-Stuart##################################

cs_test<-cox.stuart.test(dts)
cs_test

# |0.009475| < |5%| 
# has no trend

############################Kruskal-Wallis#############################

kw_test<-kruskal.test(data$consumption ~ data$order)
kw_test

# |0.4787| < |5%|  
# has no seasonality

