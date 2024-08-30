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


data<-read.table(file="electricity.csv",header=TRUE,sep=";",dec=",")
data

str(data)
glimpse(data)

dts  <-ts(data[,4], start = c(2017,9),frequency =12)
dts

g<-autoplot(dts, xlab="year",ylab="Electricity Consumption (KWh)")+
  theme_bw() 
g

ggAcf(data[, 4])

ggPacf(data[, 4])


#################################KPSS######################################

args(ur.kpss)

kpss_test<-ur.kpss(dts,type="tau",lags="short")
summary(kpss_test) 

# |0.1613| > |0.146| 
# the data has a unit root and is non-stationary


############################Man-Kendall###################################

mankendall_test<-mk.test(dts)
mankendall_test

# |0.002437| < |5%|
# has no trend.


############################Kruskal-Wallis#############################

kw_test<-kruskal.test(data$consumption ~ data$order)
kw_test

# |0.4787| < |5%|  
# has no seasonality

