#1
x1<-rnorm(10,0,1)
mean(x1)
var(x1)
sd(x1)

x2<-rnorm(10000,0,1)
mean(x2)
var(x2)
sd(x2)

x3<-rnorm(1000000,0,1)
mean(x3)
var(x3)
sd(x3)

#2

y<-read.table('2.txt',sep = ',')
y<-as.numeric(y)
x<-read.table('1.txt',sep = ',')
x<-as.numeric(x)
fit<-lm(y~x)
summary(fit)

#3
csco<-read.csv('csco.csv')
nasdaq<-read.csv('nasdaq-100.csv')
CSCO<-as.numeric(t(csco[5]))
NASDAQ<-as.numeric(t(nasdaq[5]))
len<-length(CSCO)-1
A<-rep(NA,len)
for (i in 1:len) {A[i]=log(CSCO[i]/CSCO[i+1])}
B<-rep(NA,len)
for (i in 1:len) {B[i]=log(NASDAQ[i]/NASDAQ[i+1])}
hist(A,breaks = 150,main='Histogram of Log Returns')
plot(A~B,main='Scatterplot of Log Returns')
fitAB<-lm(A~B)
confint(fitAB)

#4
require(foreign)
train<-read.dta('train.dta')
summary(train)
table(train[,1])
(model<-lm(d~x1,data=train))
predict(model,data.frame(x1=0.65))
predict(model,data.frame(x1=0.99))

#5

x<-rnorm(1000)
e<-rnorm(1000)
y<-1+2*x+e
summary(lm(y~x))

x<-rnorm(1000)
e<-rnorm(1000)
y<-1+2*x+e
summary(lm(y~x))
x<-rnorm(1000)
e<-rnorm(1000)
y<-1+2*x+e
summary(lm(y~x))
x<-rnorm(1000)
e<-rnorm(1000)
y<-1+2*x+e
summary(lm(y~x))
x<-rnorm(1000)
e<-rnorm(1000)
y<-1+2*x+e
summary(lm(y~x))
x<-rnorm(1000)
e<-rnorm(1000)
y<-1+2*x+e
summary(lm(y~x))

b<-rep(NA,1000)
for (i in 1:1000)
{x<-rnorm(1000)
 e<-rnorm(1000)
 y<-1+2*x+e
 m<-lm(y~x)
 b[i]<-coef(m)[2]}
hist(b,breaks = 70)
summary(b)
bb<-exp(b)
hist(bb,breaks = 70)
summary(bb)
