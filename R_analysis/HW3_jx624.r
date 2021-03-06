##################1#######################################
library("foreign")
griliche<-read.dta("griliches.dta")
head(griliche)
str(griliche)
summary(griliche)
attach(griliche)
summary(rns)
table(rns)
summary(mrt)
table(mrt)
summary(smsa)
table(smsa)
summary(iq)
summary(kww)
summary(age)
summary(s)
summary(expr)
summary(lw)
par(mfrow=c(3,2))
plot(lw~rns)
plot(lw~mrt)
plot(lw~smsa)
plot(lw~kww)
plot(lw~expr)
plot(lw~rns)
lm_rns<-lm(lw~rns)
abline(lm_rns)
plot(lw~mrt)
lm_mrt<-lm(lw~mrt)
abline(lm_mrt)
plot(lw~smsa)
lm_smsa<-lm(lw~smsa)
abline(lm_smsa)
plot(lw~kww)
lm_kww<-lm(lw~kww)
abline(lm_kww)
plot(lw~expr)
lm_expr<-lm(lw~expr)
par(mfrow=c(1,1))
plot(lw~s)
lm_s<-lm(lw~s)
abline(lm_s)
confint(lm_s)
lm_multi<-lm(lw~rns+mrt+smsa+med+iq+kww+age+s+expr)
confint(lm_multi)
griliche$age2<-age^2
lm_multi_2<-lm(lw~rns+mrt+smsa+med+iq+kww+age+s+expr+age2,data = griliche)
lm_multi_2
confint(lm_multi_2)
########2#################################################################
x1<-rnorm(10000)
x2<-rnorm(10000)
e<-rnorm(10000)
y<-1+x1+x2+e
fit<-lm(y~x1)
fit
z<-rnorm(10000)
v<-rnorm(10000)
w<-rnorm(10000)
x11<-z+v
x22<-w-z
y<-1+x11+x22+e
fit2<-lm(y~x11)
fit2
###################################3######################################
require(foreign)
union<-read.dta("union.dta")
head(union)
train<-union[union$year>=70&union$year<=78,]
fit1<-lm(union~year+age+grade+south+black+smsa,data=train)
fit2<-glm(union~year+age+grade+south+black+smsa,family=binomial,data=train)
test<-union[union$year>=80&union$year<=88,]
outcome1<-predict(fit1,test)
outcome2<-predict(fit2,test,'response')
length(outcome1[outcome1>0.25])
length(outcome1[outcome1<0.25])
length(outcome2[outcome2>0.25])
length(outcome2[outcome2<0.25])
get01<-function(x){if (x>0.2){x=1}else{x=0}}
count1<-as.matrix(outcome1)
count1<-apply(count1,c(1,2),get01)
count2<-as.matrix(outcome2)
count2<-apply(count2,c(1,2),get01)
table(count1,test$union)
table(count2,test$union)
