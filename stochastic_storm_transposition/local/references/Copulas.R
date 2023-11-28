library(fitdistrplus)
library(stats)
library(copula)
library(ggplot2)

# set working directory
setwd("D:/Dropbox/_GradSchool/_norfolk/stormy/stochastic_storm_transposition/local/references")
f_data = "D:/Dropbox/_GradSchool/_norfolk/stormy/stochastic_storm_transposition/local/references/EasternCO_PDSI.csv"

# read in PDSI data
PDSI = read.csv(f_data)

# find all droughts, defined as consecutive periods below the mean
drought_indices = which(PDSI$reconPDSI < 0)

source("utils.R")
droughts = findClusters(drought_indices)

# find duration and severity of all droughts
droughtQualities = findDandS(droughts, 0-PDSI$reconPDSI)
durations = droughtQualities$durations
severities = droughtQualities$severities

hist(durations)
hist(severities)

# fit marginal distribution to severities
gamma.fit.S = fitdist(severities, "gamma", "mle")
plot(gamma.fit.S)
gamma2PPCC(severities, gamma.fit.S$estimate[1], gamma.fit.S$estimate[2])


poisson.fit.D = fitdist(durations, "pois", "mle")
plot(poisson.fit.D)
ks.test(durations, "ppois", poisson.fit.D$estimate[1], alternative = "two.sided")

geom.fit.D = fitdist(durations, "geom", "mle")
plot(geom.fit.D)
ks.test(durations, "pgeom", geom.fit.D$estimate[1], alternative = "two.sided")

# use empirical distribution because neither is very good


# convert durations and severities to uniform RVs (u_D and u_S) through their CDFs
# make scatter plot of u_D vs. u_S to postulate copulas to fit
u_D = rank(durations) / (length(durations) + 1)
u_S = pgamma(severities, gamma.fit.S$estimate[1], gamma.fit.S$estimate[2])
plot(u_D, u_S)
cor(u_D, u_S)

set.seed(9)
u_D = rank(durations, ties.method = "random") / (length(durations) + 1)
plot(u_D, u_S)
cor(u_D, u_S)

# create data frame of uniform random variables to fit copula to
u = matrix(c(u_D, u_S), nrow = length(u_D), byrow=FALSE)


# test goodnesss of fit of different copulas
?gofCopula
gofCopula(gumbelCopula(), u, N=1000, estim.method="ml")
gofCopula(joeCopula(), u, N=1000, estim.method="ml")
gofCopula(claytonCopula(), 1-u, N=1000, estim.method="ml")

# fit copula that fits best
?fitCopula
fitCopula(joeCopula(), u, method="ml")


# make density contour map
?contour
contour(joeCopula(2.49, dim=2), dCopula)
points(u_D, u_S)

# cumulative distribution
contour(joeCopula(2.49, dim=2), pCopula)
points(u_D, u_S)


# repeat with other copulas on your own
contour(claytonCopula(1.72, dim=2), dCopula)
points(1-u_D, 1-u_S)

contour(gumbelCopula(1.606, dim=2), dCopula)
points(u_D, u_S)


# compute return period of most severe event using copula that fits best
# 1) using OR condition
1 / (1 - pCopula(u[which.max(u[,2]),], joeCopula(2.49, dim=2)))
# 2) using AND condition
1 / (1 - u_D[which.max(u[,2])] - u_S[which.max(u[,2])]
      + pCopula(u[which.max(u[,2]),], joeCopula(2.49, dim=2)))
# 3) using Kendall's return period
?K
joeCop = setTheta(copJoe, 2.49)
1 / (1 - pK(pCopula(u[which.max(u[,2]),], joeCopula(2.49, dim=2)), joeCop, d=2))

# find the conditional distribution of drought severity 
# given it lasts 3 years using the Joe copula
?cCopula
sortedD = sort(durations)
u1 = mean(which(sortedD==3)) / (length(sortedD) + 1)
U = cCopula(cbind(u1, runif(1000)), copula=joeCopula(2.49, d=2), inverse=TRUE)

# convert percentiles of U to severities
s_cond = qgamma(U[,2], gamma.fit.S$estimate[1], gamma.fit.S$estimate[2])

# find Kernel density estimate from generated severities
kde = density(s_cond) # in stats library

# compare with the unconditional distribution of drought severity
x = c(seq(0,15,0.1))
pdf_uncond = dgamma(x, gamma.fit.S$estimate[1], gamma.fit.S$estimate[2])

ggplot() + geom_line(aes(x=x, y=pdf_uncond, color="Unconditional"), lwd=1.5) +
  geom_line(aes(x=kde$x, y=kde$y, color="Given D=3"), lwd=1.5) + 
  xlab("Drought Severity") + ylab("Density") + 
  theme(legend.title = element_blank())

# find conditional and unconditional probability severity >= 5 given D = 3
pCond_5 = length(s_cond[s_cond>5]) / length(s_cond)
pUncond_5 = 1 - pgamma(5, gamma.fit.S$estimate[1], gamma.fit.S$estimate[2])

# find conditional and unconditional probability severity >= 10 given D = 3
pCond_10 = length(s_cond[s_cond>10]) / length(s_cond)
pUncond_10 = 1 - pgamma(10, gamma.fit.S$estimate[1], gamma.fit.S$estimate[2])


# generate 1000 drought durations and severities from the Joe copula
U = rCopula(1000, joeCopula(2.49, d=2))
Dsim = sapply(1:1000, function(i) durations[which.min(abs(U[i,1]-u_D))])
Ssim = qgamma(U[,2], gamma.fit.S$estimate[1], gamma.fit.S$estimate[2])

# plot simulated droughts and severities and observed droughts and severities
ggplot() + geom_point(aes(Dsim, Ssim, color="Simulated"), lwd=1.5) + 
  geom_point(aes(durations, severities, color="Observed"), lwd=1.5) + 
  xlab("Drought Duration") + ylab("Drought Severity") + 
  theme(legend.title = element_blank())
