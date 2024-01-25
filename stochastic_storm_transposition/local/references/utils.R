library(e1071)
library(ppcc)
library(FAdist)
library(pracma)

# function to calculate central empirical moments
memp.centered  =  function(x, order){
  if(order == 1){
    return (mean(x))
  } else if(order == 2){
    return (var(x))
  } else if(order == 3){
    return (skewness(x, type=2))
  }
}


# calculate analytical confidence intervals for normal distribution
calcNormCI = function(data, p, CI){
  n = length(data)
  xbar = mean(data)
  sigma = sd(data)
  sigmaSq = var(data)
  
  z_p = qnorm(p)
  z_crit = qnorm(1-(100-CI)/200)
  
  LB = xbar + z_p*sigma - z_crit * sqrt(sigmaSq * (1+0.5*z_p^2)/n)
  UB = xbar + z_p*sigma + z_crit * sqrt(sigmaSq * (1+0.5*z_p^2)/n)
  
  return(list(UB=UB,LB=LB))
}


# perform PPCC for log-normal distribution
lnormPPCC = function(data, meanlog, sdlog){
  # compute theoretical quantiles
  n = length(data)
  p = ppPositions(n,"Blom")
  q = qlnorm(p, meanlog, sdlog)
  
  # compute correlation coefficient between theoretical and empirical quantiles
  rho = cor(q, sort(data))
  
  # run 1000 MC simulations and compute correlation
  rhos = c(rep(NA,1000))
  for(i in 1:length(rhos)){
    # generate pseudo observations
    pobs = rlnorm(length(data), meanlog, sdlog)
    rhos[i] = cor(q, sort(pobs))
  }
  
  # compute p-value from MC simulation
  pval = sum(rhos<rho) / length(rhos)
  
  if(pval>0.05){
    conclusion="Do not reject that the data is LN2-distributed"
  } else {
    conclusion="Reject that the data is LN2-distributed"
  }
  
  return(list(rho=rho,pval=pval,conclusion=conclusion))
}


# function to calculate parameters of lnorm3 distribution using MOM
lnorm3MOM = function(data){
  sigma = uniroot(function (x) (exp(3*x^2) - 3*exp(x^2)+2) / 
                    (exp(x^2)-1)^(3/2) - skewness(data,type=2), 
                  c(0.01, sd(log(data))))$root
  mu = 0.5 * ( log( var(data) / (exp(sigma^2)-1) ) - sigma^2)
  tau = mean(data) - exp(mu + 0.5*sigma^2)
  return(list(sigma=sigma, mu=mu, tau=tau))
}

# function to calculate central theoretical moments of LN3 distribution
mlnorm3 = function(order, shape, scale, thres){
  mu = scale
  sigma = shape
  tau = thres
  if(order == 1){
    return (tau + exp(mu + 0.5*sigma^2))
  } else if(order == 2){
    return ( (exp(sigma^2)-1)*exp(2*mu + sigma^2) )
  } else if(order == 3){
    return ( (exp(3*sigma^2)-3*exp(sigma^2)+2) / (exp(sigma^2)-1)^(3/2) )
  }
}

# perform PPCC for LN3 distribution
lnorm3PPCC = function(data, shape, scale, thres){
  # compute theoretical quantiles
  n = length(data)
  p = ppPositions(n,"Blom")
  q = qlnorm3(p, shape=shape, scale=scale, thres=thres)
  
  # compute correlation coefficient between theoretical and empirical quantiles
  rho = cor(q, sort(data))
  
  # run 1000 MC simulations and compute correlation
  rhos = c(rep(NA,1000))
  for(i in 1:length(rhos)){
    # generate pseudo observations
    pobs = rlnorm3(length(data), shape=shape, scale=scale, thres=thres)
    rhos[i] = cor(q, sort(pobs))
  }
  
  # compute p-value from MC simulation
  pval = sum(rhos<rho) / length(rhos)
  
  if(pval>0.05){
    conclusion="Do not reject that the data is LN3-distributed"
  } else {
    conclusion="Reject that the data is LN3-distributed"
  }
  
  return(list(rho=rho,pval=pval,conclusion=conclusion))
}


# function to calculate parameters of P3 distribution using MOM
gamma2MOM = function(data){
  alpha = mean(data)^2 / var(data)
  beta = mean(data) / var(data)
  return(list(alpha=alpha, beta=beta))
}

# function to calculate central theoretical moments of Gamma distribution
mgamma2 = function(order, shape, rate){
  alpha = shape
  beta = rate
  if(order==1){
    return (alpha/beta)
  } else if(order==2){
    return (alpha/beta^2)
  }
}

# perform PPCC for gamma2 distribution
gamma2PPCC = function(data, shape, rate){
  # compute theoretical quantiles
  n = length(data)
  p = ppPositions(n,"Blom")
  q = qgamma(p, shape=shape, rate=rate)
  
  # compute correlation coefficient between theoretical and empirical quantiles
  rho = cor(q, sort(data))
  
  # run 1000 MC simulations and compute correlation
  rhos = c(rep(NA,1000))
  for(i in 1:length(rhos)){
    # generate pseudo observations
    pobs = rgamma(length(data), shape=shape, rate=rate)
    rhos[i] = cor(q, sort(pobs))
  }
  
  # compute p-value from MC simulation
  pval = sum(rhos<rho) / length(rhos)
  
  if(pval>0.05){
    conclusion="Do not reject that the data is Gamma-distributed"
  } else {
    conclusion="Reject that the data is Gamma-distributed"
  }
  
  return(list(rho=rho,pval=pval,conclusion=conclusion))
}


# function to calculate parameters of P3 distribution using MOM
gamma3MOM = function(data){
  alpha = 4/(skewness(data,type=2))^2
  beta = sqrt(alpha) / sd(data)
  xi = mean(data) - alpha/beta
  return(list(alpha=alpha, beta=beta, xi=xi))
}

# function to calculate central theoretical moments of P3 distribution
mgamma3 = function(order, shape, scale, thres){
  xi = thres
  alpha = shape
  beta = 1/scale
  if(order == 1){
    return (xi + alpha/beta)
  } else if(order == 2){
    return ( alpha/beta^2 )
  } else if(order == 3){
    return ( 2/sqrt(alpha) )
  }
}

# perform PPCC for gamma3 distribution
gamma3PPCC = function(data, shape, scale, thres){
  # compute theoretical quantiles
  n = length(data)
  p = ppPositions(n,"Blom")
  q = qgamma3(p, shape=shape, scale=scale, thres=thres)
  
  # compute correlation coefficient between theoretical and empirical quantiles
  rho = cor(q, sort(data))
  
  # run 1000 MC simulations and compute correlation
  rhos = c(rep(NA,1000))
  for(i in 1:length(rhos)){
    # generate pseudo observations
    pobs = rgamma3(length(data), shape=shape, scale=scale, thres=thres)
    rhos[i] = cor(q, sort(pobs))
  }
  
  # compute p-value from MC simulation
  pval = sum(rhos<rho) / length(rhos)
  
  if(pval>0.05){
    conclusion="Do not reject that the data is P3-distributed"
  } else {
    conclusion="Reject that the data is P3-distributed"
  }
  
  return(list(rho=rho,pval=pval,conclusion=conclusion))
}

# perform PPCC for LP3 distribution
LP3PPCC = function(data, shape, scale, thres){
  # compute theoretical quantiles
  n = length(data)
  p = ppPositions(n,"Blom")
  if (skewness(log(data),type=2)>0){
    q = exp(qgamma3(p, shape=shape, scale=scale, thres=thres))
  } else{
    q = exp(-qgamma3(1-p, shape=shape, scale=scale, thres=thres))
  }
  
  # compute correlation coefficient between theoretical and empirical quantiles
  rho = cor(q, sort(data))
  
  # run 1000 MC simulations and compute correlation
  rhos = c(rep(NA,1000))
  for(i in 1:length(rhos)){
    # generate pseudo observations
    if (skewness(data,type=2)>0){
      pobs = exp(rgamma3(length(data), shape=shape, scale=scale, thres=thres))
    } else{
      pobs = exp(-rgamma3(length(data), shape=shape, scale=scale, thres=thres)) 
    }
    rhos[i] = cor(q, sort(pobs))
  }
  
  # compute p-value from MC simulation
  pval = sum(rhos<rho) / length(rhos)
  
  if(pval>0.05){
    conclusion="Do not reject that the data is LP3-distributed"
  } else {
    conclusion="Reject that the data is LP3-distributed"
  }
  
  return(list(rho=rho,pval=pval,conclusion=conclusion))
}


# function to calculate parameters of Gumbel distribution using MOM
gumbelMOM = function(data){
  alpha = sqrt(6*var(data)) / pi
  xi = mean(data) - 0.5772*alpha
  return(list(alpha=alpha, xi=xi))
}

# function to calculate central theoretical moments of Gumbel distribution
mgumbel = function(order, scale, location){
  xi = location
  alpha = scale
  if(order == 1){
    return (xi + 0.5772*alpha)
  } else if(order == 2){
    return ( pi^2 * alpha^2/6 )
  }
}


# function to calculate parameters of GEV distribution using MOM
gevMOM = function(data){
  kappa = uniroot(function (x) mgev(3, x, 1, 0) - 
                    memp.centered(data,3),
                  c(-1.0, 0.32))$root
  alpha = sqrt(var(data) * (-kappa)^2 / 
    (gamma(1-2*kappa) - (gamma(1-kappa))^2))
  xi = mean(data) + (alpha/kappa) * (1 - gamma(1-kappa))
  return(list(kappa=kappa, alpha=alpha, xi=xi))
}

# function to calculate central theoretical moments of GEV distribution
mgev = function(order, shape, scale, location){
  xi = location
  alpha = scale
  kappa = shape
  if(order == 1){
    return ( xi + (alpha/-kappa) * (1-gamma(1-kappa)) )
  } else if(order == 2){
    return ( (alpha/-kappa)^2 * (gamma(1-2*kappa) - (gamma(1-kappa))^2) )
  } else if(order == 3){
    return ( sign(-kappa) * (-gamma(1-3*kappa) + 3*gamma(1-kappa)*gamma(1-2*kappa)
                            -2*(gamma(1-kappa))^3) /
               (gamma(1-2*kappa) - (gamma(1-kappa))^2)^(3/2) )
  }
}

# perform PPCC for gev distribution
gevPPCC = function(data, shape, scale, location){
  # compute theoretical quantiles
  n = length(data)
  p = ppPositions(n,"Cunane")
  q = qgev(p, shape=shape, scale=scale, location=location)
  
  # compute correlation coefficient between theoretical and empirical quantiles
  rho = cor(q, sort(data))
  
  # run 1000 MC simulations and compute correlation
  rhos = c(rep(NA,1000))
  for(i in 1:length(rhos)){
    # generate pseudo observations
    pobs = rgev(length(data), shape=shape, scale=scale, location=location)
    rhos[i] = cor(q, sort(pobs))
  }
  
  # compute p-value from MC simulation
  pval = sum(rhos<rho) / length(rhos)
  
  if(pval>0.05){
    conclusion="Do not reject that the data is GEV-distributed"
  } else {
    conclusion="Reject that the data is GEV-distributed"
  }
  
  return(list(rho=rho,pval=pval,conclusion=conclusion))
}

# calculate MOM estimators for Generalized Pareto distribution
gpMOM = function(data){
  kappa = 0.5*((mean(data) / sd(data))^2 - 1)
  alpha = (1+kappa)*(mean(data))
  return(list(alpha=alpha,kappa=kappa))
}

# calculate theoretical moments of Generalized Pareto distribtion
mgp = function(order, shape, scale){
  kappa = shape
  alpha = scale
  if(order == 1){
    return(alpha/(1+kappa))
  } else if(order ==2){
    return(alpha^2 / ((1+kappa)^2 * (1+2*kappa)) )
  }
}

# convert GPD params for POT to GEV params for AMS
GPDtoGEV = function(gp.fit, x0, lambda){
  kappa = gp.fit$estimate[1]
  alpha = gp.fit$estimate[2]
  if(kappa != 0){
    xi = x0 + alpha*(1-lambda^(-kappa))/kappa
  } else {
    xi = x0 + alpha*log(lambda)
  }
  
  alpha_star = alpha*lambda^(-kappa)
  return(list(kappa=-kappa, alpha=alpha_star, xi=xi))
}

# perform PPCC for Generalized Pareto distribution
gpPPCC = function(data, shape, scale){
  # compute theoretical quantiles
  n = length(data)
  p = ppPositions(n,"Gringorton")
  q = qgp(p, shape=shape, scale=scale)
  
  # compute correlation coefficient between theoretical and empirical quantiles
  rho = cor(q, sort(data))
  
  # run 1000 MC simulations and compute correlation
  rhos = c(rep(NA,1000))
  for(i in 1:length(rhos)){
    # generate pseudo observations
    pobs = rgp(length(data), shape=shape, scale=scale)
    rhos[i] = cor(q, sort(pobs))
  }
  
  # compute p-value from MC simulation
  pval = sum(rhos<rho) / length(rhos)
  
  if(pval>0.05){
    conclusion="Do not reject that the data is GP-distributed"
  } else {
    conclusion="Reject that the data is GP-distributed"
  }
  
  return(list(rho=rho,pval=pval,conclusion=conclusion))
}


# calculate MOM estimators for 2-parameter Weibull distribution
Weibull2MOM = function(data){
  kappa = uniroot(function(x) mean(data)^2 * (gamma(1+2/x) / gamma(1+1/x)^2 -1)
                  - var(data), c(0.02, 10))$root
  alpha = mean(data) / gamma(1+1/kappa)
  return(list(alpha=alpha,kappa=kappa))
}

# calculate theoretical moments of 2-parameter Weibull distribtion
mweibull = function(order, shape, scale){
  alpha = scale
  kappa = shape
  if(order == 1){
    return(alpha*gamma(1+1/kappa))
  } else if(order == 2){
    return(alpha^2 * (gamma(1+2/kappa) - (gamma(1+1/kappa))^2))
  }
}


# calculate MOM estimators for 3-parameter Weibull distribution
Weibull3MOM = function(data){
  F = function(x){
    kappa = x[1]; alpha = x[2]; xi = x[3]
    
    m = alpha*gamma(1+1/kappa) + xi
    s = sqrt(alpha^2 * (gamma(1+2/kappa) - (gamma(1+1/kappa))^2))
    g = (gamma(1+3/kappa) * alpha^3 - 3*m*s^2 - m^3) / s^3
    
    as.matrix(c(m-mean(data),
                s-sd(data),
                g-skewness(data,type=2)), ncol=1)
  }
  # initialize parameter estimates, x0, at 2-parameter Weibull estimates
  weibull2.params = Weibull2MOM(data)
  x0 = as.matrix(c(weibull2.params$kappa, weibull2.params$alpha, 0))
  soln = broyden(F, x0)

  kappa = soln$zero[1]
  alpha = soln$zero[2]
  xi = soln$zero[3]
  
  return(list(alpha=alpha,kappa=kappa,xi=xi))
}

# calculate theoretical moments of 3-parameter Weibull distribtion
mweibull3 = function(order, shape, scale, thres){
  alpha = scale
  kappa = shape
  xi = thres
  
  m = alpha*gamma(1+1/kappa) + xi
  s = sqrt(alpha^2 * (gamma(1+2/kappa) - (gamma(1+1/kappa))^2))
  g = (gamma(1+3/kappa) * alpha^3 - 3*m*s^2 - m^3) / s^3
    
  if(order == 1){
    return(m)
  } else if(order == 2){
    return(s^2)
  } else if(order == 3){
    return(g)
  }
}


# perform PPCC for 2-parameter Weibull distribution
weibull2PPCC = function(data, shape, scale){
  # compute theoretical quantiles
  n = length(data)
  p = ppPositions(n,"Gringorton")
  q = qweibull(p, shape=shape, scale=scale)
  
  # compute correlation coefficient between theoretical and empirical quantiles
  rho = cor(q, sort(data))
  
  # run 1000 MC simulations and compute correlation
  rhos = c(rep(NA,1000))
  for(i in 1:length(rhos)){
    # generate pseudo observations
    pobs = rweibull(length(data), shape=shape, scale=scale)
    rhos[i] = cor(q, sort(pobs))
  }
  
  # compute p-value from MC simulation
  pval = sum(rhos<rho) / length(rhos)
  
  if(pval>0.05){
    conclusion="Do not reject that the data is Weibull-distributed"
  } else {
    conclusion="Reject that the data is Weibull-distributed"
  }
  
  return(list(rho=rho,pval=pval,conclusion=conclusion))
}


# perform PPCC for 3-parameter Weibull distribution
weibull3PPCC = function(data, shape, scale, thres){
  # compute theoretical quantiles
  n = length(data)
  p = ppPositions(n,"Gringorton")
  q = qweibull3(p, shape=shape, scale=scale, thres=thres)
  
  # compute correlation coefficient between theoretical and empirical quantiles
  rho = cor(q, sort(data))
  
  # run 1000 MC simulations and compute correlation
  rhos = c(rep(NA,1000))
  for(i in 1:length(rhos)){
    # generate pseudo observations
    pobs = rweibull3(length(data), shape=shape, scale=scale, thres=thes)
    rhos[i] = cor(q, sort(pobs))
  }
  
  # compute p-value from MC simulation
  pval = sum(rhos<rho) / length(rhos)
  
  if(pval>0.05){
    conclusion="Do not reject that the data is Weibull3-distributed"
  } else {
    conclusion="Reject that the data is Weibull3-distributed"
  }
  
  return(list(rho=rho,pval=pval,conclusion=conclusion))
}


# calculate analytical confidence intervals for Gumbel distribution quantiles
calcGumbelCI = function(data, alpha, p, q_p, CI){
  # p is the percentile
  # alpha is the scale parameter of the Gumbel distribution
  # z_crit is z_{1-a/2} for a (100-a)% CI
  n = length(data)
  y = -log(-log(p))
  z_crit = qnorm(1-(100-CI)/200)
  LB = q_p - z_crit * sqrt(alpha^2*(1.11+0.52*y+0.61*y^2)/n)
  UB = q_p + z_crit * sqrt(alpha^2*(1.11+0.52*y+0.61*y^2)/n)
  
  return(list(LB=LB,UB=UB))
}


# calculate analytical confidence intervals for log-normal distribution
calcLN2CI = function(data, p, CI){
  n = length(data)
  ybar = mean(log(data))
  sigma = sd(log(data))
  sigmaSq = var(log(data))
  
  z_p = qnorm(p)
  z_crit = qnorm(1-(100-CI)/200)
  
  LB = exp(ybar + z_p*sigma - z_crit * sqrt(sigmaSq * (1+0.5*z_p^2)/n))
  UB = exp(ybar + z_p*sigma + z_crit * sqrt(sigmaSq * (1+0.5*z_p^2)/n))
  
  return(list(UB=UB,LB=LB))
}


# calculate analytical confidence intervals for LP3 distribution quantiles
calcLP3CI = function(y, p, zeta_LB_p, zeta_UB_p){
  # y is log(data)
  # z_p is the quantile of the standard normal corresponding to the pth percentile
  # zeta_LB_p and zeta_UB_p are the lower and upper quantiles corresponding to the
  # pth percentile of the non-central t distribution
  ybar = mean(y)
  s_y = sd(y)
  gamma = skewness(y, type=2)
  n = length(y)
  
  z_p = qnorm(p)
  
  VarG = 6*(n-2) / ((n+1)*(n+3))
  Kp = (2/gamma) * (1 + gamma*z_p/6 - gamma^2/36)^3 - (2/gamma)
  derivative = (1/6) * (z_p^2 - 1) * (1-3*(gamma/6)^2) + 
    (z_p^3 - 6*z_p)*gamma/54 + (2/3)*z_p*(gamma/6)^3
  eta = sqrt((1+gamma*Kp + 0.5*(1+0.75*gamma^2)*Kp^2 + n*VarG*derivative^2) / 
               (1+0.5*z_p^2))
  y_p = ybar + Kp*s_y
  
  LB = exp(y_p + eta*(zeta_LB_p - z_p)*s_y)
  UB = exp(y_p + eta*(zeta_UB_p - z_p)*s_y)
  
  return(list(LB=LB,UB=UB))
}


# Expected Moments Algorithm
EMA_LP3 = function(x, y, s, h, T, tolerance, maxiter){
  # x = log-space systematic flows
  # y = log-space historical flows
  # s = length of systematic record
  # h = length of historical record
  # T = log-space threshold
  # tolerance for convergence (total % difference in 3 parameter estimates)
  # maxiter = maximum number of iterations if convergence tolerance never met
  
  # create a vector (y_prime) of all of the floods above the threshold (T)
  # in both the historical (y) and systematic (x) records
  y_prime = c(y[which(y>T)], x[which(x>T)])
  k = length(y[which(y>T)]) # number of historical floods over the threshold
  
  # get a vector of all observed peaks below the threshold (those during the systematic record)
  x_prime = x[which(x<=T)]
  m = s - length(x_prime) # number of peaks in systematiic record that exceeded T
  
  # Step 1: estimate parameters alpha_hat, beta_hat, xi_hat with MOM using only systematic record
  LP3.params = gamma3MOM(x)
  alpha_hat = LP3.params$alpha
  beta_hat = LP3.params$beta
  xi_hat = LP3.params$xi
  
  # Step 2: update estimate of sample moments
  # calculate constants c2 and c3
  c2 = (s+k) / (s+k-1)
  c3 = (s+k)^2 / ((s+k-1)*(s+k-2))
  # iterate till convergence or max number of iterations
  for(i in 1:maxiter){
    # find expected value below the threshold with current parameter estimates
    exp_gamma = function(x) (x*dgamma3(x, alpha_hat, 1/beta_hat, xi_hat))
    integral = integrate(exp_gamma, lower=-Inf, upper=T, subdivisions=10000)
    E_xH_below = integral[[1]] / (pgamma3(T, alpha_hat, 1/beta_hat, xi_hat))
    
    # update estimate of mean
    mu_new = (sum(x_prime) + sum(y_prime) + (h-k)*E_xH_below) / (s+h)
    
    # find expected value of x^2 below the threshold with latest parameter estimates
    exp2_gamma = function(x) ((x-mu_new)^2 *
                               dgamma3(x, alpha_hat, 1/beta_hat, xi_hat))
    integral2 = integrate(exp2_gamma, lower=-Inf, upper=T, subdivisions=10000)
    E_x2H_below = integral2[[1]] / (pgamma3(T, alpha_hat, 1/beta_hat, xi_hat))
    
    # update estimate of standard deviation
    sigma_new = sqrt((c2*sum((x_prime-mu_new)^2) + sum((y_prime-mu_new)^2) + 
                       (h-k)*E_x2H_below) / (s+h))
    
    # find expected value of x^3 below the threshold with latest parameter estimates
    exp3_gamma = function(x) ((x-mu_new)^3 *
                                dgamma3(x, alpha_hat, 1/beta_hat, xi_hat))
    integral3 = integrate(exp3_gamma, lower=-Inf, upper=T, subdivisions=10000)
    E_x3H_below = integral3[[1]] / (pgamma3(T, alpha_hat, 1/beta_hat, xi_hat))
    
    # update estimate of skewness
    gamma_new = ((c3*(sum((x_prime-mu_new)^3) + sum((y_prime-mu_new)^3))
                  + (h-k)*E_x3H_below)) / ((s+h)*sigma_new^3)
    
    # update estimates LP3 parameters
    alpha_new = 4/(gamma_new)^2
    beta_new = sqrt(alpha_new) / sigma_new
    xi_new = mu_new - alpha_new/beta_new
    
    # calculate difference betwen old ("hat") and "new" estimates
    alphaDiff = 0.5 * abs(alpha_new - alpha_hat) / abs(alpha_new + alpha_hat)
    betaDiff = 0.5 * abs(beta_new - beta_hat) / abs(beta_new + beta_hat)
    xiDiff = 0.5 * abs(xi_new - xi_hat) / abs(xi_new + xi_hat)
    
    totalDiff = alphaDiff + betaDiff + xiDiff
    
    # update old ("hat") estimates with "new" estimates
    alpha_hat = alpha_new
    beta_hat = beta_new
    xi_hat = xi_new
    
    # Step 3: convergence test
    # exit loop if total difference between past and current parameter estimates is within tolerance
    if (totalDiff < tolerance) break
  }
  # return parameter estimates
  return(list(alpha=alpha_hat, beta=beta_hat, xi=xi_hat))
}

# Adjusted Moments Algorithm
AMA_LP3 = function(x, y, s, h, T){
  # x = log-space systematic flows
  # y = log-space historical flows
  # s = length of systematic record
  # h = length of historical record
  # T = log-space threshold
  
  # create a vector (y_prime) of all of the floods above the threshold (T)
  # in both the historical (y) and systematic (x) records
  y_prime = c(y[which(y>T)], x[which(x>T)])
  k = length(y[which(y>T)]) # number of historical floods over the threshold
  
  # get a vector of all observed peaks below the threshold (those during the systematic record)
  x_prime = x[which(x<=T)]
  m = s - length(x_prime) # number of peaks in systematic record that exceeded T
  
  # Use AMA to estimate T-year event
  mu_AMA = (sum(y_prime) + (1+(h-k)/(s-m))*sum(x_prime)) / (s+h)
  sigma_AMA = sqrt((sum((y_prime-mu_AMA)^2) + (1+(h-k)/(s-m))*
                     sum((x_prime-mu_AMA)^2)) / (s+h-1))
  gamma_AMA = (s+h) * (sum((y_prime-mu_AMA)^3) + (1+(h-k)/(s-m))*
                             sum((x_prime-mu_AMA)^3)) /
              ((s+h-1) * (s+h-2) * sigma_AMA^3)
  
  alpha = 4/(gamma_AMA)^2
  beta = sqrt(alpha) / sigma_AMA
  xi = mu_AMA - alpha/beta
  
  # return parameter estimates
  return(list(alpha=alpha, beta=beta, xi=xi))
}


# perform PPCC for Chi squared distribution
chisqPPCC = function(data, nu){
  # compute theoretical quantiles
  n = length(data)
  p = ppPositions(n,"Blom")
  q = qchisq(p, nu)
  
  # compute correlation coefficient between theoretical and empirical quantiles
  rho = cor(q, sort(data))
  
  plot(q,sort(data), xlab="Theoretical Quantiles", ylab="Empirical Quantiles",
       main="Chi Squared PPCC")
  lines(q,q,col="red")
  
  # run 1000 MC simulations and compute correlation
  rhos = c(rep(NA,1000))
  for(i in 1:length(rhos)){
    # generate pseudo observations
    pobs = rchisq(length(data), nu)
    rhos[i] = cor(q, sort(pobs))
  }
  
  # compute p-value from MC simulation
  pval = sum(rhos<rho) / length(rhos)
  
  if(pval>0.05){
    conclusion="Do not reject that the data is Chi Squared-distributed"
  } else {
    conclusion="Reject that the data is Chi Squared-distributed"
  }
  
  return(list(rho=rho,pval=pval,conclusion=conclusion))
}


# find clusters of consecutive periods with the same condition 
# (e.g. below/above a threshold) based on indices where that condition is met
findClusters = function(indices){
  allClusters = list()
  subCluster = list(indices[1]) # current cluster
  for(i in 2:length(indices)){
    if(indices[i] - subCluster[[length(subCluster)]] <= 1){ # same cluster
      # append index to current cluster
      subCluster[[length(subCluster)+1]] = indices[i]
    } else { # new cluster
      # append current cluster to all clusters
      allClusters[[length(allClusters)+1]] = subCluster 
      subCluster = list(indices[i]) # reset current cluster to current index
    }
  }
  # add last subCluster to allClusters
  allClusters[[length(allClusters)+1]] = subCluster
  return(allClusters)
}


# find duration (length) and severity (sum of magnitudes) of clusters
findDandS = function(clusters, magnitudes){
  durations = c(rep(0,length(clusters)))
  severities = c(rep(0,length(clusters)))
  for(i in 1:length(clusters)){
    durations[i] = length(clusters[[i]])
    severities[i] = sum(magnitudes[as.numeric(clusters[[i]])])
  }
  return(list(durations=durations, severities=severities))
}

