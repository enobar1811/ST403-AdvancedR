## CA3_Exercise3.R

## Exercise 3: MLE for linear regression via optim()

## 1. Data
df <- with(mtcars,
           data.frame(y  = mpg,
                      x1 = disp,
                      x2 = hp,
                      x3 = wt))

## 2. Negative log-likelihood for LM
nll_lm <- function(data, par) {
  X <- model.matrix(y ~ x1 + x2 + x3, data = data)
  y <- data$y
  p <- ncol(X)
  beta  <- par[1:p]
  sigma <- par[p + 1]
  if (sigma <= 0) return(.Machine$double.xmax)
  eps  <- as.vector(y - X %*% beta)
  llik <- dnorm(eps, mean = 0, sd = sigma, log = TRUE)
  -sum(llik)
}

## 3. Optimisation
X  <- model.matrix(y ~ x1 + x2 + x3, data = df)
p  <- ncol(X)
y_vec <- df$y

init_beta  <- rep(0, p)
init_beta[1] <- mean(df$y)
init_sigma <- sd(df$y)

inits <- c(init_beta, init_sigma)
lower <- c(rep(-Inf, p), .Machine$double.xmin)
upper <- c(rep( Inf, p), .Machine$double.xmax)

fit <- optim(par     = inits,
             fn      = nll_lm,
             data    = df,
             method  = "L-BFGS-B",
             lower   = lower,
             upper   = upper,
             hessian = TRUE)

beta_hat  <- fit$par[1:p]
sigma_hat <- fit$par[p + 1]

## 4. Matrix comparisons
beta_hat_mat <- solve(t(X) %*% X, t(X) %*% y_vec)
resid_mat    <- y_vec - as.vector(X %*% beta_hat_mat)
n <- length(y_vec)

sigma_mle_mat <- sqrt(sum(resid_mat^2) / n)
sigma_unb_mat <- sqrt(sum(resid_mat^2) / (n - p))

## 5. SEs from Hessian
vcov_all <- solve(fit$hessian)
se_beta  <- sqrt(diag(vcov_all)[1:p])
