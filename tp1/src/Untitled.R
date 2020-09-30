klaR::NaiveBayes


function (x, grouping, prior = NULL, usekernel = FALSE, fL = 0,
          ...)
{
  x <- data.frame(x)
  if (!is.factor(grouping))
    stop("grouping/classes object must be a factor")
  if (is.null(prior))
    apriori <- table(grouping)/length(grouping)
  else apriori <- as.table(prior/sum(prior))
  call <- match.call()
  Yname <- "grouping"
  LaplaceEst <- function(x, f = 0) (x + f)/(rowSums(x) + f *
                                              ncol(x))
  est <- function(var) {
    if (is.numeric(var)) {
      temp <- if (usekernel)
        lapply(split(var, grouping), FUN = function(xx) density(xx,
                                                                ...))
      else cbind(tapply(var, grouping, mean), tapply(var,
                                                     grouping, sd))
    }
    else LaplaceEst(table(grouping, var), f = fL)
  }
  tables <- lapply(x, est)
  if (!usekernel) {
    num <- sapply(x, is.numeric)
    temp <- as.matrix(sapply(tables, function(x) x[, 2]))
    temp[, !num] <- 1
    temp <- apply(temp, 2, function(x) any(!x))
    if (any(temp))
      stop("Zero variances for at least one class in variables: ",
           paste(names(tables)[temp], collapse = ", "))
  }
  names(dimnames(apriori)) <- Yname
  structure(list(apriori = apriori, tables = tables, levels = levels(grouping),
                 call = call, x = x, usekernel = usekernel, varnames = colnames(x)),
            class = "NaiveBayes")
}


function (object, newdata, threshold = 0.001, ...)
{
  if (missing(newdata))
    newdata <- object$x
  if (sum(is.element(colnames(newdata), object$varnames)) <
      length(object$varnames))
    stop("Not all variable names used in object found in newdata")
  newdata <- data.frame(newdata[, object$varnames])
  nattribs <- ncol(newdata)
  islogical <- sapply(newdata, is.logical)
  isnumeric <- sapply(newdata, is.numeric)
  newdata <- data.matrix(newdata)
  Lfoo <- function(i) {
    tempfoo <- function(v) {
      nd <- ndata[v]
      if (is.na(nd))
        return(rep(1, length(object$apriori)))
      prob <- if (isnumeric[v]) {
        msd <- object$tables[[v]]
        if (object$usekernel)
          sapply(msd, FUN = function(y) dkernel(x = nd,
                                                kernel = y, ...))
        else dnorm(nd, msd[, 1], msd[, 2])
      }
      else if (islogical[v]) {
        object$tables[[v]][, nd + 1]
      }
      else {
        object$tables[[v]][, nd]
      }
      prob[prob == 0] <- threshold
      return(prob)
    }
    ndata <- newdata[i, ]
    tempres <- log(sapply(1:nattribs, tempfoo))
    L <- log(object$apriori) + rowSums(tempres)
    if (isTRUE(all.equal(sum(exp(L)), 0)))
      warning("Numerical 0 probability for all classes with observation ",
              i)
    L
  }
  L <- sapply(1:nrow(newdata), Lfoo)
  classdach <- factor(object$levels[apply(L, 2, which.max)],
                      levels = object$levels)
  posterior <- t(apply(exp(L), 2, function(x) x/sum(x)))
  colnames(posterior) <- object$levels
  rownames(posterior) <- names(classdach) <- rownames(newdata)
  return(list(class = classdach, posterior = posterior))
}



data(iris)
m <- NaiveBayes(Species ~ ., data = iris)
predict(m, iris[1:10,])
library("klaR")

naive_bayes_model <- NaiveBayes(
  formula = Y ~ X,
  data = donnees
)

naive_bayes_pred <- predict(naive_bayes_model, test)
