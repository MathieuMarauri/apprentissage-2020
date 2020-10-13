
# Correction de la partie 2 du TD 2

# Set-up ------------------------------------------------------------------------------------

# install.packages("CHAID", repos="http://R-Forge.R-project.org")
library('CHAID') # modèle CHAID
library('rpart') # modèle CART
library('rpart.plot') # visualisation du modèle CART


# Question 9 - Un premier modèle ---------------------------------------------------------

# Importer les données
musch <- read.csv(file = "tp2/data/muschroom.csv", stringsAsFactors = TRUE)

# Enlever la colonne X
musch <- musch[, !names(musch) == "X"]

# Définir les échantillons train et test
train <- musch[musch$echantillon == "base", !names(musch) == "echantillon"]
test <- musch[musch$echantillon == "test", !names(musch) == "echantillon"]

# Définir les paramètres utilisés par rpart
cart_parameters <- rpart.control(
  minsplit = 60, # il faut au moins 60 observations dans un noeud pour le diviser
  minbucket = 30, # une division ne doit pas générer un noeud avec moins de 30 observations
  xval = 10, # nombre de blocs utilisés pour la validation croisée de l'élagage
  maxcompete = 4, # nombre de divisions compétitives retenues (equi reducteur)
  maxsurrogate = 4, # nombre de divisions surrogates retenues (equi divisant)
  usesurrogate = 2, # comment sont gérées les valeurs manquantes, voir la documentation pour plus d'infos
  maxdepth = 30 # la profondeur maximal de l'arbre,
  # cp = 0 # Ne pas limiter la construction de l'arbre maximal
)

# Entraînement du modèle
cart_model <- rpart(
  formula = classe ~ .,
  data = train,
  method = "class",
  control = cart_parameters,
  parms = list(split = 'gini')
)

# Effectuer une prédiction sur les données test
cart_pred <- predict(
  object = cart_model,
  newdata = test,
  type = "class"
)

# Visualiser l'arbre
rpart.plot(cart_model)

# Afficher la matrice de confusion
table(test$classe, cart_pred)

# print the model
print(cart_model)

# plot the tree
rpart.plot(cart_model)

# print the results of the model
summary(cart_model)

# evolution of the tree size and of the error based on the cp parameter
plotcp(cart_model)

# probabilities for the two classes for each observations of the test set
predict(model, test)



# error rate
error_rate <- sum(predtest != test$classe) / nrow(test)

# model with split criteria using information criterion and equal a priori probabilities
# model2 <- rpart(
#   spam ~ .,
#   data = spam,
#   method = "class",
#   control = parametres,
#   parms = list(
#     prior =c(0.5, 0.5),
#     split ='information'
#   )
# )


# Adding costs --------------------------------------------------------------------------------

# Adding special cost for errors: C(eatable/poison) = 1000, C(poison/eatable) = 1

# build the model
cart_model_cost <- rpart(
  classe ~ .,
  data = base,
  parms = list(
    split = "gini",
    loss = matrix(
      c(0, 1, 1000, 0),
      byrow = TRUE,
      nrow = 2)
  ),
  control = parametres
)

# print the model
print(model2)

# plot the tree
rpart.plot(model2)

# print the results of the model
summary(model2)

# evolution of the tree size and of the error based on the cp parameter
plotcp(model2)

# probabilities for the two classes for each observations of the test set
predict(model2, test)

# predicted class for each observtion
predtest2 <- predict(model2, test, type = "class")

# confusion matrix
table(test$classe, predtest2)

# calcul du taux d'erreur en test
sum(predtest2 != test$classe) / nrow(test)


# CHAID ---------------------------------------------------------------------------------------

# Decision tree using CHAID

# see the help page
?chaid

# chaid decision tree
model3 <- chaid(
  formula = classe ~ .,
  data = train,
  control = chaid_control(
    minsplit = 60,
    minbucket = 30
  )
)

# model results
print(model3)

# plot the model
plot(
  model3,
  uniform = TRUE,
  compress = TRUE,
  margin = 0.2,
  branch = 0.3
)
plot(model3)

