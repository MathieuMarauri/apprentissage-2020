
# Correction du TP1

# Set-up ---------------------------------------------------------------------------------

library("car") # fonction Boxplot
library("caret") # ensemble de fonctions nécessaire au processus d'apprentissage supervisé
library("class") # fonction knn
library("e1071") # librairie nécessaire pour la fonction tune.knn
library("FNN") # fast KNN
library("ggplot2") # Data visualisations using the Grammar of Graphics
library("mlbench") # dataset Vehicle
library("skimr") # visualisation de l'ensemble d'un dataset

# Set default ggplot theme
theme_set(
  theme_light(
  base_size = 20
  ) +
  theme(
    text = element_text(family = "Gibson", colour = "gray10"),
    panel.border = element_blank(),
    axis.line = element_line(colour = "gray50", size = .5),
    axis.ticks = element_blank(),
    strip.background = element_rect(colour = "gray50", fill = "transparent", size = .7),
    strip.text.x = element_text(colour = "gray10"),
    strip.text.y = element_text(colour = "gray10"),
    legend.key.size = unit(1.5, "cm")
  )
)

# Set default scales
scale_colour_continuous <- function(...) ggplot2::scale_colour_viridis_c(..., option = "viridis")
scale_colour_discrete <- function(...) ggplot2::scale_colour_viridis_d(..., option = "viridis")
scale_fill_continuous <- function(...) ggplot2::scale_fill_viridis_c(..., option = "viridis")
scale_fill_discrete <- function(...) ggplot2::scale_fill_viridis_d(..., option = "viridis")


# Exercice 1 ----------------------------------------------------------------------------------

### Question 3

# Nombre d'éléments dans l'échantillon
n_train <- 1000

# Générer X selon une loi normale (fixer une seed pour avoir des résultats similaires à chaque
# fois)
set.seed(1)
X <- rnorm(n_train)

# Générer U selon une loi uniforme (fixer une seed pour avoir des résultats similaires à chaque
# fois)
set.seed(2)
U <- runif(n_train)

# Générer le vecteur réponse Y selon le cas 2, transformer en facteur pour la fonction
# tune.knn (utilisée dans la suite)
Y1 <- rep(0, n_train)
Y1[X <= 0 & U <= 0.2] <- 1
Y1[X > 0 & U > 0.4] <- 1
Y <- as.factor(Y1)

# Créer un data.frame avec X et Y
donnees <- data.frame(X, Y)

# Visualiser les données
ggplot(
  data = donnees,
  mapping = aes(x = X, fill = Y)
) +
  geom_density(
    alpha = .7
  ) +
  labs(
    x = "X",
    y = "Densité",
    title = "Densité de X selon Y",
    fill = "Y"
  )
# Boxplot(data = donnees, X ~ Y, main = "Données apprentissage et validation")

# Refaire la même chose pour la création de l'échantillon test
n_test <- 200
set.seed(3)
X <- rnorm(n_test)
set.seed(4)
U <- runif(n_test)
Y2 <- rep(0, n_test)
Y2[X <= 0 & U <= 0.2] <- 1
Y2[X > 0 & U > 0.4] <- 1
Y <- as.factor(Y2)
test <- data.frame(X, Y)
ggplot(
  data = test,
  mapping = aes(x = X, fill = Y)
) +
  geom_density(
    alpha = .7
  ) +
  labs(
    x = "X",
    y = "Densité",
    title = "Densité de X selon Y",
    fill = "Y"
  )
# Boxplot(data = test, X ~ Y, main = "Données apprentissage et validation")


### Question 4 : K plus proches voisins

# On cherche la meilleure valeur de K par cross validation
knn_cross_ressults <- tune.knn(
  x = donnees$X, # predicteurs
  y = donnees$Y, # réponse
  k = 1:50, # essayer knn avec K variant de 1 à 50
  tunecontrol = tune.control(sampling = "cross"), # utilisation de la cross validation
  cross = 5 # 5 blocks
)

# Visualiser les résultats pour chaque K
summary(knn_cross_ressults)
plot(knn_cross_ressults)

# Relancer la cross validation plusieurs fois pour déterminer le meilleur K (la valeur peut varier
# pour chaque lancement de tune.knn)
best_k <- numeric(length = 20)
for (i in 1:20) {
  knn_cross_ressults <- tune.knn(
    x = donnees$X,
    y = donnees$Y,
    k = 1:50,
    tunecontrol = tune.control(sampling = "cross"),
    cross = 10
  )
  best_k[i] <- knn_cross_ressults$best.parameters[[1]]
  print(sprintf("Itération %s : meilleur K = %s", i, knn_cross_ressults$best.parameters))
}
table(best_k)

# On cherche la meilleure valeur de K par bootstrap
knn.boot <- tune.knn(
  x = donnees$X, # predicteurs
  y = donnees$Y, # réponse
  k = 1:50, # essayer knn avec K variant de 1 à 50
  tunecontrol = tune.control(sampling = "boot") # utilisation du bootstrap
)

# Visualiser les résultats pour chaque K
summary(knn.boot)
plot(knn.boot)

# Relancer la cross validation plusieurs fois pour déterminer le meilleur K (la valeur peut varier
# pour chaque lancement de tune.knn)
best_k <- numeric(length = 20)
for (i in 1:20) {
  knn_cross_ressults <- tune.knn(
    x = donnees$X,
    y = donnees$Y,
    k = 1:50,
    tunecontrol = tune.control(sampling = "boot") # utilisation du bootstrap
  )
  best_k[i] <- knn_cross_ressults$best.parameters[[1]]
  print(sprintf("Itération %s : meilleur K = %s", i, knn_cross_ressults$best.parameters))
}
table(best_k)

# On va ensuite tester la qualité du modèle avec le paramètre choisi
set.seed(123456)
knn_pred <- knn(
  train = as.matrix(donnees$X), # données d'apprentissage
  test = as.matrix(test$X), # données à prédire
  cl = donnees$Y, # vraies valeurs
  k = 30 # nombre de voisins
)

# Taux de bonnes prédictions
sum(knn_pred == test$Y) / n_test

# Taux d'erreur
1 - sum(knn_pred == test$Y) / n_test

# L'erreur est inférieur au risque de Bayes, notre estimation de l'espérance de l'erreur est
# peut-être mauvaise, utilisation d'un nouvel échantillon plus grand
n_test2 <- 2000
set.seed(6)
X <- rnorm(n_test2)
set.seed(7)
U <- runif(n_test2)
Y1 <- rep(0, n_test2)
Y1[X <= 0 & U <= 0.2] <- 1
Y1[X > 0 & U > 0.4] <- 1
Y <- as.factor(Y1)
test2 <- data.frame(X, Y)
set.seed(123456)
knn_pred <- knn(
  train = as.matrix(donnees$X), # données d'apprentissage
  test = as.matrix(test2$X), # données à prédire
  cl = donnees$Y, # vraies valeurs
  k = 30 # nombre de voisins
)
1 - sum(knn_pred == test2$Y) / n_test2 # c'est mieux


### Question 5 : Classifieur Bayésien naïf

# Définir la liste des hyper-paramètres possibles pour le classifieur bayésien naïf
grid <- expand.grid(
  usekernel = c(TRUE, FALSE), # si vrai utilisation d'un noyau sinon gaussien
  # fL = 0:5, # correction avec lissage de Laplace (ici ce paramètre n'est pas nécessaire, x étant continue)
  fL = 0,
  adjust = seq(1, 5, by = 1) # largeur de bande
)

# Définir la méthode de validation, ici 5-fold cross validation
control <- trainControl(method = "cv", number = 5)

# Définir les entrées de la fonction train de caret, nommer les éléments est une contrainte de
# caret
x <- donnees[, "X", drop = FALSE]
y <- donnees$Y

# On optimise les paramètres du modèle
naive_bayes <- train(
  x = x, # prédicteurs
  y = y, # réponse
  method = "nb", # classifieur utilisé, ici Naive Bayes
  trControl = control, # méthode d'échantillonnage, ici 5-fold CV
  tuneGrid = grid # liste des paramètres à comparer
)

# visualisation des résultats
plot(naive_bayes) # on voit bien que la correction de Laplace n'apporte rien

# On va ensuite tester la qualité du modèle avec le paramètre choisi (modèle non paramétrique
# et BW = 1)
naive_bayes_pred <- predict(naive_bayes, test)

# calcule de l'erreur
sum(naive_bayes_pred == test$Y) / n_test
1 - sum(naive_bayes_pred == test$Y) / n_test

# nettoyer la session
rm(list = ls())


# Exercice 2 ----------------------------------------------------------------------------------

### Question 1 : lecture de la table

# Importer le dataset vehicule et avoir un aperçu des données
data(Vehicle)
summary(Vehicle)
# skim(Vehicle)

# On décompose la base en un échantillon d'apprentissage et un échantillon de test
set.seed(123456)
train_index <- sample(1:nrow(Vehicle), size = nrow(Vehicle) * .8)
test_index <- setdiff(1:nrow(Vehicle), train_index)
train <- Vehicle[train_index, ]
test <- Vehicle[test_index, ]

# Donner la liste des prédicteurs
predicteurs <- names(Vehicle)[-19]
# on peut sélectionner moins de variable en faisant una analyse bivariée, une ACP pour garder
# seulement les premiers axes, ...

### Question 3 : K plus proches voisins

# On cherche la meilleure valeur de K par cross validation
knn.cross <- tune.knn(
  x = scale(train[, predicteurs]), # predicteurs
  y = train[, "Class"], # réponse
  k = 1:50, # essayer knn avec K variant de 1 à 50
  tunecontrol = tune.control(sampling = "cross"), # utilisation de la cross validation
  cross = 5 # 5 blocks
)

# Visualiser les résultats pour chaque K
summary(knn.cross)
plot(knn.cross)

# relancer la cross validation plusieurs fois pour déterminer le meilleur K
best_k <- numeric(length = 20)
for (i in 1:20) {
  knn_cross_ressults <- tune.knn(
    x = scale(train[, predicteurs]),
    y = train[, "Class"],
    k = 1:50,
    tunecontrol = tune.control(sampling = "cross"),
    cross = 5
  )
  best_k[i] <- knn_cross_ressults$best.parameters[[1]]
  print(sprintf("Itération %s : meilleur K = %s", i, knn_cross_ressults$best.parameters))
}
table(best_k)

# On cherche la meilleure valeur de K par bootstrap
knn_boot_ressults <- tune.knn(
  x = scale(train[, predicteurs]), # predicteurs
  y = train[, "Class"], # réponse
  k = 1:50, # essayer knn avec K variant de 1 à 50
  tunecontrol = tune.control(sampling = "boot") # utilisation du bootstrap
)

# visualiser les résultats pour chaque K
summary(knn_boot_ressults)
plot(knn_boot_ressults)

# On va ensuite tester la qualité du modèle avec le paramètre choisi
set.seed(123456)
knn_pred <- knn(
  train = as.matrix(scale(train[, predicteurs])), # données d'apprentissage
  test = as.matrix(scale(test[, predicteurs])), # données à prédire
  cl = train[, "Class"], # vraies valeurs
  k = 3 # nombre de voisins
)

# Taux de bonnes prédictions
sum(knn_pred == test[, "Class"]) / length(test_index)

# Taux d'erreur
1 - sum(knn_pred == test[, "Class"]) / length(test_index)


### Question 4 : Classifieur Bayésien naïf

# définir la liste des hyper-paramètres possibles pour le classifieur bayésien naïf
grid <- expand.grid(
  usekernel = c(TRUE, FALSE), # si vrai utilisation d'un noyau sinon gaussien
  fL = 0, # pas de correction car les x sont continues
  adjust = seq(1, 5, by = 1) # bandwidth
)

# définir la méthode de validation, ici 10-fold cross validation
control <- trainControl(method = "cv", number = 5)

# définir les entrées de al fonction train de caret, nommer les éléments est une contrainte de caret
x <- train[, predicteurs]
y <- train[, "Class"]

# On optimise les paramètres du modèle
naive_bayes <- train(
  x = x, # prédicteurs
  y = y, # réponse
  preProc = c("BoxCox", "center", "scale", "pca"), # utilisation de différents pré-traitement
  method = "nb", # classifieur utilisé, ici Naive Bayes
  trControl = control, # méthode d'échantillonnage, ici 5-fold CV
  tuneGrid = grid # liste des paramètres à comparer
)

# Visualisation des résultats
plot(naive_bayes)

# On va ensuite tester la qualité du modèle avec le paramètre choisi (modèle non paramétrique
# et BW = 1)
naive_bayes_pred <- predict(naive_bayes, test)
# naive_bayes_pred <- predict(naive_bayes, test, type = "prob")

# calcule de l'erreur
sum(naive_bayes_pred == test[, "Class"]) / length(test_index)
1 - sum(naive_bayes_pred == test[, "Class"]) / length(test_index)

# nettoyer la session
rm(list = ls())


# Exercice 3 ----------------------------------------------------------------------------------

### Lecture de la table

# importer le dataset airbnb et avoir un aperçu des données
airbnb <- read.csv(file = here("AB_NYC_2019.csv"))
summary(airbnb)
skim(airbnb)

# transformer les facteurs en numeric pour knn
airbnb$room_type <- as.numeric(airbnb$room_type)

# On décompose la base en un échantillon d'apprentissage et un échantillon de test
set.seed(123456)
train_index <- sample(1:nrow(airbnb), size = nrow(airbnb) * .7)
test_index <- setdiff(1:nrow(airbnb), train_index)
train <- airbnb[train_index, ]
test <- airbnb[test_index, ]

# donner la liste des prédicteurs (utiliser dput(names(airbnb))), les colonnes d'identifiant sont
# enlevées
predicteurs <- c(
  "latitude", "longitude",
  "room_type", "minimum_nights",
  "calculated_host_listings_count", "availability_365"
)
# seules ces variables sont utilisables car les autres ne sont pas observées dans notre cas (nouvel
# appartement sur Airbnb)

### Question 3 : K plus proches voisins

# on cherche la meilleure valeur de K par cross validation
knn.fit <- train(
  x = train[, predicteurs], # prédicteurs
  y = train[, "price"], # réponse
  tuneGrid = expand.grid(k = 20:50), # nombre de voisins
  method = "knn", # knn classifieur
  trControl = trainControl(method = "cv", number = 5) # 5-fold CV
)

# visualisation des résultats
plot(knn.fit)

# On va ensuite tester la qualité du modèle avec le paramètre choisi (modèle non param et BW = 1)
knn.pred <- predict(knn.fit, test)

# calcule de l'erreur
mean((knn.pred - test[, "price"])^2)
