## ----
## Estudo sobre métodos 
## ensemble sobre a Regressão Beta
## ----

## Dados: Taxa de beneficiários do bolsa família, em 2015.

## Bibliotecas

if(!require(pacman)) install.packages("pacman"); library(pacman)
p_load(ggplot2,dplyr, tidyverse,
       caret,   # for general data preparation and model fitting
       e1071) # for fitting the xgboost model

## Leitura do banco de dados

setwd("C:/Users/dioni/OneDrive - University of São Paulo/Doutorado em Estatística/2023.2/2 - Aprendizado de Máquina Estatístico/Codigos_Pratica/3_Trabalho_Final/dados")

bf = read.csv("bolsafamilia_tx.csv")
head(bf)

## retirar a coluna indice
bf = bf %>% subset(select=-c(X))

## ---
## Separação das covariaveis e da resposta
## ---

head(bf)

y = bf %>% subset(select=c(tx_benbf))
X = bf %>% subset(select=c(FECTOT, RAZDEP,
                           E_ANOSESTUDO, T_ANALF18M,
                           T_FBBAS, T_FBFUND,
                           T_FBMED, T_FBSUPER,
                           GINI, PIND, PPOB, RDPCT,
                           THEIL, P_FORMAL, T_BANAGUA, T_DENS,
                           T_LIXO, T_LUZ,AGUA_ESGOTO,
                           PAREDE, T_M10A14CF, T_M15A17CF,
                           I_ESCOLARIDADE,  IDHM, IDHM_L, IDHM_R,
                           tx_ocupacao_urbana, capital))

databf = cbind(X,y)
dim(databf)

## Separação entre treino e teste

set.seed(10)
# Proporção treino
prop_treino = 0.8
n_treino = round(nrow(databf) * prop_treino)

ind_treino = sample(1:nrow(databf), n_treino)

# Criar conjuntos de treino e teste 
treinobf = databf[ind_treino, ]
testebf = databf[-ind_treino, ]

dim(treinobf);dim(testebf)

## -------
## Machine Learning Regression Models
## -------

## Métricas de desempenho

MSE = function(y,ypred){mean((y - ypred)^2)}
MAE = function(y,ypred){mean(abs(y - ypred))}
R2 = function(y,ypred){
  num = sum((y-ypred)^2)
  dem =  sum((y-mean(y))^2)
  r = 1 - (num/dem)
  return(r)
}

## ----
## Máquinas de Vetores de Suporte (SVM)
## ----

timesvm_inic <- Sys.time()

modelsvm=svm(tx_benbf~.,
              data = treinobf,
             gamma = 0.01,
             cost = 1,
             kernel = "radial")

timesvm_end <- Sys.time()

ypredsvm=predict(modelsvm,data=testebf)

MSE(y=testebf$tx_benbf,ypred=ypredsvm)
sqrt(MSE(y=testebf$tx_benbf,ypred=ypredsvm))
MAE(y=testebf$tx_benbf,ypred=ypredsvm)
R2(y=testebf$tx_benbf,ypred=ypredsvm)

paste("Tempo de treinamento: ", timesvm_end - timesvm_inic)

## Tuning SVR model by varying values of maximum allowable error and cost parameter

#Tune the SVM model
OptModelsvm=tune(svm, tx_benbf~.,
                 data = treinobf,
                 ranges=list(gamma=c(1,0.1,0.01,0.001),
                             cost=c(0.1,1,10,100), kernel=c("linear", "polynomial",
                                                  "radial", "sigmoid")),
                 tunecontrol = tune.control(cross = 10))

#Print optimum value of parameters
print(OptModelsvm)

#Plot the perfrormance of SVM Regression model
plot(OptModelsvm)

## ---
## Random Forest Regressor (RFR)
## ---

#Tune the Random Forest model
install.packages("randomForest")
library(randomForest)

timeRF_inic <- Sys.time()

mdRFtree = randomForest(tx_benbf~.,
                        data = treinobf,
                        ntree = 25,
                        maxfeatures=20,
                        maxnodes = 9)
timeRF_end <- Sys.time()

ypredRF = predict(mdRFtree,newdata=testebf)

## Metricas
MAE(y=testebf$tx_benbf,ypred=ypredRF)
MSE(y=testebf$tx_benbf,ypred=ypredRF)
sqrt(MSE(y=testebf$tx_benbf,ypred=ypredRF))
R2(y=testebf$tx_benbf,ypred=ypredRF)

## Tempo de treinamento
paste("Tempo de treinamento: ", timeRF_end-timeRF_inic)

OptModelRF=tune(randomForest, tx_benbf~.,
                 data = treinobf,
                 ranges=list(ntree = c(25,50,100,500),
                             maxfeatures = c(5,10,20,25),
                             maxnodes = c(3,6,9),
                 tunecontrol = tune.control(cross = 10)))

#Print optimum value of parameters
print(OptModelRF)

#Plot the perfrormance of SVM Regression model
plot(OptModelRF)




