## ----
## Estudo sobre métodos 
## ensemble sobre a Regressão Beta
## ----

## Dados: Taxa de beneficiários do bolsa família, em 2015.

## Bibliotecas

if(!require(pacman)) install.packages("pacman"); library(pacman)
p_load(ggplot2,dplyr, betareg, beepr,betaboost,e1071,caret,compiler,gridExtra)


## Leitura do banco de dados

#setwd("C:/Users/dioni/OneDrive - University of São Paulo/Doutorado em Estatística/2023.2/2 - Aprendizado de Máquina Estatístico/3_Trabalho_Final")

bf = read.csv("https://raw.githubusercontent.com/Dionisioneto/Aprendizado_de_Maquina_Estatistico/main/3_Trabalho_Final/dados/bolsafamilia_tx.csv")
head(bf)

## retirar a coluna indice
bf = bf %>% subset(select=-c(X))

## ---
## Separação das covariaveis e da resposta
## ---

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

## Realizando um pequena mudança na resposta
## para que o velopr não seja 1
## 1 munícipios teve todos os seus candidatos escolhidos

y[y>0.99999] = 0.9999

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


## -----
## Métricas de ajuste
## -----

MSE = function(y,ypred){mean((y - ypred)^2)}
MAE = function(y,ypred){mean(abs(y - ypred))}
R2 = function(y,ypred){
  num = sum((y-ypred)^2)
  dem =  sum((y-mean(y))^2)
  r = 1 - (num/dem)
  return(r)
}



## -------
## Modelo de Regressão Beta
## -------
#library(gamlss)

timebreg_inic <- Sys.time()

breg = betareg(tx_benbf ~.,
                data=treinobf,
               link="loglog")

timebreg_end <- Sys.time()

## Tempo de treinamento
paste("Tempo de treinamento: ", timebreg_end-timebreg_inic)


#summary(breg)

ypredbreg = predict(breg,testebf)

MAE(y=testebf$tx_benbf,ypred=ypredbreg)
MSE(y=testebf$tx_benbf,ypred=ypredbreg)
sqrt(MSE(y=testebf$tx_benbf,ypred=ypredbreg))

#R2(y=testebf$tx_benbf,ypred=ypredbreg)




## Validação Cruzada (k-fold cross validation)
## para a regressão beta

## O elemento de variação é a função de ligação

#linksfun = c("logit", "probit", "cloglog", "cauchit", "log", "loglog")

# for (link in linksfun) {
#   
#   breg = betareg(tx_benbf ~.,
#                  data=treinobf,
#                  link=link)
#   
#   ypredbreg = predict(breg,testebf)
#   
#   msebeta = MSE(ypred=ypredbreg,ytrue=testebf$tx_benbf)
#   msebeta
#   
# }

set.seed(10)
kfolds = 10

tcv_data = treinobf %>% 
  mutate(fold = sample(1:kfolds, size=dim(treinobf)[1], replace=T))

cv_err = rep(0, kfolds)
matrix_cv = matrix(data=0,nrow=length(linksfun),
                   ncol = kfolds,
                   dimnames = list(linksfun))

enableJIT(3)
for(link in 1:length(linksfun)){
  for(i in 1:kfolds){
    t_data = filter(tcv_data, fold!=i)
    v_data = filter(tcv_data, fold==i)
    
    fitbeta = betareg(tx_benbf ~.,
                   data=t_data,
                   link = linksfun[link])
    
    predsbeta = predict(fitbeta, newdata=v_data)
    
    err = v_data$tx_benbf - predsbeta 
    mse = mean(err^2)
    
    # Record the RMSE
    matrix_cv[link,i] <- sqrt(mse)
  }
}

matrix_cv

## as medias da validação cruzada

colMeans(matrix_cv)
linkmin=which(colMeans(matrix_cv)==min(colMeans(matrix_cv)))

matrix_cv ## ligação log-log é a que retorno o menor RMSE

## ----
## Iremos treinar o modelo final com o loglog
## ----

modbeta = betareg(tx_benbf ~.,
            data=treinobf,
              link = "loglog")

ypbreg = predict(modbeta,testebf)

msebetareg = mean((testebf$tx_benbf - ypbreg)^2)
msebetareg ## EQM


MSE = function(y,ypred){mean((y - ypred)^2)}
MAE = function(y,ypred){mean(abs(y - ypred))}


## RMSE
sqrt(msebetareg)

## MAE
mean(abs(testebf$tx_benbf - ypbreg))

R2 = function(y,ypred){
  num = sum((y-ypred)^2)
  dem =  sum((y-mean(y))^2)
  r = 1 - (num/dem)
  return(r)
}

## R2: Coeficiente de Determinação
R2(y=testebf$tx_benbf,ypred=ypbreg) 
  
## ----
## Algoritmo de Estimação Bagging
## ----

## ----
## Algoritmo de Bagging para a Regressão Beta
## ----

bagging_betareg = function(xtreino,ytreino,xteste,n_estimadores,n_amostra,linkfun){
  
  ## predicoes do bagging em uma matriz
  matriz_bag = matrix(data=0,nrow=dim(xteste)[1],ncol=n_estimadores)
  
  enableJIT(3)
  for(preditor in 1:n_estimadores){
    random_id = sample(1:dim(xtreino)[1],replace=T)
    
    xtreino_bag = xtreino[random_id,]
    ytreino_bag = ytreino[random_id]
    
    dados_bag = cbind(ytreino_bag,xtreino_bag)
    colnames(dados_bag)[1] = "y"
    
    reg_bag = betareg(y ~ .,
                      data=dados_bag,
                      link=linkfun)
    
    y_pred_bag = predict(reg_bag, newdata=xteste)
    matriz_bag[,preditor] = y_pred_bag
  }
  
  predicoes_bag = rowSums(matriz_bag)/n_estimadores
  return(predicoes_bag)
}

timebag_inic <- Sys.time()

y_pred_bagging= bagging_betareg(xtreino=treinobf[, -which(names(treinobf) == "tx_benbf")],
                                ytreino=treinobf[, "tx_benbf"],
                                xteste=testebf[, -which(names(testebf) == "tx_benbf")],
                                n_estimadores=20,
                                n_amostra=dim(treinobf)[1],
                                linkfun = "loglog")

# Marcar o final do tempo
timebag_fim <- Sys.time()

paste("Tempo de treinamento: ", timebag_fim-timebag_inic)

MAE(y=testebf$tx_benbf,ypred=y_pred_bagging)
MSE(y=testebf$tx_benbf,ypred=y_pred_bagging)
sqrt(MSE(y=testebf$tx_benbf,ypred=y_pred_bagging))


## grid search beta reg

n_estimators = seq(10,800,by=10)

matrix_cv_bag = matrix(data=0,nrow=length(n_estimators),ncol=1)

for(i in 1:length(n_estimators)){
    y_pred_bag = bagging_betareg(xtreino=treinobf[, -which(names(treinobf) == "tx_benbf")],
                                ytreino=treinobf[, "tx_benbf"],
                                xteste=testebf[, -which(names(testebf) == "tx_benbf")],
                                n_estimadores=n_estimators[i],
                                n_amostra=dim(treinobf)[1],
                                linkfun = "logit")
    
    matrix_cv_bag[i,1] = MSE(y=testebf$tx_benbf,ypred=y_pred_bag)
    print(paste("estimators: ", n_estimators[i]))
  
}


#plot(matrix_cv_bag,type='b')
cv_bag = as.data.frame(matrix_cv_bag)
cv_bag$ests = n_estimators 
colnames(cv_bag)[1] = "EQM"

ggplot(data=cv_bag, aes(x=n_estimators, y=EQM, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  theme_minimal()


## ---
## Realizando o processo de Grid Search
## ---
# 
# set.seed(12)
# 
# b_values = c(200,500,1000)
# linksfun
# 
# kfolds = 10
# 
# tcv_bag = treinobf %>% 
#   mutate(fold = sample(1:kfolds, size=dim(treinobf)[1], replace=T))
# 
# 
# matrix_cv_bag = matrix(0,nrow=length(b_values),ncol=kfolds,
#                        dimnames = list(b_values))
# 
# 
# timebag_inic <- Sys.time()
# 
# enableJIT(3)
# for (b in 1:length(b_values)){
#     for(i in 1:kfolds){
#       t_data = filter(tcv_data, fold!=i)
#       v_data = filter(tcv_data, fold==i)
#       
#       predsbetabag = bagging_betareg(xtreino=t_data[, -which(names(t_data) == "tx_benbf")],
#                                    ytreino=t_data[, "tx_benbf"],
#                                    xteste=v_data[, -which(names(v_data) == "tx_benbf")],
#                                    n_estimadores=b_values[b],
#                                    n_amostra=dim(t_data)[1],
#                                    linkfun = "loglog")
#       
#       errbag = v_data$tx_benbf - predsbetabag 
#       msebag = mean(errbag^2)
#       
#       # Record the RMSE
#       matrix_cv_bag[b,i] <- sqrt(mse)
#     }
# }


# Marcar o final do tempo
#timebag_fim <- Sys.time()


## MSE
MSE(y=testebf$tx_benbf,ypred=y_pred_bagging)

## RMSE
sqrt(MSE(y=testebf$tx_benbf,ypred=y_pred_bagging))

## MAE
MAE(y=testebf$tx_benbf,ypred=y_pred_bagging)


## R2: Coeficiente de Determinação
R2(y=testebf$tx_benbf,ypred=y_pred_bagging) 



# ## Calculando o EQM
# 
# msebetabag1 = MSE(ypred=y_pred_bagging,ytrue=testebf$tx_benbf)
# 
# msebetabag1
# 
# sqrt(msebetabag1)


## ----
## Algoritmo de Estimação Random Forest
## ----


rf_betareg = function(xtreino,ytreino,xteste,
                           n_estimadores,n_amostra,n_features){
  
  ## predicoes do Random Forest em uma matriz
  matriz_rf = matrix(data=0,nrow=dim(xteste)[1],ncol=n_estimadores)
  
  enableJIT(3)
  for(preditor in 1:n_estimadores){
    random_id = sample(1:dim(xtreino)[1],replace=T)
    random_idf = sample(1:dim(xtreino)[2],replace=F)[1:n_features]
    
    xtreino_rf = xtreino[random_id,random_idf]
    ytreino_rf = ytreino[random_id]
    
    dados_rf = cbind(ytreino_rf,xtreino_rf)
    colnames(dados_rf)[1] = "y"
    
    reg_rf = betareg(y ~ .,
                      data=dados_rf,
                      link='logit')
    
    y_pred_rf = predict(reg_rf, newdata=xteste)
    matriz_rf[,preditor] = y_pred_rf
  }
  
  predicoes_rf = rowSums(matriz_rf)/n_estimadores
  return(predicoes_rf)
}

timerfbeta_inic = Sys.time()


y_pred_rf1 = rf_betareg(xtreino=treinobf[, -which(names(treinobf) == "tx_benbf")],
                        ytreino=treinobf[, "tx_benbf"],
                        xteste=testebf[, -which(names(testebf) == "tx_benbf")],
                        n_estimadores=100,
                        n_amostra=dim(treinobf)[1],
                        n_features=25)

timerfbeta_fim = Sys.time()


paste("Tempo de treinamento: ", (timerfbeta_fim-timerfbeta_inic)*60)

## MAE
MAE(y=testebf$tx_benbf,ypred=y_pred_rf1)

## MSE
MSE(y=testebf$tx_benbf,ypred=y_pred_rf1)

## RMSE
sqrt(MSE(y=testebf$tx_benbf,ypred=y_pred_rf1))


## R2: Coeficiente de Determinação
R2(y=testebf$tx_benbf,ypred=y_pred_rf1) 


n_estimators = seq(50,500,by=50)
n_features = c(5,10,15,25)

matrix_cv_RF = matrix(data=0,nrow=length(n_estimators),ncol=length(n_features))

for(i in 1:length(n_estimators)){
  for(j in 1:length(n_features)){
    y_pred_rf = rf_betareg(xtreino=treinobf[, -which(names(treinobf) == "tx_benbf")],
                            ytreino=treinobf[, "tx_benbf"],
                            xteste=testebf[, -which(names(testebf) == "tx_benbf")],
                            n_estimadores=n_estimators[i],
                            n_amostra=dim(treinobf)[1],
                            n_features=n_features[j])
    
    matrix_cv_RF[i,j] = MSE(y=testebf$tx_benbf,ypred=y_pred_rf)
    print(paste("par: (",n_estimators[i],",",n_features[j],")"))
  }
}

cv_RF = as.data.frame(matrix_cv_RF)
colnames(cv_RF) = c("f5","f10","f15","f25")
cv_RF$estimators = seq(50,500,by=50)

gf5 = ggplot(data=cv_RF, aes(x=estimators, y=f5, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title="k = 5") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

gf10 = ggplot(data=cv_RF, aes(x=estimators, y=f10, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title="k = 10") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

gf15 = ggplot(data=cv_RF, aes(x=estimators, y=f15, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title="k = 15") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

gf25 = ggplot(data=cv_RF, aes(x=estimators, y=f25, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title="k = 25") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

grid.arrange(gf5,gf10,gf15,gf25,ncol=2)

## ----
## Algoritmo de Boosing para a Regressão Beta
## Gradiente Boosting
## ----

## biblioteca betaboost implementa
## tem que puxar do github

library("devtools")

#install_github("boost-R/betaboost")
library("betaboost")

timeboost_inic <- Sys.time()

betaboosting = betaboost(tx_benbf ~.,
                data=treinobf, iterations = 500,
                form.type = "betaboost")

timeboost_fim <- Sys.time()

timeboost_fim-timeboost_inic

#summary(betaboosting)

y_predbetaboost1 = predict(betaboosting,testebf)

## nomalização da resposta para o intervalo (0,1)
y_predbetaboost1 = (y_predbetaboost1 - min(y_predbetaboost1))/(max(y_predbetaboost1) - min(y_predbetaboost1))

MSE(y=testebf$tx_benbf,ypred=y_predbetaboost1)
sqrt(MSE(y=testebf$tx_benbf,ypred=y_predbetaboost1))

MAE(y=testebf$tx_benbf,ypred=y_predbetaboost1)

R2(y=testebf$tx_benbf,ypred=y_predbetaboost1)

## montar o estudo do betaboost para a avaliação do MSE

bootsample = seq(50,800,50)
MSEbetareg = NULL

for (c in 1:length(bootsample)){
  betaboosting = betaboost(tx_benbf ~.,
                           data=treinobf, iterations = bootsample[c],
                           form.type = "betaboost")
  
  y_predbetaboost1 = predict(betaboosting,testebf)
  
  ## nomalização da resposta para o intervalo (0,1)
  y_predbetaboost1 = (y_predbetaboost1 - min(y_predbetaboost1))/(max(y_predbetaboost1) - min(y_predbetaboost1))
  
  MSEbetareg[c] = MSE(y=testebf$tx_benbf,ypred=y_predbetaboost1)
  print(bootsample[c])
}

plot(1:length(bootsample),MSEbetareg)

#plot(matrix_cv_bag,type='b')
cv_betaboost = as.data.frame(MSEbetareg)
cv_betaboost$ests = bootsample 
colnames(cv_betaboost)[1] = "EQM"

ggplot(data=cv_betaboost, aes(x=ests, y=EQM, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  theme_minimal()

## ---
## Implementação do algoritmo de Gradient Boosting
## Para a Regressão Beta
## ---

## Passo 0: Predição inicial para F0
## F0 é a média das observações

f0 = rep(0,length(testebf$tx_benbf))
f = f0
#r = treinobf$tx_benbf

G = 20 ## número de preditores para o modelo
alpha = 0.25 ## learning rate do modelo

## dados para  o boosting
treinobfboost = treinobf


# matriz de armazenamento das predições
mboost = matrix(data=0,nrow=nrow(testebf),ncol=G)

timeboost_inic <- Sys.time()

resemp = treinobf$tx_benbf

enableJIT(3)
for (g in 1:G){
  
  resemp01 = (resemp-min(resemp))/(max(resemp)-min(resemp))
  resemp01[resemp01>0.99999] = 0.9999
  resemp01[resemp01<0.0001] = 0.0001
  
  treinobfboost$resemp01 = resemp01
  
  mdres = betareg(resemp01 ~ .-tx_benbf,
                 data=treinobfboost,
                 link="loglog")
  
  respred = predict(mdres,newdata=testebf)
  reshat = respred*(max(respred) - min(respred)) + min(respred)
  
  f = f + (alpha*reshat)
  resemp = resemp - (alpha*reshat)
  
  ## salvar a predição f
  mboost[,g] = reshat*alpha
  
  print(g)

}

timeboost_end <- Sys.time()

beep(sound = 2, expr = NULL)


## Predições para a variável resposta

predunscaled = rowSums(mboost)

## Precisamos padronizar os valores para [0,1].

predscaled = (predunscaled-min(predunscaled))/(max(predunscaled)-min(predunscaled))

hist(predscaled)


## MSE e RMSE
MSE(y=testebf$tx_benbf,ypred=predscaled)
sqrt(MSE(y=testebf$tx_benbf,ypred=predscaled))

## MAE
MAE(y=testebf$tx_benbf,ypred=predscaled)

## R2
R2(y=testebf$tx_benbf,ypred=predscaled)

## tempo
timeboost_end - timeboost_inic 


## estudo dos vaores de estimadores e dos alphas

GS = seq(50,500,50)
alphas = c(0.25,0.50,0.75,0.95)

matrix_cv_boost = matrix(0,nrow = length(GS), ncol = length(alphas))

enableJIT(3)
for (i in 1:length(GS)){
  for (j in 1:length(alphas)){
    ## Passo 0: Predição inicial para F0
    ## F0 é a média das observações
    
    f0 = rep(0,length(testebf$tx_benbf))
    f = f0
    #r = treinobf$tx_benbf
    
    G = GS[i] ## número de preditores para o modelo
    alpha = alphas[j] ## learning rate do modelo
    
    # matriz de armazenamento das predições
    mboost = matrix(data=0,nrow=nrow(testebf),ncol=G)

    
    resemp = treinobf$tx_benbf
    
    enableJIT(3)
    for (g in 1:G){
      
      resemp01 = (resemp-min(resemp))/(max(resemp)-min(resemp))
      resemp01[resemp01>0.99999] = 0.9999
      resemp01[resemp01<0.0001] = 0.0001
      
      
      treinobfboost$resemp01 = resemp01
      
      mdres = betareg(resemp01 ~ .-tx_benbf,
                      data=treinobfboost,
                      link="loglog")
      
      respred = predict(mdres,newdata=testebf)
      reshat = respred*(max(respred) - min(respred)) + min(respred)
      
      f = f + (alpha*reshat)
      resemp = resemp - (alpha*reshat)
      
      ## salvar a predição f
      mboost[,g] = reshat*alpha
      
    }
    
    ## Predições para a variável resposta
    
    predunscaled = rowSums(mboost)
    
    ## Precisamos padronizar os valores para [0,1].
    
    predscaled = (predunscaled-min(predunscaled))/(max(predunscaled)-min(predunscaled))

    ## MSE 
    matrix_cv_boost[i,j] = MSE(y=testebf$tx_benbf,ypred=predscaled)
    
    print(paste("par: (",i,",",j,")"))
  }
}

matrix_cv_boost
dim(matrix_cv_boost)

matrix_cv_boost[,1]

plot(1:10,matrix_cv_boost[,1],type="b")
plot(1:10,matrix_cv_boost[,2],type="b")
plot(1:10,matrix_cv_boost[,3],type="b")
plot(1:10,matrix_cv_boost[,4],type="b")

cv_boost = as.data.frame(matrix_cv_boost)
colnames(cv_boost ) = c("a25","a50","a75","a95")
cv_boost$estimators = seq(50,500,50)

ga25 = ggplot(data=cv_boost, aes(x=estimators, y=a25, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title=expression(alpha==0.25)) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ga50 = ggplot(data=cv_boost, aes(x=estimators, y=a50, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title=expression(alpha==0.50)) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ga75 = ggplot(data=cv_boost, aes(x=estimators, y=a75, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title=expression(alpha==0.75))+
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ga95 = ggplot(data=cv_boost, aes(x=estimators, y=a95, group=1)) +
  geom_line(linetype = "dashed")+
  geom_point(color='royalblue4',size=2.5) + 
  xlab("\n Number of estimators") +
  ylab("MSE \n")+
  labs(title=expression(alpha==0.95)) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

grid.arrange(ga25,ga50,ga75,ga95,ncol=2)


### Adicionar a performance do modelo linear

# mdlinear = lm(tx_benbf ~.,
#               data=treinobf)
# 
# 
# ypredlm = predict(mdlinear,testebf)
# 
# MAE(y=testebf$tx_benbf,ypred=ypredlm)
# MSE(y=testebf$tx_benbf,ypred=ypredlm)
# sqrt(MSE(y=testebf$tx_benbf,ypred=ypredlm))



