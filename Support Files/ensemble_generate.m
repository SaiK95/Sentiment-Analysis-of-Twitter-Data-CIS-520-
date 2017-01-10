function [svm, logistic, cs4vmM, naive, logitboost, gentleboost] = ensemble_generate()
addpath ./liblinear
load('words_train_mod.mat');
Xtrain = Xmod;
Ytrain = Ymod;

%% SVM on un-modified data
svm = fitclinear(Xtrain,Ytrain,'Regularization','Ridge', 'Lambda', 0.003,'Solver', 'lbfgs');

%% Logistic Regression
logistic = train(Ytrain, sparse(Xtrain), ['-s 0', 'col']);

%% LogitBoost
t = templateTree;
logitboost = fitensemble(Xtrain,Ytrain,'LogitBoost',750, t);

%% Naive Bayes
naive = fitcnb(Xtrain, Ytrain, 'DistributionNames','mn');

%% Gentle Boost
t = templateTree;
gentleboost = fitensemble(Xtrain,Ytrain,'GentleBoost',500, t);

%% PCAed Logisitic: No model (Run directly on server)


%% Semi-Supervised Model: CS4VM

load('Unlab.mat')
opt.maxiter = 50;
opt.c1 = 1;
opt.c2 = 3;
opt.gaussian = 0;
opt.cost = 2.6;

tempY = [Ytrain];
tempY(tempY==0)=-1;
Yun = zeros(size(Yun));
[Yun,~,~] = cs4vm([tempY; Yun], [Xtrain; Xun], Xun, opt);

Yun(Yun==-1)=0;
[cs4vmM, label.cs4vm, precision.cs4vm, confusion.cs4vm, pweight.cs4vm, nweight.cs4vm]...
    = logistic_predict([Xtrain; Xun], [Ytrain; Yun], Xtrain, Ytrain)

%% 