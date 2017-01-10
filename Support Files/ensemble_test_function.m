function [Y_hat svm, logistic, logitboost, gentleboost, naive, pcalogistic, cs4vmM, label, precision, confusion, pweight, nweight, ens, ensprecision, enslabel] = ensemble_test_function()

load('words_train_mod.mat');
Xtrain = Xmod(1751:end,:);
Ytrain = Ymod(1751:end,:);
Xtest = Xmod(1:1750,:);
Ytest = Ymod(1:1750,:);


%% Support Vector Machine

[svm, label.svm, precision.svm, confusion.svm, pweight.svm, nweight.svm]...
= svm_predict(Xtrain, Ytrain, Xtest, Ytest)


%% Logistic Regression
[logistic, label.logistic, precision.logistic, confusion.logistic, pweight.logistic, nweight.logistic]...
    = logistic_predict(Xtrain, Ytrain, Xtest, Ytest)


%% LogitBoost
[logitboost, label.logitboost, precision.logitboost, confusion.logitboost, pweight.logitboost, nweight.logitboost]...
    = logitboost_predict(Xtrain, Ytrain, Xtest, Ytest)


%% Naive Bayes
[naive, label.naive, precision.naive, confusion.naive, pweight.naive, nweight.naive]...
    = naivebayes_predict(Xtrain, Ytrain, Xtest, Ytest)


%% Gentle Boost
[gentleboost, label.gentleboost, precision.gentleboost, confusion.gentleboost, pweight.gentleboost, nweight.gentleboost]...
    = gentleboost_predict(Xtrain, Ytrain, Xtest, Ytest)

%% Semi Supervised Model: PCAed Logistic Regression

load('C:\1. Sakthi\Courses\3. CIS 520 Machine Learning\Project\temp\ensemble\data\Xfull.mat')
[U ,~, ~] = svds(sparse(Xfull), 232);

[pcalogistic, label.pcalogistic, precision.pcalogistic, confusion.pcalogistic, pweight.pcalogistic, nweight.pcalogistic]...
    = logistic_predict(U(1751:5025,:), Ytrain, U(1:1750,:), Ytest)


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
    = naivebayes_predict([Xtrain; Xun], [Ytrain; Yun], Xtest, Ytest)

algs = fieldnames(precision);
for i = 1:numel(algs)
    y(i) = 1-precision.(algs{i});
end
bar(y);
set(gca,'XTickLabel', algs);
xlabel('Algo');
ylabel('Test Error');
title('Algorithm Comparisions');

print -djpeg -r72 plot_1.jpg;

%% Ensemble
Xens = [label.cs4vm, label.pcalogistic, label.gentleboost,  label.naive, label.logistic, label.svm ];
Yens = Ytest;

% Naive Stuff on weights
ens.naive = fitcnb(Xens(1:1250,:), Xens(1:1250))%,'DistributionNames','mn');
enslabel.naive = predict(ens.naive, Xens(1251:end,:));
ensprecision.naive = mean(Yens(1251:end) == enslabel.naive)

% logistic
ens.logistic = train(Yens(1:1250), sparse(Xens(1:1250,:)), ['-s 0', 'col']);
enslabel.logistic = predict(Yens(1251:end), sparse(Xens(1251:end,:)), ens.logistic, ['-q', 'col']);
ensprecision.logistic = mean(Yens(1251:end) == enslabel.logistic)

% Neural Nets
ens.nn = feedforwardnet([7 8]);
ens.nn = train(ens.nn, Xens(1:1250,:)', Yens(1:1250)');
enslabel.nn = ens.nn(Xens(1251:end,:)');
enslabel.nn = round(enslabel.nn)';
ensprecision.nn = mean(Yens(1251:end) == enslabel.nn)

algs = fieldnames(ensprecision);
for i = 1:numel(algs)
    y(i) = 1-ensprecision.(algs{i});
end
bar(y);
set(gca,'XTickLabel', algs);
xlabel('Algorithm');
ylabel('Test Error');
title('Ensemble Algorithm Comparisions');

print -djpeg -r72 Ensemble_test_error.jpg;


% Using the pweights and nweights
names = fieldnames(label);
for i=1:length(names)
    temp = getfield(label, names{i});
    
    temp(temp==1) = getfield(pweight, names{i});
    temp(temp==0) = -getfield(nweight, names{i});
    t(:,i)= temp;
end

% Naive Stuff on weights
ens.naive = fitcnb(t(1:1250,:), Yens(1:1250));
enslabel.naive = predict(ens.naive, t(1251:end,:));
ensprecision.naive = mean(Yens(1251:end) == enslabel.naive)

% logistic
ens.logistic = train(Yens(1:1250), sparse(t(1:1250,:)), ['-s 0', 'col']);
enslabel.logistic = predict(Yens(1251:end), sparse(t(1251:end,:)), ens.logistic, ['-q', 'col']);
ensprecision.logistic = mean(Yens(1251:end) == enslabel.logistic)

% Neural Nets
ens.nn = feedforwardnet([7 8]);
ens.nn = train(ens.nn, t(1:1250,:)', Yens(1:1250)');
enslabel.nn = ens.nn(t(1251:end,:)');
enslabel.nn = round(enslabel.nn)';
ensprecision.nn = mean(Yens(1251:end) == enslabel.nn)


xx = [enslabel.nn enslabel.logistic enslabel.naive];
Y_hat = sum(xx');
Y_hat(Y_hat<=1)=0;
Y_hat(Y_hat>=2)=1;
Y_hat = Y_hat';

algs = fieldnames(ensprecision);
for i = 1:numel(algs)
    y(i) = 1-ensprecision.(algs{i});
end
bar(y);
set(gca,'XTickLabel', algs);
xlabel('Algorithm');
ylabel('Test Error');
title('Ensemble Algorithm Comparisions - On weighted outputs');

print -djpeg -r72 Ensemble_wighted_test_error.jpg;

% Stepwise Regression
ens.stepwise = stepwisefit(Xens(1:1250,:), Yens(1:1250));

%% Wait forrr it!!! Ensemble of Ensemblesss - (Not used for final model)

XensEns = [enslabel.naive enslabel.logistic enslabel.nn];
YensEns = Yens(1251:1750);

% Voting
Ensenslabel.vote = sum(XensEns');
Ensenslabel.vote(Ensenslabel.vote<=1)=0;
Ensenslabel.vote(Ensenslabel.vote>=2)=1;
Ensenslabel.vote = Ensenslabel.vote';
Ensensprecision.vote = mean(YensEns == Ensenslabel.vote)

% Naive Bayes
Ensens.naive = fitcnb(XensEns(1:400,:), YensEns(1:400));
Ensenslabel.naive = predict(Ensens.naive, XensEns(401:end,:));
Ensensprecision.naive = mean(YensEns(401:end) == Ensenslabel.naive)

% Logistic
Ensens.logistic = train(YensEns(1:400), sparse(XensEns(1:400,:)), ['-s 0', 'col']);
Ensenslabel.logistic = predict(YensEns(401:end), sparse(XensEns(401:end,:)), Ensens.logistic, ['-q', 'col']);
Ensensprecision.logistic = mean(YensEns(401:end) == Ensenslabel.logistic)

algs = fieldnames(Ensensprecision);
for i = 1:numel(algs)
    y(i) = 1-Ensensprecision.(algs{i});
end
bar(y);
set(gca,'XTickLabel', algs);
xlabel('Algorithm');
ylabel('Test Error');
title('Ensemble of Ensemble');

print -djpeg -r72 Ensembleof_Ensemble.jpg;

