function [model, label, precision, confusion, pweight, nweight] = logistic_predict(Xtrain, Ytrain, Xtest, Ytest)

addpath 'C:\1. Sakthi\Courses\3. CIS 520 Machine Learning\Project\temp\lib\liblinear'
model = train(Ytrain, sparse(Xtrain), ['-s 0', 'col']);
label = predict(Ytest, sparse(Xtest), model, ['-q', 'col']);
precision = mean(Ytest == label);
confusion = confusionmat(Ytest, label);

conf = confusionmat(Ytest(1:1250), label(1:1250));
pweight = conf(2,2)/sum(conf(2,:));
nweight = conf(1,1)/sum(conf(1,:));

end