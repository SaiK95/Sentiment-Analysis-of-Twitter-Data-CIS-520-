function [model, label, precision, confusion, pweight, nweight] = logitboost_predict(Xtrain, Ytrain, Xtest, Ytest)

t = templateTree;
model = fitensemble(Xtrain,Ytrain,'LogitBoost',750, t);
label = predict(model, Xtest);
precision = mean(Ytest == label);
confusion = confusionmat(Ytest, label);

conf = confusionmat(Ytest(1:1250), label(1:1250));
pweight = conf(2,2)/sum(conf(2,:));
nweight = conf(1,1)/sum(conf(1,:));

end