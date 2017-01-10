function [bestacc bestlambda] = svmtune(Xtrain, Ytrain, Xtest, Ytest)

i=0; bestacc=0.5;
c=0.00005;
while c<(5*10^-3)
    i=i+1;
    m = fitclinear(Xtrain,Ytrain,'Regularization','Ridge', 'Lambda', c,'Solver', 'lbfgs');
    c=c+0.00005;
    y_hat_train = predict(m, Xtrain);
    acc_train(i)= mean(y_hat_train == Ytrain);
    y_hat = predict(m, Xtest);
    acc(i)=mean(y_hat == Ytest);
    if acc(i)>bestacc
        bestacc = acc(i);
        bestlambda = c;
    end
end
end