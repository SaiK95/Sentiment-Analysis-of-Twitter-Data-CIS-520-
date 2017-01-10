function [bestlabel, bestacc, bestmodel, precision]= tunepcalogistic(U,Ymod)
% 540 gave best results: 78.3, 235 gave best results: 78.15 
bestacc = 0.5;
bestc = 0;
for i=1:100
    c=150+ i;
    Xtrain= U(1501:5025,1:c);
    Ytrain=Ymod(1501:5025);
    Xtest= U(1:1500,1:c);
    Ytest= Ymod(1:1500);
    
    model = train(Ytrain, sparse(Xtrain), ['-s 0', 'col']);
    label = predict(Ytest, sparse(Xtest), model, ['-q', 'col']);
    precision(i) = mean(Ytest == label);
    
    if precision(i)>bestacc
        bestacc = precision(i);
        bestc= c;
        bestmodel=model;
        bestlabel=label;
    end
end
end
    