function [precision]= tunethishit(Xtrain, Xtest, Ytrain, Ytest, Xun, Yun)

opt.maxiter = 50;
opt.c1 = 1;
opt.c2 = 3;
opt.gaussian = 0;
opt.cost = 2.6;

i=0;j=0;
for c1=[1:0.1:15]
     i=i+1;
     opt.cost=c1;
     for c2=[0.1:1:10]
         j=j+1;
         opt.c2=c2;
        Yun = zeros(size(Yun));
        [Yun,~,~] = cs4vm(Ytrain,Xtrain,Xun,opt);
        model.naive = fitcnb(Xtrain, [Ytrain(1:3525); Yun], 'DistributionNames','mn');
        label.naive = predict(model.naive, Xtest);
        precision(i) = mean(Ytest == label.naive)
     end
end
