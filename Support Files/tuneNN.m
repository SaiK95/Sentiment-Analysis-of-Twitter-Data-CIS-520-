function [nNeurons bestNeurons bestacc bestmodel] =  tuneNN(Xens, Yens)

c=1;
bestacc = 0.5;
for i=1:15
    for j=0:15
        for k=0:0
            for m=1:100
            if (j~=0) & (k~=0)
                ens.nn = feedforwardnet([i j k]);
                ens.nn = train(ens.nn, Xens(1:1250,:)', Yens(1:1250)');
                enslabel.nn = ens.nn(Xens(1251:end,:)');
                enslabel.nn = round(enslabel.nn)';
                ensprecision.nn = mean(Yens(1251:end) == enslabel.nn)
                nNeurons(c,:) = [i j k];
                c=c+1;
            elseif (j~=0) & (k==0)
                ens.nn = feedforwardnet([i j]);
                ens.nn = train(ens.nn, Xens(1:1250,:)', Yens(1:1250)');
                enslabel.nn = ens.nn(Xens(1251:end,:)');
                enslabel.nn = round(enslabel.nn)';
                ensprecision.nn = mean(Yens(1251:end) == enslabel.nn)
                nNeurons(c,:) = [i j k];
                c=c+1;
            elseif (j==0)
                ens.nn = feedforwardnet([i]);
                ens.nn = train(ens.nn, Xens(1:1250,:)', Yens(1:1250)');
                enslabel.nn = ens.nn(Xens(1251:end,:)');
                enslabel.nn = round(enslabel.nn)';
                ensprecision.nn = mean(Yens(1251:end) == enslabel.nn)
                nNeurons(c,:) = [i 0 0];
                c=c+1;
            else
            end
            if  ensprecision.nn > bestacc
                bestacc = ensprecision.nn;
                bestNeurons = [i,j,k];
                bestmodel = ens.nn;
            end
            end
        end
    end
end