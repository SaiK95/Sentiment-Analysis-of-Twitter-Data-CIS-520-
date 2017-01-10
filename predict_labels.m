function [Y_hat] = predict_labels(word_counts, cnn_feat, prob_feat, color_feat, raw_imgs, raw_tweets)
% Inputs:   word_counts     nx10000 word counts features
%           cnn_feat        nx4096 Penultimate layer of Convolutional
%                               Neural Network features
%           prob_feat       nx1365 Probabilities on 1000 objects and 365
%                               scene categories
%           color_feat      nx33 Color spectra of the images (33 dim)
%           raw_imgs        nx30000 raw images pixels
%           raw_tweets      nx1 cells containing all the raw tweets in text
% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 0 for sad)

addpath ./liblinear

load('ens4.mat')
load('weight.mat')
load('words_train_mod.mat')
load('gentleboost.mat')
load('logistic.mat')
load('naive.mat')
load('svm.mat')
load('cs4vm.mat')
load('logitboost.mat')


test_y = zeros(size(word_counts,1),1);

label.svm = predict(svm, word_counts);
label.logistic = predict(test_y, sparse(word_counts), logistic, ['-q', 'col']);
label.logitboost = predict(logitboost, word_counts);
% PCAed Logsitic
[U ,~, ~] = svds(sparse([Xmod; word_counts]), 232);
model = train(Ymod, sparse(U(1:5025,:)), ['-s 0', 'col']);
label.pcalogistic = predict(test_y, sparse(U(5026:end,:)), model, ['-q', 'col']);
label.cs4vm = predict(cs4vm, word_counts);
label.gentleboost = predict(gentleboost, word_counts);
label.naive = predict(naive, word_counts);


% % PCAed Logsitic
% [U ,~, ~] = svds(sparse([Xmod; word_counts]), 232);
% model = train(Ymod, sparse(U(1:5025,:)), ['-s 0', 'col']);
% label.pcalogistic = predict(test_y, sparse(U(5026:end,:)), model, ['-q', 'col']);
% 
% Xens = [label.cs4vm, label.pcalogistic, label.gentleboost,  label.naive, label.logitboost, label.logistic, label.svm ];

names = fieldnames(label);
for i=1:length(names)
    temp = getfield(label, names{i});    
    temp(temp==0) = getfield(pweight, names{i});
    temp(temp==1) = -getfield(nweight, names{i});
    t(:,i)= temp;
end

enslabel.naive = predict(ens.naive, t);
enslabel.logistic = predict(test_y, sparse(t), ens.logistic, ['-q', 'col']);
enslabel.nn = ens.nn(t');
enslabel.nn = round(enslabel.nn)';

xx = [enslabel.nn enslabel.logistic enslabel.naive];
Y_hat = sum(xx');
Y_hat(Y_hat<=1)=0;
Y_hat(Y_hat>=2)=1;
Y_hat = Y_hat';

end