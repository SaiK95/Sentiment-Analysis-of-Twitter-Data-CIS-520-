The project followed the given methodology:
1. We used tunethishit.m, tunepcalogistic.m, tuneNN.m and svmtune.m to tune different algorithms on the training data. 
2. We then used the ensemble_test_function to get cross-validation accuracies on the training data and validate the models.
3. We used ensemble_generate.m to generate the models for the testing function predict_labels.m

We have used two variables Xmod (5025 x 100000) and Ymod (5025 x 1). We obtained these from: a) X and Y from words_train.mat b) We labelled words_train_unlabeled.mat using happy and sad emoticons giving us 525 additional training observations. We will elaborate more on the text processing in our final report.

The algorithms we have used for ensembling are as follows:
a) Gentle Boost (uses gentleboost_predict.m)
b) LogitBoost (uses logitboost_predict.m)
c) Logistic Regression (uses logistic_predict.m)
d) Logistic Regression on PCA-ed data (runs on the submission server)
e) Naive Bayes (uses naivebayes_predict.m)
f) Semi-Supervised Support Vector Machines (uses cs4vm.m)
g) Support Vector Machines (uses svm_predict.m)

We obtained positive and negative weights for each of the above methods using their confusion matrices and weighed the methods appropriately and ensembled them using:
i) Logistic Regression
ii) Naive Bayes
iii) Neural Networks

We then predict the finals labels using majority vote of the three ensembles.

ALL THESE ARE IN THE SUPPORT FILES folder.



