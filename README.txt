Console outputs

1) Pre-processing missing data points:
 [[ 128   74]
 [ 155 1289]]

1) Shapes of X and y:
(272, 1553), (272,)

2) Explained variance ratios for PCA for the first two components: [ 0.12176881  0.05541132]

3a) Logistic Regression: Variable column indices selected (array([ 119,  508,  545,  548,  962, 1085, 1089, 1154, 1330, 1346], dtype=int32),)
3a) Logistic Regression: Intercept and coefs
 [-0.85762501] [[-1.35851627 -1.34306753  1.50839532 -1.44451659  1.71698144  1.53002778
   1.97664116 -1.47444799  1.2719607   1.36411043]]

3a) f1 0.239819004525 [ 0.41176471  0.07692308  0.23076923]

3b) Using all PCs, MSE = 0.511723084773
3b) Using first 1 PC, MSE = 0.173110217652
3b) weight of PCA1 in all variables is 226.797471106

3b) Variable 1240.0 is the closest variable to its PCA1 projection

4) Number of PCA components required for 75% variance explanation: 60

4) SVM+PCA confusion matrix:
 [[57 13]
 [12  8]]

4) SVM+PCA F1 average score and for each k fold:
 0.459805959806 [ 0.64864865  0.5         0.23076923]