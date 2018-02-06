import numpy as np
import hypertools as hyp
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import RFE
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn import cross_validation
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import paired_distances
from sklearn.metrics import confusion_matrix

'''
Notes:
Python code written in Python 3.5.2 on windows 32 for "Ayasdi Data Scientist Challenge One"

References:
Hypertools: https://arxiv.org/pdf/1701.08290.pdf
'''

#Path for dataset
FILE_NAME = 'H:\\Python\\Ayasdi\\dataset_challenge_one 2018.tsv'

#helper function to plot summary stats from data
def plot_summary(data):

    '''
    Input of numpy array of r x c
    r: samples
    c: features
    '''
    
    #Calculate statistics for each variable for a summary plot
    Summary_stats = stats.describe(data)

    #lolims
    lowerlims = Summary_stats.minmax[0]

    #uplims
    upperlims = Summary_stats.minmax[1]

    #standard errors
    errors = np.sqrt(Summary_stats.variance)

    #configure and plot
    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.set(xlim=(0,data.shape[1]),
           ylim=(min(lowerlims), max(upperlims)), 
           xlabel='variable labels',
           ylabel='Statisic values',
           title='Summary of variables\' distributions')

    ax.errorbar(x=range(0,data.shape[1]), 
                y=Summary_stats.mean, 
                yerr=errors,  
                linestyle='None', 
                marker='.', fmt='.k', 
                ecolor='c', 
                elinewidth=0.3, 
                label='means')

    ax.plot(upperlims, color='m', linewidth=0.2, label='max')

    ax.plot(lowerlims, color='m', linewidth=0.2, label='min')

    return plt

#Load dataset
Dataset = np.genfromtxt(FILE_NAME, delimiter='\t', skip_header=1)

#find missing datapoints
print("1) Pre-processing missing data points:\n " + str(np.argwhere(np.isnan(Dataset))))

#Data pre-process
#Two variables were anomolous in that they had data points missing
#Use mean to fill values

from sklearn.preprocessing import Imputer, StandardScaler
imp = Imputer(strategy='mean')

Dataset = imp.fit_transform(Dataset)

#Separate classifer from dataset
y = np.array(Dataset[:,-1], dtype=np.int)
X = np.array(Dataset[:,:-1], dtype=np.float64)

#Centred data for PCA
Standardised_X = StandardScaler().fit_transform(X)

print("1) Shapes of X and y:\n" + str(X.shape) + ", " + str(y.shape))

#1) Explore data
plot_summary(X).show()

#Notable features of the data exploration of non standardised data:
#1) Means of variables mostly around zero
#2) Outliers of mean on the negative side
#3) Standard errors in cyan showing upside and downside variance asymmetry reflected in the mean
#4) Max/min values in magenta floored and capped at -2/+2

#1) create animation of classifiers with first 3 components using hypertools
hyp.plot(X, 'o', group=list(y), animate=True, legend=True, save_path='H:\\Python\\Ayasdi\\out\\PCA3Danimation2.mp4')

#Observations
#1. Data is plotted with classifiers represented by two colours
#2. Each frame of animation displays a portion of the data
#3. Displayed portion advances through rows of data
#4. This may held to identify any temporal nature of the dataset
#5. Zero labelled data features towards end of the dataset
#6. Anomolous distributions - nothing further to add at the moment

#2) Present a 2 dim PCA analysis with class
#helper function for 2 dimension PCA and return scatterplot
def PCA_scatter(model, X, Y):
    '''
    Needs SKLearn module for PCA and matplotlib for scatter plot
    X
        r: samples
        c: features    
    Y classes
    Returns plot
    '''
    
    projected = model.fit_transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x = projected[:, 0]
    y = projected[:, 1]
    classes = Y
    unique = list(set(Y))
    colours = ['red', 'blue']

    for i, u in enumerate(unique):
        xi = [x[j] for j  in range(len(x)) if classes[j] == u]
        yi = [y[j] for j  in range(len(x)) if classes[j] == u]
        plt.scatter(xi, yi, c=colours[i], label=str(u))

    #ax.scatter(projected[:, 0], projected[:, 1], c=Y, edgecolor='none', label=Y)
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    ax.legend(scatterpoints=1)
    
    # Percentage of variance explained for each components
    print('2) Explained variance ratios for PCA for the first two components: %s' % str(pca.explained_variance_ratio_))

    return plt

pca = PCA(2)

PCA_scatter(pca, Standardised_X, y).show()

#Observations
#1. As with the animation, we can see zero classified clustering is captured by these 2 components
#2. Response variables classified as one interspersed and showing more variability with zero classified responses

#3a) calculate a statistic for each variable and explain usage

# split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Given binary classification, logistic regression is a natural choice
classifier = LogisticRegression()

# create the recursive feature elimination model and select attributes
rfe = RFE(classifier, 10) 
rfe = rfe.fit(X_train, y_train)

# summarize the selection of the attributes
print("3a) Logistic Regression: Variable column indices selected", np.where(rfe.support_ == True))
print("3a) Logistic Regression: Intercept and coefs\n", rfe.estimator_.intercept_, rfe.estimator_.coef_)
#Comments
#1) RFE tests differing subsets which is better than univariate analysis as it can potentially spot variable correlation
#2) Variable represented by column 1089 (i.e. variable 1090) has the highest co-efficient so has single highest explanatory power


#top row is variable label indexed at zero, second row is co-eff
factors = np.vstack((np.where(rfe.support_ == True), rfe.estimator_.coef_))
#make co-effs absolute for ranking of explanatory power
factors[1,:] = np.abs(factors[1,:])
#sort by abs co-eff
factors = factors[:, np.argsort(factors[1])]

#set up bar plot of top ten coefficients
labels = factors[0].astype(int)
y_pos = np.arange(len(labels))
coef = factors[1]
plt.barh(y_pos,coef, align='center', alpha=0.5)
plt.yticks(y_pos, labels)
plt.xlabel('Logistic Regression: Absolute value of coefficient')
plt.ylabel('Logistic Regression: Column index of variable')
plt.title('Logistic Regression: Top ten logistic regression weights')
plt.show()

#test model
scores = cross_val_score(rfe, X_train, y_train, scoring='f1')
print("3a) f1", np.mean(scores), scores)

#Observations for 3)
#1) Test model for prediction power using F1 stat
#2) Precision = fraction of data classified as 1 that are actually 1 (TP/(TP+FP))
#3) Recall = fraction of data classified as 1 that were correctly classified as 1 (TP/(TP+FN))
#4) F1 measures harmonic mean of of precision and recall, penalises imbalanced precision and recall.
#5) Low f1 score shows logistic model of the selected 10 variables is a poor model

#3) Calculate statistic for every variable that describes its relationship with PCA1

# https://stackoverflow.com/questions/1730600/principal-component-analysis-in-python
# singular value decomposition factorises your data matrix such that:
# 
#   X = U*S*V.T     (where '*' is matrix multiplication)
# 
# * U and V are the singular matrices, containing orthogonal vectors of
#   unit length in their rows and columns respectively.
#
# * S is a diagonal matrix containing the singular values of M - these 
#   values squared divided by the number of observations will give the 
#   variance explained by each PC.
#
# * if X is considered to be an (observations, features) matrix, the PCs
#   themselves would correspond to the rows of S^(1/2)*V.T. if X is 
#   (features, observations) then the PCs would be the columns of
#   U*S^(1/2).
#
# * since U and V both contain orthonormal vectors, U*V.T is equivalent 
#   to a whitened version of X.

U, s, Vt = np.linalg.svd(Standardised_X, full_matrices=False)
V = Vt.T

# PCs are already sorted by descending order 
# of the singular values (i.e. by the
# proportion of total variance they explain)

# if we use all of the PCs we can reconstruct the noisy signal perfectly
S = np.diag(s)
Xhat = np.dot(U, np.dot(S, V.T))
print("3b) Using all PCs, MSE = " + str(np.mean((X - Xhat)**2)))

# if we use only the first n_1 PCs the reconstruction is less accurate
# Xhat2 represents the variable representation of just PCA1 alone
n_1 = 1
Xhat2 = np.dot(U[:, :n_1], np.dot(S[:n_1, :n_1], V[:,:n_1].T))
print("3b) Using first " + str(n_1) + " PC, MSE = " + str(np.mean((X - Xhat2)**2)))

print("3b) weight of PCA1 in all variables is " + str(s[0]))

#calculate the column-wise distances between projected variable (Xhat2) and X
#I have chosen this metric to as the measure of the difference between the
#PCA1 projection to the original variables

dists = paired_distances(Xhat2.T, Standardised_X.T, metric='euclidean')
labels = range(0,dists.shape[0])
dists_with_labels = np.vstack((labels, dists))
dists_with_labels = dists_with_labels[:, np.argsort(dists_with_labels[1])]

#take top 10 for ease of visualisation
dists_with_labels = dists_with_labels[:,:10]

#set up bar plot of top ten coefficients
labels = dists_with_labels[0].astype(int)
y_pos = np.arange(len(labels))
coef = dists_with_labels[1]
plt.barh(y_pos,coef, align='center', alpha=0.5)
plt.yticks(y_pos, labels)
plt.xlabel('Distances')
plt.ylabel('Column index of variable')
plt.title('Comparison of distances of each variable with PCA1 projection')
plt.show()

print("3b) Variable " + str(dists_with_labels[0,0]) + " is the closest variable to its PCA1 projection")

#4) Create classifier model

# split centred dataset
X_train, X_test, y_train, y_test = cross_validation.train_test_split(Standardised_X, y, test_size=0.33, random_state=0)

#find index of components to explain some % of variance
pca = PCA().fit(Standardised_X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('4) number of components')
plt.ylabel('4) cumulative explained variance');
n = next(x[0] for x in enumerate(np.cumsum(pca.explained_variance_ratio_)) if x[1] > 0.75)
print("4) Number of PCA components required for 75% variance explanation: " + str(n))

clf = Pipeline([('pca', PCA(n_components=n, whiten=True)),
                ('svm', LinearSVC(C=1.0))])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("4) SVM+PCA confusion matrix:\n", confusion_matrix(y_pred, y_test))
scores = cross_val_score(clf, X_train, y_train, scoring='f1')
print("4) SVM+PCA F1 average score and for each k fold:\n", np.mean(scores), scores)

#Observations
#1) Combining PCA with SVM model improves F1 score over logistic regression model
#2) PCA helps reduce dimensionality for SVM model - have used 75% cut off for variance explanation arbitrarily
#3) Cross validation was done with 33% for both confusion matrix and F1 score
#4) With more time, a grid search and hyperparameter tuning could improve results
