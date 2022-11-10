################
# IRIS and Indian Pines PCA and LDA Comparison
# CAROLYN SAVAS
# OCTOBER 2022
################
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
import scipy.io as sci
from scipy.io import loadmat
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#########################
# INDIAN PINES PCA & LDA
#########################
def pines_PCA_LDA():
    df = sci.loadmat(r"C:\\Users\\carol\\Desktop\\SCHOOL\\CS 488\\indianR.mat")
    x=np.array(df['X']) 
    bands, samples=x.shape

    gth_mat =sci.loadmat(r"C:\\Users\\carol\\Desktop\\SCHOOL\\CS 488\\indianR.mat") 
    gth_mat ={i:j for i, j in gth_mat.items() if i[0] != '_'} 
    pines_gt = pd.DataFrame({i: pd.Series(j[0]) for i, j in gth_mat.items()})
    pines_gth = np.array(df['gth'])
    n=[] 
    ind=[] 
    for i in range(bands):
        n.append(i+1) 
    for i in range(bands):
        ind.append('band'+str(n[1]))
        
    features = ind
    
    ##########
    # PCA
    ##########
    # data prepping
    scaler_model = MinMaxScaler()
    scaler_model.fit(x.astype(float))
    pines_x=scaler_model.transform(x)
    pines_pca = PCA(n_components=2)
    pines_principalComponents = pines_pca.fit_transform(pines_x)
    # transform data
    pines_ev = pines_pca.explained_variance_ratio_
    print(
        "explained variance ratio (first two components): %s"
        % str(pines_pca.explained_variance_ratio_)
    )
    pines_principal_df = pd.DataFrame(data = pines_principalComponents, columns = ['PC-1', 'PC-2'])

    pines_xT = pines_x.transpose()
    pines_x_pca = np.matmul(pines_xT, pines_principalComponents)
    pines_x_pca.shape

    pines_x_pca_df = pd.DataFrame(data = pines_x_pca, columns = ['PC-1', 'PC-2'])
    pines_x_pca_df = pd.concat([pines_x_pca_df, pines_gt], axis = 1)
    ######################
    # Plot Variance Ratio 
    ######################
    plt.bar([1,2],list(pines_ev*100),label='Principal Components', color='b')
    plt.legend()
    plt.xlabel('Principal Components')
    pc=[]
    pc.append('PC1')
    pc.append('PC2')
    plt.xticks([1,2],pc,fontsize=8,rotation=30)
    plt.ylabel('Variance Ratio')
    plt.title('Variance Ratio of Indian Pines Dataset')
    plt.savefig("PINES_plt_var_ratio.png")
    plt.close()
    #############
    # PCA PLOT
    #############
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('PC-1', fontsize=15)
    ax.set_ylabel('PC-2', fontsize=15)
    ax.set_title('PCA on Indian Pines Dataset', fontsize = 20)
    class_num = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    colors = ['r','g','b','y','m','c','k','r','g','b','y','m','c','k','b','r']
    markerm = ['o','o','o','o','o','o','o','+','+','+','+','+','+','+','*','*']
    for target, color, m in zip(class_num, colors, markerm):
        indicesToKeep = pines_x_pca_df['gth'] == target
        ax.scatter(pines_x_pca_df.loc[indicesToKeep, 'PC-1'], pines_x_pca_df.loc[indicesToKeep, 'PC-2'], c = color, marker = m, s = 9)
        ax.legend(class_num)
        ax.grid()
    plt.savefig("PINES_pca.png")
    plt.close()
    #######
    # LDA
    #######
    pines_gt_reformat = np.array(pines_gth)

    pines_lda = LinearDiscriminantAnalysis(n_components=2)
    pines_discriminant = pines_lda.fit_transform(pines_x.transpose(), pines_gt_reformat.transpose().ravel())

    # transform data
    pines_x_lda = np.matmul(pines_x, pines_discriminant)
    pines_x_lda.shape

    pines_gt_reformat = pd.DataFrame(data=pines_gth.transpose(), columns= ['gth'])

    pines_x_lda_df = pd.DataFrame(data = pines_discriminant, columns = ['D-1', 'D-2'])
    pines_x_lda_df = pd.concat([pines_x_lda_df, pines_gt_reformat], axis = 1)
    
    ###########
    # LDA PLOT
    ###########
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('D-1', fontsize=15)
    ax.set_ylabel('D-2', fontsize=15)
    ax.set_title('LDA on Indian Pines Dataset', fontsize = 20)
    class_num = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    colors = ['r','g','b','y','m','c','k','r','g','b','y','m','c','k','b','r']
    markerm = ['o','o','o','o','o','o','o','+','+','+','+','+','+','+','*','*']
    for target, color, m in zip(class_num, colors, markerm):
        indicesToKeep = pines_x_lda_df['gth'] == target
        ax.scatter(pines_x_lda_df.loc[indicesToKeep, 'D-1'],
                pines_x_lda_df.loc[indicesToKeep, 'D-2'],
                c = color, marker = m, s = 9)
    ax.legend(class_num)
    ax.grid()
    plt.savefig("PINES_lda.png")
    plt.close()
    X_lda = pines_discriminant
    X_pca = pines_x_pca
    y= pines_gth.transpose().ravel()
    x = pines_x.transpose()
    return x, X_lda, X_pca, y

##################
# IRIS PCA & LDA
##################
def iris_PCA_LDA():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    target_names = iris.target_names
    pca = PCA(n_components=2)
    X_pca = pca.fit(X).transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit(X, y).transform(X)

    # Percentage of variance explained for each components
    print(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )
    # Plot Variance Ratio
    plt.bar([1,2],list(pca.explained_variance_ratio_*100),label='Principal Components', color='b')
    plt.legend()
    plt.xlabel('Principal Components')
    pc=[]
    pc.append('Setosa')
    pc.append('Verisolor')
    pc.append('Virginica')
    plt.xticks([1,2],pc,fontsize=8,rotation=30)
    plt.ylabel('Variance Ratio')
    plt.title('Variance Ratio of Iris Dataset')
    plt.savefig("IRIS_plt_var_ratio.png")
    plt.close()
    
    # plot PCA
    plt.figure()
    colors = ["navy", "turquoise", "darkorange"]
    lw = 2
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("PCA of IRIS dataset")
    plt.savefig("IRIS_pca.png")
    plt.close()
    # plot LDA
    plt.figure()
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        plt.scatter(
            X_lda[y == i, 0], X_lda[y == i, 1], alpha=0.8, color=color, label=target_name
        )
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    plt.title("LDA of IRIS dataset")
    plt.savefig("IRIS_lda.png")
    plt.close()
    
    return X_lda, X_pca, y

##########################################
# helper function for naive bayes graphing
##########################################

def plot_learning_curve(classifier, X, y, steps=10, train_sizes=np.linspace(0.1, 1.0, 10), label="", color="r", axes=None):
    estimator = Pipeline([("scaler", MinMaxScaler()), ("classifier", classifier)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    train_scores = []
    test_scores = []
    train_sizes = []
    for i in range(0, X_train.shape[0], X_train.shape[0]//steps):
        if(i ==0):
            continue
        X_train_i = X_train[0:i, :]
        y_train_i = y_train[0:i]
        estimator.fit(X_train_i, y_train_i)
        train_scores.append(estimator.score(X_train_i, y_train_i)*100)
        test_scores.append(estimator.score(X_test, y_test)*100)
        train_sizes.append(i+1)
    if(X_train.shape[0] % steps !=0):
        estimator.fit(X_train, y_train)
        train_scores.append(estimator.score(X_train, y_train)*100)
        test_scores.append(estimator.score(X_test, y_test)*100)
        train_sizes.append(X_train.shape[0])
    if axes is None:
        _, axes = plt.subplot(2)
        
    train_s=np.linspace(10, 100, num=5)
    axes[0].plot(train_sizes, test_scores, "o-", color=color, label=label)
    axes[1].plot(train_sizes, test_scores, "o-", color=color, label=label)
    
    print("Training Accuracy of ", label, ": ", train_scores[-1], "%")
    print("Testing Accuracy of ", label, ": ", test_scores[-1], "%")
    print("")
    return plt    
#############################################################
# NAIVE BAYES ANALYSIS OF DIFFERENT CLASSIFICATION TECHNIQUES  
#############################################################   
def Naive_Bayes(X, y, test_percent, pngname):
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=test_percent, random_state=1, shuffle=True)
    if test_percent == 0.7: # produce extra graph at 70% test split
        models=[]
        models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC(gamma='auto')))
        
        # evaluate each model in turn
        results = []
        names = []
        for name, model in models:
            kfold = StratifiedKFold(n_splits=2, random_state=1, shuffle=True)
            cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
            results.append(cv_results)
            names.append(name)
            print('%s: %f %% (%f)' % (name, cv_results.mean()*100, cv_results.std()))
        pyplot.boxplot(results, labels=names)
        pyplot.title('Algorithm Comparison')
        pyplot.savefig("algo_compare_"+pngname)
        pyplot.close() 
    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    num_steps = 10
    classifier_labels = {'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=1), 'red'), 
                        'Random Forest': (RandomForestClassifier(random_state=1), "green"), 
                        'SVM - Linear': (SVC(kernel="linear", random_state=1), "blue"), 
                        'SVM - RBF': (SVC(kernel='rbf', random_state=1), "yellow"),
                        'SVM - Poly': (SVC(kernel='poly',random_state=1), "orange"),
                        'Gaussian Naive Bayes': (GaussianNB(), "lime")}
    for label in classifier_labels:
        classifier = classifier_labels[label][0]
        color = classifier_labels[label][1]
        plot_learning_curve(classifier, X, y, steps=num_steps, label=label, color=color, axes=axes)
          
    axes[0].set_xlabel("% of Training Examples")
    axes[0].set_ylabel("Overall Classification Accuracy")
    axes[0].set_title("Model Evaluation - Cross-Validation accuracy")
    axes[0].legend()
    
    axes[1].set_xlabel("% of Training Examples")
    axes[1].set_ylabel("Training/Recall Accuracy")
    axes[1].set_title("Model Evaluation - Training Accuracy")
    axes[1].legend()
    
    plt.savefig(pngname)
    plt.close()

###############
# MAIN PROGRAM
###############       
iris_X_lda, iris_X_pca, iris_y = iris_PCA_LDA()
pines_X, pines_X_lda, pines_X_pca, pines_y = pines_PCA_LDA()

test_percents = [0.9, 0.8, 0.7, 0.6, 0.5] # for training sizes ={10%, 20%, 30%, 40%, 50%}
#######       
# Iris
#######
# with dimensionality reduction - LDA
for test_percent in test_percents:
    print("IRIS W/ LDA: TEST PERCENT ", test_percent, "% ")
    Naive_Bayes(iris_X_lda, iris_y, test_percent, "IRIS_lda_bayes.png")
# with dimensionality reduction - PCA
for test_percent in test_percents:
    print("IRIS W/ PCA: TEST PERCENT ", test_percent, "% ")
    Naive_Bayes(iris_X_pca, iris_y, test_percent, "IRIS_pca_bayes.png")
# without dimensionality reduction
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
for test_percent in test_percents:
    print("IRIS W/O DIM REDUC: TEST PERCENT  ", test_percent, "% ")
    Naive_Bayes(X, y, test_percent, "IRIS_noDimReduc_bayes.png")

################        
# INDIAN PINES
################
# with dimensionality reduction - LDA 
for test_percent in test_percents:
    print("PINES W/ LDA: TEST PERCENT ", test_percent, "% ")
    Naive_Bayes(pines_X_lda, pines_y, test_percent, "PINES_lda_bayes.png")
# with dimensionality reduction - PCA
for test_percent in test_percents:
    print("PINES W/ PCA: TEST PERCENT ", test_percent, "% ")
    Naive_Bayes(pines_X_pca, pines_y, test_percent, "PINES_pca_bayes.png")
# without dimensionality reduction     
for test_percent in test_percents:
    print("PINES W/O DIM REDUC: TEST PERCENT  ", test_percent, "% ")
    Naive_Bayes(pines_X, pines_y, test_percent, "PINES_bayes.png")
