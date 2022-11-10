# What's the best dimensionality reduction technique and classifier for different datasets?
## VARIANCE RATIO
### IRIS: 
Variance Ratio of IRIS Dataset: [0.92461872, 0.05306648]
As seen below, Setosa has the highest variance, while Versicolor and Virginicia have very low variance. Since there is not a lot of variety (1 really high, 2 very low) there is a risk our data can overfit. If the PCs had more variety it would prevent overfit.
### Indian Pines
Variance Ratio of Indian Pines Dataset: [0.95794497, 0.03918137]
Indian Pines has over 10 PCs but only the first two have a variance ratio. PC1 has a very high variance ratio while PC2 has a low variance ratio. Again, since there is not a lot of variety (1 really high, 1 very low) there is a risk our data can overfit. If there were more PCs that had more variety it would prevent overfit.

# PCA, LDA, and no diminsionality reduction
## IRIS
### Clustering
As seen below, both LDA and PCA did fairly well at clustering classes. Both did best at clustering “setosa”, while “versicolor” and “virginica” had some overlap. To me it appears that LDA did a slightly better job. LDA uses dimensionality reduction to cluster while PCA uses principal components to cluster.

### Cross Validation Accuracy and Training Accuracy
**PCA**
As you can see above, SVM-Poly and kNN do the best as the % of Training increases, and did best overall. SVM-Linear and Logistic Regression have a huge dip at 60% of Training, perhaps there was an issue with the way the training data was divided.In both Indian Pines and IRIS, SVM-Poly does the best for PCA.

**LDA**
As seen above in, Naive Bayes performed extremely well compared to the others. Naive Bayes begins at 95-100% accuracy, showing that it only needs 10-20% training data to accurately classify. Overfitting can occur if the model is trained too much, so in this case it would be better for Naive Bayes to only be trained about 20%. SVM-RBF and SVM-Poly did about equally well for both Indian Pines and IRIS’s LDA.

**No dimensionality reduction**
As seen above, SVM - Poly did poorly until about 70% training, but after that jump it was stagnant. Previously, it performed best for PCA of IRIS and Indian Pines.  

## Indian Pines
As seen below, PCA did not do a good job of clustering classes, while LDA did a much better job. With PCA all the classes are overlapping and there are no distinct clusters. There are also significant outliers (blue and red) on the other side of the chart from its larger cluster. With LDA, although there is a lot of overlap, it is much better than PCA. In LDA there are distinct groups (see red), and the clusters are more defined even where there is overlap. LDA uses dimensionality reduction to cluster while PCA uses principal components to cluster.
#### Clustering
As seen below, PCA did not do a good job of clustering classes, while LDA did a much better job. With PCA all the classes are overlapping and there are no distinct clusters. There are also significant outliers (blue and red) on the other side of the chart from its larger cluster. With LDA, although there is a lot of overlap, it is much better than PCA. In LDA there are distinct groups (see red), and the clusters are more defined even where there is overlap. LDA uses dimensionality reduction to cluster while PCA uses principal components to cluster.

#### Cross Validation Accuracy and Training Accuracy
**PCA**
As seen above, SVM-Poly does the best while Naive Bayes does extremely poorly. In both Indian Pines and IRIS, SVM-Poly does the best for PCA.

**LDA**
As seen above, SVM- RBF did the best, while Naive Bayes did extremely badly. While Naive Bayes did really well with IRIS’ LDA, that was not the case for Indian Pines’ LDA. SVM-RBF and SVM-Poly did about equally well for both Indian Pines and IRIS’s LDA.
**No dimensionality reduction**
As seen above, Random Forest and SVM-Poly did extremely well while Naive Bayes did extremely badly. SVM-Poly was really good for both IRIS and Indian Pines LDAs.  
# Conclusion
Best Classifier: SVM-Poly, it performed best for IRIS LDA,Indian Pines LDA, and non-reduced Indian Pines.  
Worst Classifiers: Naive Bayes and Logistic Regression
--- Naive Bayes performed worst for all Indian Pines datasets: Indian Pines PCA, Indian Pines LDA, and non-reduced Indian Pines
--- Logistic Regression performed worst for all IRIS datasets: IRIS PCA, IRIS LDA, and non-reduced IRIS
