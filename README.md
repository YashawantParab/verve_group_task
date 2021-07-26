# verve_group_task


### Question 1: Imagine that you were asked to use this dataset to build a classification model, with gender as the target. Look at the information we have given you and identify 3-5 potential problems you can see with the provided dataset that might make building a classification model difficult.
###### Answer: 
The question arises under the customer segmentation or mainly under Market segmentation. The unit of analysis represents the events that occur during the interaction with the application. In market segmentation, the target segment plays a vital role which aiming marketing efforts at specific groups of consumers. To monetize, a publisher needs to focus on demographic segments which are usually the most important criteria for identifying target markets.
The given dataset has a potential problem that needs to be considered to build a classification model more accurately. 
#### Imbalanced data:
In a classification problem, imbalanced data makes a major impact on predicting the target variable. The accuracy paradox of this imbalanced data in the classification problem states that the data have excellent accuracy (such as 99%), but the accuracy is only reflecting the underlying class distribution.
The frequency of clicks that are being studied is an example of an imbalanced dataset in this scenario. We have a dataset of 3700 rows in which 18 clicks were pressed and 3682 were not, clearly indicating the majority and minority classes. The majority class is a "No," whereas the minority class is a "Yes."
According to the study, if we train our model on such imbalanced data, it will be 90% accurate. That is because our model will always look for classes that contain 90% of the data and will intelligently predict such classes with high accuracy, even if the actual conclusion is incorrect.
#### Missing Data: 
Another issue that could arise is the presence of missing values. Training a machine learning model with a dataset that contains a large number of missing values can have a significant impact on the model's quality. The interactions between the User and the Device are depicted in the diagram above. In this case, the device name is a significant characteristic for gender prediction. There are some missing device names in the dataset, which may have an impact on estimating the customer's gender.
#### Lack of Information: 
I also observed that the offered dataset is lacking in information that could aid with gender prediction. In such a situation, behavioral segmentation can be extremely useful. It contains information such as surfing history, purchase patterns that can be used to determine gender.

### Question 2: Describe briefly how you would find the features that are likely to be the most important for your model.
##### Answer: 
In data science, feature engineering is the most important part as they focus on key principles of the data and their relationship with another attribute. The feature selection goal is to reduce the dimensionality of the dataset by removing unnecessary features, transforming existing features, and construct new features to improve the performance of a model.
From the given dataset we can elect features that are most relevant to build a gender prediction model. device_name, app_category, ad_category, and click are the main features that have influential information from which we can predict gender.
The given dataset has around 60% of missing values in device_name which affects the gender prediction model where this column signifies the main user information. Also, the Click attribute has an exceptionally high recurrence of getting overlooked in the advertisement.
Categorical feature selection.
There are many feature selection techniques in data science and statistics. We will discuss the following techniques which are relevant to the given dataset. 
#### Chi-Squared statistic: 
The chi-squared statistical theory test is an illustration of a test for independence between categorical factors. This Feature selection method is used for removing the variable which is not contributing to the model accuracy. This statistical approach determines whether there is a statically significant difference between the expected frequencies and the observed frequencies in multiple categories of a contingency table.
Considering the frequency tables of the given dataset, we need to perform the chi-Squared distribution with k degrees of freedom which results in the distribution of a sum of the square of K independent standard normal random variable.
#### Mutual information statistics: 
Mutual information (MI) is used to measure the mutual dependency of two random variables. This method quantifies the amount of information gained about one random variable through perceiving the other random variable. 
We can perform feature selection using mutual information on the given dataset and analyze the score. Larger values are better because those are the values that are more contributing to predicting features and assist to predict the gender of the user and advertise viewer.



### Question 3: Identify which model you would try first, and at least one advantage and disadvantage of this choice.
##### Answer: Modeling with selected features
After removing the unwanted and independent variables we can design a model to foresee the gender classification. There are different approaches where we can predict gender based on the selected features. 
	
	Naïve bias approach
	K-NN
	Support vector machine
	Random forest 

### Naïve Bayes Classifier theorem: 
Naïve Bayes algorithm is a supervised machine learning algorithm used for solving a classification problem. This classification algorithm uses a probabilistic approach and helps to build fast machine learning models that can make a quick prediction. 
The Naïve Bayes approach is the probabilistic method based on the Bayes’ theorem where c = (c1, c2) be the gender class, and F = (f1, f2, f3… fn) are features selected from feature engineering.  According to the Bayes’ theorem: 

$$x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}$$
\\(x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}\\)


$\prod_{x = a}^{b} f(x)$

\begin{equation}
X_{k} = A X_{k - 1} + B {u_k} + w_k
\label{eq: xk}
\end{equation}


P(c┤|F)=(P(c)P(F|c))/(P(F))
The naïve Bayes hypothesis is that:  
P͞͞(F│c)= ∏_(i=1)^n▒〖P͞͞(fi│c) 〗   
According to the naïve Bayes hypothesis, the probability is belonging to each class is like: 
P͞͞(F│c)=argmaxP(C=c│F)=argmaxP(C|c) ∏_(i=1)^n▒〖P͞͞(f〖=f〗_i│C=c) 〗   
Its basic assumption is that all features are independent of each other. In our binary classification problem, each viewer sample has only two predictions (Male, Female). 
##### Advantages
	It can be used for multiclass and binary classification.
	Perform well on the categorical input variable
	Need less training data
	The given problem has multi-class prediction using multiple variables, 
##### Disadvantages: 
	The naïve Bayes classifier model assumes all predictors or features are independent which rarely happens in real life. 
	Zero frequency can makes problems in productivity

### K-Nearest Neighbors (KNN): 
KNN is a supervised machine learning algorithm used for both classification and regression problems. It uses a Euclidean distance method to forms a majority vote between the K most similar instances. It uses a K hyperparameter that can be chosen by considering the trade-off between bias and variance.
##### Advantages: 
	KNN is a non-parametric classifier that makes no assumptions about the distribution of classes.
	This classifier is easy to implement and does not get impacted by the outliers.
##### Disadvantages:
	Defining a K value is difficult as it should work well with test data and training data.
	It is computationally extensive 

### Support Vector Machine (SVM):
Support vector machine is one of the powerful and most widely used machine learning algorithms that use kernel tricks to handle nonlinear input spaces. This method discriminates the data points using a hyperplane with the largest amount of margin. 
A hyperplane is a decision plane that separates between a set of objects having different class membership.
Margin defines the gap between the two lines on the closest point. This is calculated as the perpendicular distance from the line to support vectors or closest points.  
Kernel: SVM uses a kernel practice or tricks to build, classify and predict the information where the kernel is used to transform the low dimensional input data space into a higher-dimensional space.
There are various kernel tricks SVM uses to convert a nonseparable problem to a separable problem.

	Linear kernel
	Polynomial Kernel
	Radial basis function kernel 

##### Advantages: 
	Handle multiple continuous and categorical variables. 
	Use less memory as they use a subset of training points in a decision phase.
##### Disadvantages:
	Difficult to when the data is too large because of training time. 
	Also, it works inadequately with overlapping classes.


Above-stated all the machine learning algorithms have their characteristics and they perform well in their playground. 
The approach and the theory about the Naïve Bayes classifier are looking perfect match to predict the gender from our dataset. It is highly scalable with a number of predictors and data points. The probabilistic nature makes this algorithm very fast in computation. It requires less training data and it performs better when the assumptions of the independence of features hold.
Issues with the Naïve Bayes classifier model is that it assumes all predictors or features are independent which rarely happens in real life. In a categorical variable if the category of the test dataset is not observed by the training dataset, then the model assigns a 0 probability and it’s difficult for the prediction. This Zero frequency assignation can be solved by smoothing techniques.
We can perform a Deep interest neural network (DIN). To build a neural network we need more data as well as the data which are more likely close to the predictor (Gender). DIN adaptively calculates the representation vectors of user interests considering the relevance of historical behaviors concerning company ads.

### References:
	https://medium.com/@MohammedS/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b
	https://machinelearningmastery.com/feature-selection-with-categorical-data/ 
	https://www.researchgate.net/publication/3845468_Gender_classification_with_support_vector_machines 
	https://www.programmersought.com/article/89554712508/ 
	https://www.cis.uni-muenchen.de/~hs/teach/18w/pdf/13bayesflat.pdf 
	https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python 
	https://arxiv.org/pdf/1706.06978.pdf



