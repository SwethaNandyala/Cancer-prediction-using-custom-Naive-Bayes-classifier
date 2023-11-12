import pandas as pd
import numpy as np
class custom_naive_bayes():
    
    
    def __init__(self):
        
        self.ind_features = []
        self.target_classes = []
        self.class_prior_prob = dict()
        self.ind_features_sumary = dict()
        
        
    def calculate_class_probability(self,y):
        # calculate the prior
        # calculates the prob of each class in the target variable
        for i in range(len(self.target_classes)):
            temp = y[y==self.target_classes[i]]
            class_probablity = len(temp) / len(y)
            self.class_prior_prob[self.target_classes[i]] = np.round(class_probablity,3)

    def calculate_independent_feature_summary(self,X,y):

        data = pd.concat([X,y],axis=1)
        target_feature = y.name

        for i in self.ind_features:
            #create a temp dict to store mean and std of each class
            class_mean_std = {}

            for j in self.target_classes:
                temp_df = data[data[target_feature] == j][i]
                mean = temp_df.mean()
                std = temp_df.std()
                class_mean_std[j] = (np.round(mean,3), np.round(std,3))

            # for each feature append the mean std of each class to the main dict
            self.ind_features_sumary[i] = class_mean_std

    def fit(self,X,y):

        #taking the list of clumns and target from X and y
        self.ind_features.extend(list(X.columns))
        self.target_classes.extend(list(y.unique()))
        self.target_classes.sort()

        self.calculate_class_probability(y)
        self.calculate_independent_feature_summary(X,y)


    def calculate_probabilty(self,x_i,mean,std_dev):
        #calculates the probability of a feature being in certain class
        exponent = np.exp(-(x_i-mean)**2 / (2* (std_dev**2)))
        prob = (1/np.sqrt(2*(np.pi)*(std_dev**2)))*(exponent)
        return prob

    def calculate_likelihood(self,X): 
        
        #calculate the likelihood of being in class_0 class_1 class_2

        X_likelihood_prob = pd.DataFrame()

        for cls in self.target_classes:

            temp = pd.DataFrame()

            for each_col in self.ind_features:            
                #calculate the prob of each feature
                mean = self.ind_features_sumary[each_col][cls][0]
                std = self.ind_features_sumary[each_col][cls][1]
                temp[each_col] = X[each_col].apply(lambda x: self.calculate_probabilty(x,mean,std))            

            X_likelihood_prob["Prob_"+str(cls)] = temp.apply(lambda row: np.prod(row), axis = 1)
        return X_likelihood_prob

    def calculate_posterior_prob(self,X):
        
        #posterior P(class/X) = P(X/class)*prior

        X_likelihood_prob = self.calculate_likelihood(X)
        target_cols = X_likelihood_prob.columns

        for i in range(len(self.target_classes)):
            #multiply with prior
            prior_probablity = self.class_prior_prob[self.target_classes[i]]
            X_likelihood_prob[target_cols[i]] = X_likelihood_prob[target_cols[i]].apply(lambda x: x*prior_probablity)

        #take the index of maximum value
        X_likelihood_prob["Predicted_Label"] = X_likelihood_prob.apply(lambda row: np.argmax(row),axis =1 )
        
        return X_likelihood_prob["Predicted_Label"]

    def predict(self,X):
        
        X_likelihood_prob = self.calculate_likelihood(X)
        X_posterior_prob  = self.calculate_posterior_prob(X)
        return X_posterior_prob
