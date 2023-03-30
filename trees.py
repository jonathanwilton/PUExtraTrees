# PU ExtraTrees - A Random Forest Classifier for PU Learning
from tree import PUExtraTree
from joblib import Parallel, delayed
import scipy
import numpy as np

class PUExtraTrees:
    def __init__(self, n_estimators = 100,
                 risk_estimator = 'nnPU',
                 loss = 'quadratic',
                 max_depth = None, 
                 min_samples_leaf = 1, 
                 max_features = 'sqrt', 
                 max_candidates = 1,
                 n_jobs = 1):
        """
        An extra-trees binary classifier that can be trained using only positive and unlabeled samples, or positive and negative samples.
        
        Parameters
        ----------
        risk_estimator : {"PN", "uPU", "nnPU"}, default='nnPU'
            PU data based risk estimator. Supports supervised (PN) learning, unbiased PU (uPU) learning and nonnegative PU (nnPU) learning.
        loss : {"quadratic", "logistic"}, default='quadratic'
            The function to measure the cost of making an incorrect prediction. Supported loss functions are:
            "quadratic" l(v,y) = (1-vy)^2 and 
            "logistic" l(v,y) = ln(1+exp(-vy)).
        max_depth : int or None, default=None
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_leaf samples. 
        min_samples_leaf : int, default=1
            The minimum number of samples required to be at a leaf node. The default is 1.
        max_features : int or {"sqrt", "all"}, default="sqrt"
            The number of features to consider when looking for the best split. If "sqrt", then max_features = ceil(sqrt(n_features)). If "all", then max_features = n_features. 
        max_candidates : int, default=1
            Number of randomly chosen split points to consider for each candidate feature. 
        n_jobs : int, default=1
            The number of jobs to run in parallel. fit and predict are all parallelized over the trees.
         
        Returns
        -------
        None.

        """
        
        self.n_estimators = n_estimators
        self.risk_estimator = risk_estimator
        self.loss = loss
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_candidates = max_candidates
        self.n_jobs = n_jobs
        
        self.leaf_count = 0
        self.current_max_depth = 0
        self.is_trained = False # indicate if tree empty/trained
    
    def train_tree(self, P = None, U = None, N = None, pi = None):
        """
        Train a single decision tree.

        Parameters
        ----------
        P : array-like of shape (n_p, n_features), default=None
            Training samples from the positive class. 
        U : array-like of shape (n_u, n_features), default=None
            Unlabelled training samples.
        N : array-like of shape (n_n, n_features), default=None
            Training samples from the negative class if performing supervised (PN) learning.
        pi : float
            Prior probability that an example belongs to the positive class.

        Returns
        -------
        g : ET classifier
            An instance of the single tree RF classifier.

        """
        g = PUExtraTree(risk_estimator = self.risk_estimator,
                        loss = self.loss,
                        max_depth = self.max_depth, 
                        min_samples_leaf = self.min_samples_leaf, 
                        max_features = self.max_features, 
                        max_candidates = self.max_candidates)
        g.fit(P = P, U = U, N = N, pi = pi)
        return g
    
    def predict_tree(self, g, X):
        """
        Predict classes for examples in X using the single DT g.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The test samples.

        Returns
        -------
        preds : array of shape (n_samples,)
            The predicted classes.

        """
        return g.predict(X)
    
    def fit(self, P = None, U = None, N = None, pi = None):
        """
        Train the random forest.

        Parameters
        ----------
        pi : float
            Prior probability that an example belongs to the positive class. 
        P : array-like of shape (n_p, n_features), default=None
            Training samples from the positive class.
        U : array-like of shape (n_u, n_features), default=None
            Unlabeled training samples.
        N : array-like of shape (n_n, n_features), default=None
            Training samples from the negative class if performing PN learning.

        Returns
        -------
        self
            Returns instance of self.

        """
        self.gs = Parallel(n_jobs = min(self.n_jobs, self.n_estimators), prefer="threads")(delayed(self.train_tree)(P = P, U = U, N = N, pi = pi) for i in range(self.n_estimators))
        self.is_trained = True
        return self
    
    def predict(self, X):
        """
        Predict classes for examples in X.
        The predicted class of an input sample is the majority vote by the trees in the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The test samples.

        Returns
        -------
        preds : array of shape (n_samples,)
            The predicted classes.

        """
        self.preds = Parallel(n_jobs = min(self.n_jobs, self.n_estimators), prefer="threads")(delayed(self.predict_tree)(g, X) for g in self.gs)
        return scipy.stats.mode(np.array(self.preds), axis = 0, keepdims = False)[0]
    
    def n_leaves(self, tree):
        """
        Get the number of leaf nodes in a specified tree

        Parameters
        ----------
        tree : int
            The index of the tree.

        Returns
        -------
        Number of leaf nodes in the specified tree.

        """
        
        return self.gs[tree].n_leaves()
    
    def get_depth(self, tree):
        """
        Get the depth of a specified tree in the forest.

        Parameters
        ----------
        tree : int
            The index of the tree.

        Returns
        -------
        Depth of the specified tree.

        """
        
        return self.gs[tree].get_depth()
    
    def get_max_depth(self):
        """
        Return the depth of the deepest tree in the forest.

        Returns
        -------
        Maximum depth : int

        """
        
        depths = []
        for tree in self.gs:
            depths += [tree.get_depth()]
        return np.max(depths)
    
    def feature_importances(self):
        """
        Get the risk reduction feature importances.

        Returns
        -------
        importances : array of shape (n_features,)
            The risk reduction feature importances.

        """
        importances = np.zeros([self.gs[0].d])
        for tree in self.gs:
            importances += tree.feature_importances()/self.n_estimators
        
        return importances
        
