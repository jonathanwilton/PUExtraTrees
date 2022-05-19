# PU ExtraTrees

import numpy as np
from classifier_loss import risk
import scipy.stats
from joblib import Parallel, delayed
import scipy.sparse
import time


class PUExtraTrees:
    def __init__(self, n_estimators = 1, 
                 estimator = 'nnPU',
                 criterion = 'gini',
                 max_depth = None, 
                 min_samples_leaf = 1, 
                 max_features = 'auto', 
                 max_candidates = 1, 
                 n_jobs = 1):
        """
        

        Parameters
        ----------
        n_estimators : int, optional
            The number of trees in the forest. The default is 1.
        estimator : {"PN", "uPU", "nnPU"}, optional
            PU data based estimator. Supports supervised (PN) learning, unbiased PU (uPU) learning and nonnegative PU (nnPU) learning. The default is 'nnPU'.
        criterion : {"gini", "entropy"}, optional
            The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. The default is 'gini'.
        max_depth : int or None, optional
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_leaf samples. The default is None.
        min_samples_leaf : int, optional
            The minimum number of samples required to be at a leaf node. The default is 1.
        max_features : int or {"auto", "all"}, optional
            The number of features to consider when looking for the best split. If "auto", then max_features = ceil(sqrt(n_features)). If "all", then max_features = n_features. The default is 'auto'.
        max_candidates : int, optional
            Number of randomly chosen split points to consider for each candidate feature. The default is 1.
        n_jobs : int, optional
            The number of jobs to run in parallel. fit, predict and predict_proba are all parallelized over the trees. The default is 1.

        Returns
        -------
        None.

        """
        
        self.n_estimators = n_estimators
        self.estimator = estimator
        self.criterion = criterion
        self.is_trained = False # indicate if tree empty/trained
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_candidates = max_candidates
        self.leaf_count = [0 for i in range(self.n_estimators)]
        self.n_jobs = n_jobs
        self.p = None
        self.current_max_depth = 0
        
        self.nodes = [{(0,0): {'data': None, 'j': None, 'xi': None, 'g': None, 
                               'is_leaf': None, 'loss': None, 'impurity_decrease': None}} 
                      for i in range(self.n_estimators)]
        
        
    def create_successor(self, node, tree, side):
        """
        Create a child node (either T or F) in the tree.

        Parameters
        ----------
        node : tuple of length 2.
            The parent node. First element is the depth in the tree, second element is the position at that depth.
        tree : int
            Which tree to create children node in.
        side : {"T", "F"} or {"L", "R"}
            Whether the node corresponds to a True or False split.

        Returns
        -------
        None.

        """
        
        row, column = node
        if side in ['T','L']:
            self.nodes[tree][(row+1, 2*column)] = {'data': None, 'j': None, 
                                                   'xi': None, 'g': None, 
                                                   'is_leaf': None, 'loss': None, 
                                                   'impurity_decrease': None}
        elif side in ['F','R']:
            self.nodes[tree][(row+1, 2*column+1)] = {'data': None, 'j': None, 
                                                     'xi': None, 'g': None,
                                                     'is_leaf': None, 'loss': None,
                                                     'impurity_decrease': None}
        elif side not in ['T','F','L','R']:
            print('choose valid position of child node: \'L\', \'R\', \'T\', \'F\'')
    
    
    
    def get_parent(self, node, tree, return_truth_val):
        """
        Return parent node of the specified node in "tree". 
        Scratch everything, just assume everything works so no checks

        Parameters
        ----------
        node : tuple of length 2
            The child node.
        tree : int
            Which tree to find parent in.
        return_truth_val : bool
            Indicate whether the truth value should also be returned, that is, whether the child node corresponds to a true or false split.

        Returns
        -------
        tuple of length 2 or (tuple of length 2, bool)
            The parent node, optionally the relationship to the parent node.

        """
        
        parent = (node[0] - 1, node[1] // 2)
        if return_truth_val:
            if node[1] % 2 == 0:
                return parent, True
            else:
                return parent, False
        else:
            return parent
        
    
    
    def get_ancestory(self, node, tree):
        """
        Get chain of parents all the way to root node.

        Parameters
        ----------
        node : tuple of length 2
            Child node.
        tree : int
            The tree in which to get the ancestory information from.

        Returns
        -------
        list
            List of nodes.
        bools : list
            List of bools with the relationships to the parents.

        """
        
        chain = [node]
        bools = []
        while chain[-1] != (0,0):
            parent, relationship = self.get_parent(chain[-1], tree, True)
            chain += [parent]
            bools += [relationship]
        
        return chain[1:], bools


    def load_forest(self, nodes):
        # load saved forest
        self.nodes = nodes
        self.is_trained = True
        
    


    def fit(self, P = None, U = None, N = None, pi = None):
        """
        

        Parameters
        ----------
        P : array-like of shape (n_p, n_features), optional
            Training samples from the positive class. The default is None.
        U : array-like of shape (n_u, n_features), optional
            Unlabelled training samples. The default is None.
        N : array-like of shape (n_n, n_features), optional
            Training samples from the negative class. The default is None.
        pi : float or None, optional
            Prior probability that an example belongs to the positive class. If pi = None and self.estimator in {"uPU", "nnPU"} then we will attempt to estimate pi from the data. The default is None.

        Returns
        -------
        self
            Return instance of self.

        """
        
        if self.estimator in ['uPU', 'nnPU']:
            X = np.concatenate((P, U), axis = 0)
            y = np.concatenate((np.ones(len(P)), np.zeros(len(U))))
        elif self.estimator in ['PN']:
            X = np.concatenate((P, N), axis = 0)
            y = np.concatenate((np.ones(len(P)), -np.ones(len(N))))
        
        # X = X.astype(np.float32)
        y = y.astype(np.int8).flatten()
        n, self.d = X.shape
        n_p = (y == 1).sum()
        n_u = (y == 0).sum()
        n_n = (y == -1).sum()
        self.pi = pi
                
        if self.pi is None:
            print('Please specify pi')
            break

                
        if self.max_features == 'auto':
            max_features = int(np.ceil(np.sqrt(X.shape[1])))
        elif self.max_features == 'all':
            max_features = X.shape[1]
        elif self.max_features in [i for i in range(1, self.d+1)]:
            max_features = self.max_features
        else:
            print('select valid number of max features to consider splitting on.')
        
        
        for i in range(self.n_estimators):
            self.nodes[i][(0,0)]['data'] = scipy.sparse.coo_matrix(np.ones(n).astype(bool))
        
        def data_at_node(node, tree):
            # return subset of training data in partition specified by certain node            
            
            if self.nodes[tree][node]['data'] is not None:
                return self.nodes[tree][node]['data'].toarray()[0]
            else:
                # get indices of data at parent
                parent_node, relationship = self.get_parent(node, tree, True)
                ind_parent = self.nodes[tree][parent_node]['data'].toarray()[0].copy()
                checks = (X[ind_parent, self.nodes[tree][parent_node]['j']] <= self.nodes[tree][parent_node]['xi']) == relationship
                ind_parent[ind_parent] = checks.flatten()
                self.nodes[tree][node]['data'] = scipy.sparse.coo_matrix(ind_parent)
                return ind_parent
            
        
        def impurity_split(sigma, j, xi):
            mask = (X[sigma, j] <= xi).flatten()
                                        
            nwT = len(y[sigma][mask])
            nwF = len(y[sigma][~mask])
            
            imT = impurity_node(y[sigma][mask])
            imF = impurity_node(y[sigma][~mask])
        
            weight_T = 1
            weight_F = 1
                        
            return weight_T * imT + weight_F * imF
        
        
        def impurity_node(y_sigma):
            # impurity of single node
            if self.estimator in ['uPU', 'nnPU']:
                nwp = (y_sigma == 1).sum()
                nwu = (y_sigma == 0).sum()
                
                wpk = self.pi*nwp/n_p
                wnk = nwu/n_u - self.pi*nwp/n_p
                
                if nwu > 0:
                    ystar = self.pi*nwp/n_p * n_u/nwu 
                else: # pure positive
                    if self.estimator == 'uPU':
                        return -float('inf')
                    else:
                        return 0
                
                if self.estimator == 'uPU':
                    if self.criterion == 'gini':
                        return (nwu/n_u)*ystar*(1-ystar)
                    
                    elif self.criterion == 'entropy':
                        if (wpk > 0) and (wnk > 0):
                            return (nwu/n_u)*(-ystar * np.log(ystar) - (1-ystar) * np.log(1-ystar))
                        elif (ystar == 1) or (ystar == 0):
                            return 0
                        else:
                            return -float('inf')
                        
                elif self.estimator == 'nnPU':
                    if self.criterion == 'gini':
                        if (wnk <= 0): #pure positive
                            return 0
                        else:
                            return (nwu/n_u)*ystar*(1-ystar)
                        
                    elif self.criterion == 'entropy':
                        if (wpk > 0) and (wnk > 0):
                            return (nwu/n_u)*(-ystar * np.log(ystar) - (1-ystar) * np.log(1-ystar))
                        else:
                            return 0
                            
            elif self.estimator in ['PN']:
                nwp = (y_sigma == 1).sum()
                nwn = (y_sigma == -1).sum()
                ystar = self.pi*nwp/n_p /(self.pi*nwp/n_p + (1-self.pi)*nwn/n_n)
                
                if self.criterion == 'gini':
                    return ((nwp+nwn)/n)*ystar*(1-ystar)
                
                elif self.criterion == 'entropy':
                    if (ystar == 0) or (ystar == 1):
                        return 0
                    else:
                        return ((nwp+nwn)/n)*(-ystar * np.log(ystar) - (1-ystar) * np.log(1-ystar))
        
        
        def regional_prediction_function(sigma, return_risk = False):            
            risk_pos, risk_neg = risk(y = y[sigma], pi = self.pi, n_p = n_p, n_u = n_u, n_n = n_n, estimator = self.estimator)
            if return_risk:
                if risk_neg < risk_pos:
                    return -1, risk_neg
                elif risk_neg > risk_pos:
                    return 1, risk_pos
                else:
                    g = 2*np.random.binomial(1,0.5)-1
                    return g, (g == 1)*risk_pos + (g == -1)*risk_neg
            else:
                if risk_neg < risk_pos:
                    return -1
                elif risk_neg > risk_pos:
                    return 1
                else:
                    return 2*np.random.binomial(1,0.5)-1
                    

        def construct_subtree(node, tree, sigma):
            
            # check criteria
            impurity = impurity_node(y[sigma])
            
            # check node pure
            if self.estimator in ['nnPU', 'PN']:
                c1 = impurity > 0 
            elif self.estimator == 'uPU':
                c1 = impurity > -float('inf')            
            
            # check max depth reached
            if self.max_depth is None:
                c2 = True
            else:
                c2 = node[0] < self.max_depth # max depth reached
            c3 = self.min_samples_leaf < sigma.sum() # minimum samples in node reached
            att_ptp = np.ptp(X[sigma], axis = 0)
            c4 = att_ptp.sum() > 0 # check if there is any variability in features
                        
            # check if any of the criteria satisfied
            # if so, turn into a leaf node
            if c1*c2*c3*c4 == 0:
                self.nodes[tree][node]['is_leaf'] = True
                lab, loss = regional_prediction_function(sigma, return_risk = True)
                
                self.nodes[tree][node]['g'] = lab
                self.nodes[tree][node]['loss'] = loss
                self.leaf_count[tree] += 1
                self.nodes[tree][node]['impurity_decrease'] = 0
            else:
                self.nodes[tree][node]['is_leaf'] = False
                atts = []
                for i in range(self.d):
                    if att_ptp[i] > 0:
                        atts += [i]
                
                attributes = np.random.choice(atts, size = min(max_features, len(atts)), replace = False)
                candidates = []
                candidate_attributes = []
                candidate_cut_points = []
                for i in range(len(attributes)):
                    for j in range(self.max_candidates):
                        # need to guard against errors caused by finite precision
                        a_,b_,c_,d_ = np.unique(X[sigma, attributes[i]])[[0,1,-2,-1]]                        
                        cut_point = np.random.uniform(a_ + 2*(b_-a_)/5, c_ + 3*(d_-c_)/5)
                        candidates += [[attributes[i], cut_point]]
                        candidate_attributes += [attributes[i]]
                        candidate_cut_points += [cut_point]

                impurities = []
                for i in range(len(candidates)):
                    impurities += [impurity_split(sigma, candidate_attributes[i], candidate_cut_points[i])]
                
                minimiser = np.argmin(impurities)
                best_attribute = candidate_attributes[minimiser]
                best_cut_point = candidate_cut_points[minimiser]
                
                self.nodes[tree][node]['j'] = int(best_attribute)
                self.nodes[tree][node]['xi'] = best_cut_point
                self.nodes[tree][node]['impurity_decrease'] = impurity - impurities[minimiser]

                # create successors of current node
                self.create_successor(node, tree, 'T')
                self.create_successor(node, tree, 'F')
                
                ## get set of data in these successors
                succs = ((node[0]+1, 2*node[1]), (node[0]+1, 2*node[1]+1))
                sigma_T = data_at_node(succs[0], tree)
                sigma_F = data_at_node(succs[1], tree)                
                
                if node[0] > self.current_max_depth:
                    self.current_max_depth = node[0]

                construct_subtree(succs[0], tree, sigma_T)
                construct_subtree(succs[1], tree, sigma_F)
        
        Parallel(n_jobs = min(self.n_jobs, self.n_estimators), prefer="threads")(delayed(construct_subtree)((0,0), i, np.ones(n).astype(bool)) for i in range(self.n_estimators))
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
        predictions: array of shape (n_samples)
            The predicted classes.

        """

        # parallelise over trees
        def predict_single_tree(tree):
            preds = np.zeros(len(X)).astype(np.int8)
            for i in range(len(X)):
                X_ = X[i]
                a,b = 0,0
                tnode = self.nodes[tree][(a,b)]
                if not tnode['is_leaf']:
                    while not tnode['is_leaf']:
                        check = X_[tnode['j']] <= tnode['xi']
                        
                        if check:
                            b = 2*b
                        else:
                            b = 2*b + 1
                        
                        a += 1
                        tnode = self.nodes[tree][(a,b)]
                
                if tnode['is_leaf']:
                    preds[i] = tnode['g']
        
            return preds
        
        
        # first check to see if the tree is empty/trained
        if self.is_trained:
            predictions = Parallel(n_jobs = min(self.n_jobs, self.n_estimators), require='sharedmem')(delayed(predict_single_tree)(tree) for tree in range(self.n_estimators))
            return scipy.stats.mode(np.array(predictions), axis = 0)[0][0]
        else:
            print('tree not finished training!')
    
    
    def n_leaves(self, tree):
        """
        Get the number of leaf nodes in a tree (number of regions created in feature space).

        Returns
        -------
        temp : int
            Number of leaf nodes in the classifier.

        """
        return self.leaf_count[tree]
    
    def get_depth(self, tree):
        """
        Return the depth of a decision tree in the forest. The depth of a tree is the maximum distance between the root and any leaf.

        Parameters
        ----------
        tree : int
            The tree to find the depth of.

        Returns
        -------
        depth: int
            The maximum depth of the tree.

        """
        max_depth = -1
        for node in self.nodes[tree].keys():
            if node[0] > max_depth:
                max_depth = node[0]
        return max_depth
    
    def feature_importances(self):
        """
        Compute the impurity-based feature importances.

        Returns
        -------
        array-like of shape (n_features,)
            Impurity-based feature importances.

        """
        
        impurities = np.zeros([self.n_estimators, self.d])
        for tree in range(self.n_estimators):
            for node in self.nodes[tree]:
                if self.nodes[tree][node]['j'] is not None:
                    impurities[tree, self.nodes[tree][node]['j']] += self.nodes[tree][node]['impurity_decrease'] 
        
        return impurities.mean(axis = 0)
