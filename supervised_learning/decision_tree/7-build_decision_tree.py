#!/usr/bin/env python3
"""
1-build_decision_tree.py
"""

import numpy as np


class Node:
    """
    Represents an internal node in the tree, where a decision is made
    """
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Function that find the maximum of the depths of the nodes
        """
        if self.left_child or self.right_child:
            left_depth = self.left_child.max_depth_below()
            right_depth = self.right_child.max_depth_below()
            return max(left_depth, right_depth)
        else:
            return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Function that count the number of nodes in a decision tree,
        potentially excluding the root and internal nodes to count
        only the leaves
        """
        if self.left_child is None and self.right_child is None:
            return 1
        else:
            countleft = (
                self.left_child.count_nodes_below(only_leaves=only_leaves)
                if self.left_child is not None
                else 0)
            countright = (
                self.right_child.count_nodes_below(only_leaves=only_leaves)
                if self.right_child is not None
                else 0)
            if only_leaves:
                return countleft + countright
            else:
                return 1 + countleft + countright

    def left_child_add_prefix(self, text):
        """
        This method adds visual prefixes to show the tree hierarchy when
        it is printed.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0]+"\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """
        This method adds visual prefixes to show the tree hierarchy when
        it is printed.
        """
        lines = text.split("\n")
        new_text = "    +--" + lines[0]+"\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return (new_text)

    def __str__(self):
        """
        Function that returns a readable representation
        """
        str = f"{'root' if self.is_root else '-> node'} \
[feature={self.feature}, threshold={self.threshold}]\n"
        if self.left_child:
            str += self.left_child_add_prefix(
                self.left_child.__str__().strip())
        if self.right_child:
            str += self.right_child_add_prefix(
                self.right_child.__str__().strip())
        return str

    def get_leaves_below(self):
        """
        This method return a list of all leaves of the tree
        """
        if self.left_child is None and self.right_child is None:
            return [self]
        else:
            total = []
            if self.left_child is not None:
                total.extend(self.left_child.get_leaves_below())
            if self.right_child is not None:
                total.extend(self.right_child.get_leaves_below())
            return total

    def update_bounds_below(self):
        """
        Recursively assign lower and upper feature bounds to each node
        """
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1*np.inf}

        if self.left_child:
            self.left_child.lower = self.lower.copy()
            self.left_child.upper = self.upper.copy()
            self.left_child.lower[self.feature] = self.threshold

        if self.right_child:
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()
            self.right_child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def get_leaves_below(self):
        """
        Return a flat list of all leaf nodes under this node
        """
        hojas = []
        if self.left_child:
            hojas += self.left_child.get_leaves_below()
        if self.right_child:
            hojas += self.right_child.get_leaves_below()
        return hojas

    def update_indicator(self):
        """
        Calculates the indicator function based on lower and upper bounds.
        """

        def is_large_enough(x):
            """
            Returns True if features are greater than lower bounds
            """
            return np.array([np.greater(x[:, key], self.lower[key])
                             for key in self.lower.keys()]).all(axis=0)

        def is_small_enough(x):
            """
            Returns True if features are less than or equal to upper bounds
            """
            return np.array([np.less_equal(x[:, key], self.upper[key])
                             for key in self.upper.keys()]).all(axis=0)

        self.indicator = lambda x: np.all(np.array(
            [is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        """
        Returns the predicted class for sample x by traversing
        the tree from this node
        """
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """
    Represents the decisions
    """
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Function that find the maximum of the depths of the nodes
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Function that count the number of nodes in a decision tree,potentially
        excluding the root and internal nodes to count only the leaves
        """
        return 1

    def __str__(self):
        """
        Function that returns a readable representation
        """
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """
        This method return a list of all leaves of the tree
        """
        return [self]

    def update_bounds_below(self):
        """
        Function that does nothing
        """
        pass

    def pred(self, x):
        """
        Returns the predicted class for sample
        x by traversing the tree from this node
        """
        return self.value


class Decision_Tree():
    """
    It is the class that structures the entire tree
    """
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Function that find the maximum of the depths of the nodes
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Function that count the number of nodes in a decision tree,
        potentially excluding the root and internal nodes to count
        only the leaves
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Function that returns a readable representation
        """
        return self.root.__str__()

    def get_leaves(self):
        """
        This method return a list of all leaves of the tree
        """
        return self.root.get_leaves_below()

    def update_bounds(self):
        """
        Recursively assign lower and upper feature bounds to each node
        """
        self.root.update_bounds_below()

    def pred(self, x):
        """
        Returns the predicted class for sample
        x by traversing the tree from this node
        """
        return self.root.pred(x)

    def update_predict(self):
        """
        Prepares and assigns a vectorized predict function that,
        without loops,computes predictions
        for a sample matrix in bulk.
        """
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: (
            np.column_stack([leaf.indicator(A) for leaf in leaves])
            .dot(np.array([leaf.value for leaf in leaves]))
        )

    def fit(self, explanatory, target, verbose=0):
        """
        Fit the decision tree to the training data.
        """
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)

        self.update_predict()

        if verbose == 1:
            print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {
                                    self.accuracy(
                                        self.explanatory, self.target
                                        )
                }""")

    def np_extrema(self, arr):
        """
        Train the decision tree on explanatory data and targets
        """
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """
        Train the decision tree on explanatory data and targets
        """
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max-feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        """
        Split a node into left and right children based on criterion
        """
        node.feature, node.threshold = self.split_criterion(node)
        mask = self.explanatory[:, node.feature] > node.threshold

        left_population = node.sub_population & mask
        right_population = node.sub_population & ~mask

        # Is left node a leaf ?
        n_left = np.sum(left_population)
        cond_min = (n_left < self.min_pop)
        cond_depth = (node.depth >= self.max_depth)
        vals_left = self.target[left_population]
        cond_pure = (np.unique(vals_left).size == 1)
        is_left_leaf = cond_min or cond_depth or cond_pure

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        n_right = np.sum(right_population)
        cond_min2 = (n_right < self.min_pop)
        vals_right = self.target[right_population]
        cond_pure2 = (np.unique(vals_right).size == 1)
        is_right_leaf = cond_min2 or cond_depth or cond_pure2

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """
        Create a leaf node with the majority target value
        """
        vals = self.target[sub_population]
        counts = np.bincount(vals)
        value = counts.argmax()
        leaf_child = Leaf(
            value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """
        Train the decision tree on explanatory data and targets
        """
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """
        Compute the accuracy of predictions on test data
        """
        return np.sum(
            np.equal(self.predict(test_explanatory), test_target)
            )/test_target.size
