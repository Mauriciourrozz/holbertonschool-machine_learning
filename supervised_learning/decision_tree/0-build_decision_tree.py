#!/usr/bin/env python3
"""
0-build_decision_tree.py
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
