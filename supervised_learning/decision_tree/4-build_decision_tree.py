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

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                f = self.feature
                t = self.threshold
                if child is self.left_child:
                    child.upper[f] = t
                else:
                    child.lower[f] = t
        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def get_leaves_below(self):
        """
        Return a flat list of all leaf nodes under this node
        """
        if not self.left_child and not self.right_child:
            return [self]
        else:
            hojas = []
            if self.right_child:
                hojas.extend(self.right_child.get_leaves_below())
            if self.left_child:
                hojas.extend(self.left_child.get_leaves_below())
            return hojas


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
