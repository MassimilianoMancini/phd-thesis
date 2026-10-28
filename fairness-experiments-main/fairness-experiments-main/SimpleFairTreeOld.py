import pandas as pd
import numpy as np
from fairnessMeasuresAPI import fm

class Node:
    """Decision tree node"""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def predict_one(self, row):
        """Classifies a single example by traversing the tree"""
        if self.value is not None:
            return self.value
        if row[self.feature] < self.threshold:
            return self.left.predict_one(row)
        return self.right.predict_one(row)

class SimpleFairTree:
    """Class for creating a simplified Decision Tree"""
    def __init__(self, max_depth=5, measure='entropy'):
        self.max_depth = max_depth
        self.root = None
        if measure not in ['entropy', 'gini']:
            raise ValueError("Measure must be either 'entropy' or 'gini'")
        if measure == 'entropy':
            self.measure = self.entropy
        elif measure == 'gini':
            self.measure = self.gini

    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs))
    
    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def split_dataset(self, df, feature, threshold):
        left_mask = df[feature] < threshold
        right_mask = ~left_mask
        return df[left_mask], df[right_mask]
    
    def find_best_split(self, df_partial, target_col):
        """Finds the best feature and threshold"""
        
        best_feature, best_threshold, best_gain = None, None, 0
        current_measure = self.measure(df_partial[target_col])

        for feature in df_partial.columns.drop(target_col):
            thresholds = df_partial[feature].unique()
            for threshold in thresholds:
                left_df, right_df = self.split_dataset(df_partial, feature, threshold)
                if len(left_df) == 0 or len(right_df) == 0:
                    continue

                left_measure, right_measure = self.measure(left_df[target_col]), self.measure(right_df[target_col])
                left_weight, right_weight = len(left_df) / len(df_partial), len(right_df) / len(df_partial)
                gain = current_measure - (left_weight * left_measure + right_weight * right_measure)

                if gain > best_gain:
                    best_feature, best_threshold, best_gain = feature, threshold, gain

        return best_feature, best_threshold

    def find_best_split_fair(self, dataset: pd.DataFrame, target_col: str, fairness_measure: fm, measure=None, fair_weight=None):
        """Finds the best feature and threshold, applying fairness penalty using updated class labels in df_full"""
        
        best_feature, best_threshold, best_gain = None, None, 0
        current_measure = self.measure(dataset[target_col])

        for feature in dataset.columns.drop([target_col]):
            thresholds = dataset[feature].unique()
            for threshold in thresholds:
                left_df, right_df = self.split_dataset(dataset, feature, threshold)
                if len(left_df) == 0 or len(right_df) == 0:
                    continue

                # Assign mode as predicted class for left and right subsets
                left_class = left_df[target_col].mode()[0]
                right_class = right_df[target_col].mode()[0]
                if left_class == right_class:
                    continue

                if dataset[fairness_measure.getSensibleAttribute()].nunique() == 1:
                    bias = 0
                else:
                    df = dataset.copy()
                    df.loc[left_df.index, 'y_hat'] = left_class
                    df.loc[right_df.index, 'y_hat'] = right_class
                    df['y_hat'] = df['y_hat'].astype(int)
                    fairness_measure.setDataset(df)
                    bias = fairness_measure.measure(measure=measure, result='abs')

                left_measure, right_measure = self.measure(left_df[target_col]), self.measure(right_df[target_col])
                left_weight, right_weight = len(left_df) / len(dataset), len(right_df) / len(dataset)

                gain = current_measure - (left_weight * left_measure + right_weight * right_measure) - (fair_weight * bias)

                if gain > best_gain:
                    best_feature, best_threshold, best_gain = feature, threshold, gain

        return best_feature, best_threshold

    def build_tree(self, dataset: pd.DataFrame, target_col: str, depth=0):
        """Recursively builds the decision tree while keeping the full dataset for fairness calculations"""
        # Base case: if all samples belong to the same class or max depth is reached
        if len(dataset[target_col].unique()) == 1 or depth >= self.max_depth:
            return Node(value=dataset[target_col].mode()[0])

        # Find best feature and threshold using the full dataset for fairness penalty
        feature, threshold = self.find_best_split(dataset, target_col)
        if feature is None:
            return Node(value=dataset[target_col].mode()[0])

        # Split dataset and create child nodes recursively
        left_subset, right_subset = self.split_dataset(dataset, feature, threshold)
        left_node = self.build_tree(left_subset, target_col, depth + 1)
        right_node = self.build_tree(right_subset, target_col, depth + 1)

        return Node(feature, threshold, left_node, right_node)

    def build_tree_fair(self, dataset: pd.DataFrame, target_col: str, fairness_measure: fm, depth=0, measure=None, fair_weight=None):
        """Recursively builds the decision tree while keeping the full dataset for fairness calculations"""
        # Base case: if all samples belong to the same class or max depth is reached
        if len(dataset[target_col].unique()) == 1 or depth >= self.max_depth:
            return Node(value=dataset[target_col].mode()[0])

        # Find best feature and threshold using the full dataset for fairness penalty
        feature, threshold = self.find_best_split_fair(dataset, target_col, fairness_measure, measure=measure, fair_weight=fair_weight)
        if feature is None:
            return Node(value=dataset[target_col].mode()[0])

        # Split dataset and create child nodes recursively
        left_subset, right_subset = self.split_dataset(dataset, feature, threshold)
        left_node = self.build_tree_fair(left_subset, target_col, fairness_measure, depth + 1, measure=measure, fair_weight=fair_weight)
        right_node = self.build_tree_fair(right_subset, target_col, fairness_measure, depth + 1, measure=measure, fair_weight=fair_weight)

        return Node(feature, threshold, left_node, right_node)

    def fit(self, X, target_col, fairness_measures=None, measure=None, fair_weight=None):
        """Trains the model using feature DataFrame and class labels"""

        if fairness_measures is not None:
            self.root = self.build_tree_fair(dataset=X, target_col=target_col, fairness_measure=fairness_measures, measure=measure, fair_weight=fair_weight)
        else:
            # If no fairness measures are provided, build the tree normally
            self.root = self.build_tree(dataset=X, target_col=target_col)

    def predict(self, df):
        """Classifies a dataset automatically"""
        return df.apply(lambda row: self.root.predict_one(row), axis=1)
    
    def explain(self):
        """Prints the tree structure"""
        def print_tree(node, depth=0, prefix="Root: "):
            if node.value is not None:
                print("  " * depth + f"{prefix} Class {node.value}")
                return
            print(f"  " * depth + f"{prefix} [{node.feature} < {node.threshold}]")
            print_tree(node.left, depth + 1, prefix="Left: ")
            print_tree(node.right, depth + 1, prefix="Right: ")

        print_tree(self.root)

    def print_tree_by_level(self):
        """Prints the tree structure level by level (BFS)"""
        from collections import deque
        queue = deque()
        if self.root is None:
            print("Tree is empty.")
            return
        queue.append((self.root, 0))
        current_level = 0
        level_nodes = []
        while queue:
            node, level = queue.popleft()
            if level != current_level:
                print(f"Level {current_level}: {' | '.join(level_nodes)}")
                level_nodes = []
                current_level = level
            if node.value is not None:
                level_nodes.append(f"Leaf: {node.value}")
            else:
                level_nodes.append(f"[{node.feature} < {node.threshold}]")
                queue.append((node.left, level + 1))
                queue.append((node.right, level + 1))
        if level_nodes:
            print(f"Level {current_level}: {' | '.join(level_nodes)}")

    def get_tree_structure(self):
        """Returns the tree structure as a nested dictionary"""
        def node_to_dict(node):
            if node.value is not None:
                return {'value': node.value}
            return {
                'feature': node.feature,
                'threshold': node.threshold,
                'left': node_to_dict(node.left),
                'right': node_to_dict(node.right)
            }
        return node_to_dict(self.root)
    def get_tree_depth(self):
        """Returns the depth of the tree"""
        def depth(node):
            if node.value is not None:
                return 0
            return 1 + max(depth(node.left), depth(node.right))
        
        return depth(self.root)
    def get_num_leaves(self):
        """Returns the number of leaves in the tree"""
        def count_leaves(node):
            if node.value is not None:
                return 1
            return count_leaves(node.left) + count_leaves(node.right)
        
        return count_leaves(self.root)
    def get_num_nodes(self):
        """Returns the total number of nodes in the tree"""
        def count_nodes(node):
            if node is None:
                return 0
            return 1 + count_nodes(node.left) + count_nodes(node.right)
        
        return count_nodes(self.root)
    def get_feature_importance(self):
        """Returns the importance of each feature in the tree"""
        feature_importance = {}

        def traverse(node):
            if node.value is not None:
                return
            if node.feature not in feature_importance:
                feature_importance[node.feature] = 0
            feature_importance[node.feature] += 1
            traverse(node.left)
            traverse(node.right)

        traverse(self.root)
        return feature_importance
    def get_leaf_values(self):
        """Returns the values of the leaves in the tree"""
        leaf_values = []

        def traverse(node):
            if node.value is not None:
                leaf_values.append(node.value)
                return
            traverse(node.left)
            traverse(node.right)

        traverse(self.root)
        return leaf_values
    def get_leaf_nodes(self):
        """Returns the leaf nodes of the tree"""
        leaf_nodes = []

        def traverse(node):
            if node.value is not None:
                leaf_nodes.append(node)
                return
            traverse(node.left)
            traverse(node.right)

        traverse(self.root)
        return leaf_nodes
    def get_internal_nodes(self):
        """Returns the internal nodes of the tree"""
        internal_nodes = []

        def traverse(node):
            if node.value is not None:
                return
            internal_nodes.append(node)
            traverse(node.left)
            traverse(node.right)

        traverse(self.root)
        return internal_nodes
    def get_num_internal_nodes(self):
        """Returns the number of internal nodes in the tree"""
        return len(self.get_internal_nodes())
    def get_num_leaf_nodes(self):
        """Returns the number of leaf nodes in the tree"""
        return len(self.get_leaf_nodes())
    def get_max_leaf_value(self):
        """Returns the maximum leaf value in the tree"""
        return max(self.get_leaf_values()) if self.get_leaf_values() else None
    def get_min_leaf_value(self):
        """Returns the minimum leaf value in the tree"""
        return min(self.get_leaf_values()) if self.get_leaf_values() else None
    def get_average_leaf_value(self):
        """Returns the average leaf value in the tree"""
        leaf_values = self.get_leaf_values()
        return sum(leaf_values) / len(leaf_values) if leaf_values else None
    def get_median_leaf_value(self):    
        """Returns the median leaf value in the tree"""
        leaf_values = self.get_leaf_values()
        return np.median(leaf_values) if leaf_values else None
    def get_leaf_value_counts(self):
        """Returns the counts of each leaf value in the tree"""
        leaf_values = self.get_leaf_values()
        return pd.Series(leaf_values).value_counts().to_dict() if leaf_values else {}
    def get_leaf_value_frequencies(self):
        """Returns the frequencies of each leaf value in the tree"""
        leaf_values = self.get_leaf_values()
        total_leaves = len(leaf_values)
        if total_leaves == 0:
            return {}
        return pd.Series(leaf_values).value_counts(normalize=True).to_dict() if leaf_values else {}
    def get_leaf_value_distribution(self):
        """Returns the distribution of leaf values in the tree"""
        leaf_values = self.get_leaf_values()
        if not leaf_values:
            return {}
        return pd.Series(leaf_values).value_counts(normalize=True).to_dict()
    def get_leaf_value_statistics(self):
        """Returns statistics of leaf values in the tree"""
        leaf_values = self.get_leaf_values()
        if not leaf_values:
            return {}
        return {
            'max': max(leaf_values),
            'min': min(leaf_values),
            'mean': np.mean(leaf_values),
            'median': np.median(leaf_values),
            'std': np.std(leaf_values)
        }
    def get_tree_height(self):
        """Returns the height of the tree"""
        def height(node):
            if node is None:
                return 0
            return 1 + max(height(node.left), height(node.right))
        
        return height(self.root)
    def get_tree_size(self):
        """Returns the size of the tree (number of nodes)"""
        def size(node):
            if node is None:
                return 0
            return 1 + size(node.left) + size(node.right)
        
        return size(self.root)
    def get_tree_depth_and_size(self):
        """Returns the depth and size of the tree"""
        return self.get_tree_depth(), self.get_tree_size()
    def get_tree_statistics(self):
        """Returns statistics of the tree"""
        return {
            'depth': self.get_tree_depth(),
            'size': self.get_tree_size(),
            'num_leaves': self.get_num_leaves(),
            'num_internal_nodes': self.get_num_internal_nodes(),
            'num_nodes': self.get_num_nodes(),
            'max_leaf_value': self.get_max_leaf_value(),
            'min_leaf_value': self.get_min_leaf_value(),
            'average_leaf_value': self.get_average_leaf_value(),
            'median_leaf_value': self.get_median_leaf_value(),
            'leaf_value_counts': self.get_leaf_value_counts(),
            'leaf_value_frequencies': self.get_leaf_value_frequencies(),
            'leaf_value_distribution': self.get_leaf_value_distribution(),
            'leaf_value_statistics': self.get_leaf_value_statistics()
        }
    def get_tree_info(self):
        """Returns a comprehensive overview of the tree"""
        return {
            'structure': self.get_tree_structure(),
            'depth': self.get_tree_depth(),
            'size': self.get_tree_size(),
            'num_leaves': self.get_num_leaves(),
            'num_internal_nodes': self.get_num_internal_nodes(),
            'num_nodes': self.get_num_nodes(),
            'feature_importance': self.get_feature_importance(),
            'leaf_values': self.get_leaf_values(),
            'leaf_nodes': self.get_leaf_nodes(),
            'internal_nodes': self.get_internal_nodes(),
            'max_leaf_value': self.get_max_leaf_value(),
            'min_leaf_value': self.get_min_leaf_value(),
            'average_leaf_value': self.get_average_leaf_value(),
            'median_leaf_value': self.get_median_leaf_value(),
            'leaf_value_counts': self.get_leaf_value_counts(),
            'leaf_value_frequencies': self.get_leaf_value_frequencies(),
            'leaf_value_distribution': self.get_leaf_value_distribution(),
            'leaf_value_statistics': self.get_leaf_value_statistics()
        }
    def __str__(self):
        """Returns a string representation of the tree"""
        def node_to_str(node, depth=0):
            if node.value is not None:
                return "  " * depth + f"Leaf: {node.value}\n"
            left_str = node_to_str(node.left, depth + 1)
            right_str = node_to_str(node.right, depth + 1)
            return "  " * depth + f"[{node.feature} < {node.threshold}]\n" + left_str + right_str

        return node_to_str(self.root)
    def __repr__(self):
        """Returns a string representation of the tree for debugging"""
        return f"SimpleTree(max_depth={self.max_depth}, root={self.root})"
    def __len__(self):
        """Returns the number of nodes in the tree"""
        return self.get_num_nodes()
    def __contains__(self, item):
        """Checks if a value is in the tree (by checking leaf values)"""
        return item in self.get_leaf_values()
    def __getitem__(self, item):
        """Gets the leaf node for a specific value"""
        for leaf in self.get_leaf_nodes():
            if leaf.value == item:
                return leaf
        raise KeyError(f"Value {item} not found in the tree.")
    def __iter__(self):
        """Iterates over the leaf values of the tree"""
        for leaf in self.get_leaf_values():
            yield leaf
    def __eq__(self, other):
        """Checks if two trees are equal by comparing their structures and values"""
        if not isinstance(other, SimpleFairTree):
            return False
        return self.get_tree_structure() == other.get_tree_structure()
    def __ne__(self, other):
        """Checks if two trees are not equal by comparing their structures and values"""
        return not self.__eq__(other)
    def __hash__(self):
        """Returns a hash of the tree structure"""
        return hash(str(self.get_tree_structure()))
    def __bool__(self):
        """Checks if the tree is non-empty"""
        return self.root is not None
