import pandas as pd
import numpy as np

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

class SimpleTree:
    """Class for creating a simplified Decision Tree"""
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs))

    def split_dataset(self, df, feature, threshold):
        left_mask = df[feature] < threshold
        right_mask = ~left_mask
        return df[left_mask], df[right_mask]

    def find_best_split(self, df, target_col):
        best_feature, best_threshold, best_gain = None, None, 0
        current_entropy = self.entropy(df[target_col])

        for feature in df.columns.drop(target_col):
            thresholds = df[feature].unique()
            for threshold in thresholds:
                left_df, right_df = self.split_dataset(df, feature, threshold)
                if len(left_df) == 0 or len(right_df) == 0:
                    continue

                left_entropy, right_entropy = self.entropy(left_df[target_col]), self.entropy(right_df[target_col])
                gain = current_entropy - (len(left_df)/len(df)*left_entropy + len(right_df)/len(df)*right_entropy)
                gain = gain/current_entropy if current_entropy > 0 else 0
                if gain > best_gain:
                    best_feature, best_threshold, best_gain = feature, threshold, gain

        return best_feature, best_threshold

    def build_tree(self, df, target_col, depth=0):
        if len(df[target_col].unique()) == 1 or depth >= self.max_depth:
            return Node(value=df[target_col].mode()[0])

        feature, threshold = self.find_best_split(df, target_col)
        if feature is None:
            return Node(value=df[target_col].mode()[0])
        
        left_df, right_df = self.split_dataset(df, feature, threshold)
        left_node = self.build_tree(left_df, target_col, depth+1)
        right_node = self.build_tree(right_df, target_col, depth+1)

        return Node(feature, threshold, left_node, right_node)

    def fit(self, X, y):    
        """Trains the model using feature DataFrame and class labels"""
        df = X.copy()
        df['target'] = y  
        self.root = self.build_tree(df, 'target')

    def predict(self, df):
        """Classifies a dataset automatically"""
        return df.apply(lambda row: self.root.predict_one(row), axis=1)
    
    def explain(self):
        """Prints the tree structure"""
        def print_tree(node, depth=0):
            if node.value is not None:
                print("  " * depth + f"Leaf: {node.value}")
                return
            print("  " * depth + f"[{node.feature} < {node.threshold}]")
            print_tree(node.left, depth + 1)
            print_tree(node.right, depth + 1)

        print_tree(self.root)
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
        if not isinstance(other, SimpleTree):
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
