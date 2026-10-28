import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class Node:
    # Decision tree node
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, bias=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.bias = bias

    def predict_one(self, row):
        # Classifies a single example by traversing the tree
        if self.value is not None:
            return self.value
        if row[self.feature] < self.threshold:
            return self.left.predict_one(row)
        return self.right.predict_one(row)

class SimpleFairTree:
    # Class for creating a simplified Decision Tree
    def __init__(self, max_depth=5, criterion='entropy', splitter='best', random_state=None, fairness_measure=None, fairness_weights=None, fairness_attributes=None):
        self._criterions = ['entropy', 'gini', 'classification_error']
        self._splitters = ['best', 'fair']
        self._fairness_measures = ['sp']

        self.root = None
        self.max_depth = max_depth
        self.fairness_measure_id = fairness_measure
        self.criterion = eval('self.' + criterion)
        self.splitter = splitter
        self.fair_measure = fairness_measure
        if fairness_weights is not None and not isinstance(fairness_weights, list):
            self.fair_weights = [fairness_weights] * len(fairness_attributes)
        else:
            self.fair_weights = fairness_weights
        self.fair_attributes = fairness_attributes
        np.random.seed(random_state)

    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs))
    
    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def classification_error(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.max(probs)

    def split_dataset(self, df, feature, threshold):
        left_mask = df[feature] < threshold
        right_mask = ~left_mask
        return df[left_mask], df[right_mask]
    
    def fairness_measure(self, df, fairness_attribute, target_col):
        groups = df[fairness_attribute].unique()        
        g0 = groups[0]
        g1 = groups[1]

        if self.fairness_measure_id == 'sp':
            total_g0 = len(df[df[fairness_attribute] == g0])
            total_g1 = len(df[df[fairness_attribute] == g1])
            success_g0 = len(df[(df[fairness_attribute] == g0) & df['y_hat']])
            success_g1 = len(df[(df[fairness_attribute] == g1) & df['y_hat']])
            bias = abs((success_g0 / total_g0) - (success_g1 / total_g1))

        if self.fairness_measure_id == 'ta':
            total_g0 = len(df[df[fairness_attribute] == g0])
            total_g1 = len(df[df[fairness_attribute] == g1])
            tp_g0 = len(df[(df[fairness_attribute] == g0) & df[target_col] & df['y_hat']])
            tp_g1 = len(df[(df[fairness_attribute] == g1) & df[target_col] & df['y_hat']])
            tn_g0 = len(df[(df[fairness_attribute] == g0) & ~df[target_col] & ~df['y_hat']])
            tn_g1 = len(df[(df[fairness_attribute] == g1) & ~df[target_col] & ~df['y_hat']])
            bias = abs(((tp_g0 + tn_g0) / total_g0) - ((tp_g1 + tn_g1) / total_g1))
        return bias

    def find_best_split(self, dataset, target_col):
        """Finds the best feature and threshold"""

        best_feature, best_threshold, best_gain, bias, rec_bias = None, None, None, 0, 0
        features = dataset.columns.drop(target_col)
        features = features.to_numpy()
        np.random.shuffle(features)

        for feature in features:
            thresholds = dataset[feature].unique()
            for threshold in thresholds:
                left_df, right_df = self.split_dataset(dataset, feature, threshold)
                if len(left_df) == 0 or len(right_df) == 0:
                    continue

                left_impurity, right_impurity = self.criterion(left_df[target_col]), self.criterion(right_df[target_col])
                left_weight, right_weight = len(left_df) / len(dataset), len(right_df) / len(dataset)
                if self.splitter == 'fair':
                    bias = 0
                    for fair_attribute, fair_weight in zip(self.fair_attributes, self.fair_weights):
                        if dataset[fair_attribute].nunique() == 2:
                            if left_df[target_col].mode()[0] != right_df[target_col].mode()[0]:
                                df = dataset.copy()
                                df.loc[left_df.index, 'y_hat'] = left_df[target_col].mode()[0]
                                df.loc[right_df.index, 'y_hat'] = right_df[target_col].mode()[0]
                                df['y_hat'] = df['y_hat'].astype(int)
                                bias += fair_weight * self.fairness_measure(df, fair_attribute, target_col)

                gain = left_weight * left_impurity + right_weight * right_impurity + bias
 
                if best_gain is None or gain < best_gain:
                    best_feature, best_threshold, best_gain, rec_bias = feature, threshold, gain, bias

        return best_feature, best_threshold, rec_bias

    def build_tree(self, dataset: pd.DataFrame, target_col: str, depth=0):
        # Recursively builds the decision tree while keeping the full dataset for fairness calculations
        # Base case: if all samples belong to the same class or max depth is reached
        if len(dataset[target_col].unique()) == 1 or depth >= self.max_depth:
            return Node(value=dataset[target_col].mode()[0])
        # Find best feature and threshold using the full dataset for fairness penalty
        feature, threshold, bias = self.find_best_split(dataset, target_col)
        if feature is None:
            return Node(value=dataset[target_col].mode()[0], bias=bias)

        # Split dataset and create child nodes recursively
        left_subset, right_subset = self.split_dataset(dataset, feature, threshold)
        left_node = self.build_tree(left_subset, target_col, depth + 1)
        right_node = self.build_tree(right_subset, target_col, depth + 1)

        # If both child nodes are leaf nodes with the same value, return a leaf node
        if left_node.value is not None and right_node.value is not None and left_node.value == right_node.value:
            return Node(value=left_node.value, bias=bias)

        # Create and return the current node
        return Node(feature, threshold, left_node, right_node, bias=bias)

    def fit(self, X, target_col):
        # Trains the model using feature DataFrame and class labels

        self.feature_types = {col: str(X[col].dtype) for col in X.columns}
        self.root = self.build_tree(dataset=X, target_col=target_col)

    def predict(self, df):
        # Classifies a dataset automatically
        return df.apply(lambda row: self.root.predict_one(row), axis=1)
    
    def explain(self):
        # Prints the tree structure
        def print_tree(node, depth=0, prefix="Root: "):
            if node.value is not None:
                print("  " * depth + f"{prefix} Class {node.value}")
                return
            print(f"  " * depth + f"{prefix} [{node.feature} < {node.threshold}] Bias: {node.bias}")
            print_tree(node.left, depth + 1, prefix="Left: ")
            print_tree(node.right, depth + 1, prefix="Right: ")

        print_tree(self.root)

    def get_feature_importance(self):
        # Returns the importance of each feature in the tree
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

    def plot_tree(self, filename=None):
        # Plots the tree structure graphically using networkx and matplotlib. If filename is given, saves the plot to file.
        def add_edges(graph, node, parent=None, label=""):
            if node is None:
                return
            node_id = id(node)
            if parent is not None:
                graph.add_edge(parent, node_id, label=label)
            if node.value is not None:
                graph.add_node(node_id, label=f"Class {node.value}", value=node.value)
            else:
                # Check if the feature is boolean in the training data
                # Assume self.feature_types is a dict {feature_name: dtype} set during fit
                is_bool = False
                if hasattr(self, "feature_types"):
                    is_bool = self.feature_types.get(node.feature, None) == "bool"
                # If not available, fallback to threshold == 0.5 as a heuristic
                if is_bool or node.threshold == 0.5:
                    graph.add_node(node_id, label=f"{node.feature}", value=None)
                    add_edges(graph, node.right, node_id, "True")
                    add_edges(graph, node.left, node_id, "False")

                else:
                    graph.add_node(node_id, label=f"{node.feature} < {node.threshold}\nBias: {round(node.bias, 3) if node.bias is not None else 'N/A'}", value=None)
                    add_edges(graph, node.left, node_id, "True")
                    add_edges(graph, node.right, node_id, "False")

        G = nx.DiGraph()
        add_edges(G, self.root)

        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        for k in pos:
            pos[k] = (pos[k][0] * 50, pos[k][1]) 

        labels = nx.get_node_attributes(G, 'label')
        edge_labels = nx.get_edge_attributes(G, 'label')
        node_values = nx.get_node_attributes(G, 'value')

        # Assign colors: 
        # - internal nodes (no class): 'lightblue'
        # - class 0: 'lightgreen'
        # - class 1: 'salmon'
        node_colors = []
        for node in G.nodes():
            value = node_values.get(node, None)
            if value is None:
                node_colors.append('lightblue')
            elif value == 0:
                node_colors.append('salmon')
            elif value == 1:
                node_colors.append('lightgreen')
            else:
                node_colors.append('gray')  # fallback for other classes

        plt.figure(figsize=(100, 30))

        nx.draw_networkx_nodes(G, pos, node_size=20000, node_color=node_colors, node_shape='s')
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')
        nx.draw_networkx_edges(G, pos, arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
        plt.title("Decision Tree Structure")
        plt.axis('off')
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()
