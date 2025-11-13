import numpy as np

def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

def information_gain(X, y, attr_index):
    total_entropy = entropy(y)
    values, counts = np.unique([row[attr_index] for row in X], return_counts=True)
    weighted_entropy = 0.0
    for val, count in zip(values, counts):
        subset_y = [y[i] for i in range(len(X)) if X[i][attr_index] == val]
        weighted_entropy += (count / len(X)) * entropy(subset_y)
    return total_entropy - weighted_entropy

class DecisionTree:

    def __init__(self, X, y, threshold=0.75, max_depth=None):
        self.threshold = threshold
        self.max_depth = max_depth
        self.default_label = self.get_most_common_label(y)
        self.tree = self.build_tree(X, y, depth=0)

    def get_most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def build_tree(self, X, y, depth):
        if len(set(y)) == 1:
            return {"label": y[0]}

        values, counts = np.unique(y, return_counts=True)
        most_common_label = values[np.argmax(counts)]
        if counts.max() / len(y) >= self.threshold:
            return {"label": most_common_label}

        if self.max_depth is not None and depth >= self.max_depth:
            return {"label": most_common_label}

        num_attrs = len(X[0])
        gains = [information_gain(X, y, i) for i in range(num_attrs)]
        best_attr = np.argmax(gains)

        tree = {"attribute": best_attr, "branches": {}}
        attr_values = set(row[best_attr] for row in X)

        for val in attr_values:
            subset_X = [row for i, row in enumerate(X) if X[i][best_attr] == val]
            subset_y = [y[i] for i in range(len(X)) if X[i][best_attr] == val]
            tree["branches"][val] = self.build_tree(subset_X, subset_y, depth + 1)

        return tree

    def predict(self, x):
        node = self.tree
        while "label" not in node:
            attr = node["attribute"]
            val = x[attr]
            if val not in node["branches"]:
                return self.default_label 
            node = node["branches"][val]
        return node["label"]

def train_decision_tree(X, y):
    return DecisionTree(X, y, threshold=0.75)
