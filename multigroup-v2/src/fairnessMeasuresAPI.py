import numpy as np

import pandas as pd


class fm:
    """
    A class to compute fairness measures for a dataset.
    """
    def __init__(self, dataset: pd.DataFrame, sensible_attribute: str, true_success: str, prediction: str, score=None, sigma=1/3, tau=1/3, gamma=1/3):
        
        # The pandas dataset
        self.d = dataset
        
        # The name of the attribute in d representing the sensible group
        self.g = sensible_attribute

        # array of sensible groups
        self.gs = sorted(self.d[self.g].unique())

        # The name of the attribute in d representing r
        self.r = score

        # The name of the attribute in d representing y
        self.y = true_success

        # The name of the attribute in d representing y_hat
        self.h = prediction

        # Weights for measures
        self.sigma = sigma
        self.tau = tau
        self.gamma = gamma

    def main():
        pass

    def getDataset(self):
        """
        Returns the dataset used for fairness measures.
        """
        return self.d
    
    def setDataset(self, dataset: pd.DataFrame):
        """
        Sets a new dataset for fairness measures.
        """
        self.d = dataset
        self.gs = sorted(self.d[self.g].unique())

    def getSensibleAttribute(self):
        """
        Returns the sensible attribute used for fairness measures.
        """
        return self.g
    
    def measure(self, measure:str, result:str):
        valid_measures = ['sp', 'ta', 'ca', 'ts']
        valid_results = ['matrix', 'maxdiff', 'abs', 'mean', 'variance', 'latex']

        if measure not in valid_measures:
            raise ValueError(f"Invalid measure: {measure}. Must be one of {valid_measures}.")
        if result not in valid_results:
            raise ValueError(f"Invalid result: {result}. Must be one of {valid_results}.")

        if measure == 'sp':
            matrix = self._matrix(self._2_statistical_parity)
        if measure == 'ta':
            matrix = self._matrix(self._2_total_accuracy)
        if measure == 'ca':
            matrix = self._matrix(self._2_calibration)
        if measure == 'ts':
            matrix = self._matrix(self._2_true_statistical_parity)
        
        if len(self.gs) == 2:
            if result == 'abs':
                return abs(matrix[0][1])
            else:
                return matrix[0][1]
        
        if result == 'matrix':
            return matrix

        if result == 'maxdiff':
            return np.max(np.abs(matrix))
        
        if result == 'mean':
            n = len(self.gs)
            values = []
            for i in range(n):
                for j in range(i+1, n):
                    values.append(abs(matrix[i][j]))
            return np.average(values)
        
        if result == 'variance':
            n = len(self.gs)
            values = []
            for i in range(n):
                for j in range(i+1, n):
                    values.append(abs(matrix[i][j]))
            return np.var(values)
        
        if result == 'latex':
            n = len(self.gs)
            latex_str = r"\begin{tabular}{r|r|r|r|r|r||r}" + "\n"
            for race in self.gs:
                latex_str += " & " + r"\textbf{" + race + "}"
            latex_str += " & " + r"\textbf{1-vs-all} \\" + "\n"+ r"\hline" + "\n"
            for i in range(n):
                latex_str += r"\textbf{" + str(self.gs[i]) + "} & " + " & ".join([f"{matrix[i][j]:.3f}" for j in range(n+1)]) + r" \\" + "\n" + r"\hline" + "\n"
            latex_str += r"& \multicolumn{5}{l}{\textbf{EP: " + f"{self.measure(measure=measure, result='mean'):.3f}, DM: {self.measure(measure=measure, result='variance'):.3f}" + r"}} \\" + "\n" + r"\hline" + "\n"
            latex_str += "\\end{tabular}"
            return latex_str

    def _matrix(self, measure):
        n = len(self.gs)
        matrix = np.zeros((n, n + 1))
        for i in range(n):
            for j in range(i+1, n):
                matrix[i][j] = measure(self.gs[i], self.gs[j])
                matrix[j][i] = -matrix[i][j]
            matrix[i][n] = measure(self.gs[i])

        return np.array(matrix)
    
    def _2_true_statistical_parity(self, g0, g1=None):
        if g1 is not None:
            total_g0 = self.d[self.d[self.g] == g0].shape[0]
            total_g1 = self.d[self.d[self.g] == g1].shape[0]
            success_g0 = self.d[(self.d[self.g] == g0) & self.d[self.y]].shape[0]
            success_g1 = self.d[(self.d[self.g] == g1) & self.d[self.y]].shape[0]
        else:
            total_g0 = self.d[self.d[self.g] == g0].shape[0]
            total_g1 = self.d[self.d[self.g] != g0].shape[0]
            success_g0 = self.d[(self.d[self.g] == g0) & self.d[self.y]].shape[0]
            success_g1 = self.d[(self.d[self.g] != g0) & self.d[self.y]].shape[0]

        if (total_g0 * total_g1 != 0):
            return success_g0/total_g0 - success_g1/total_g1
        else:
            return 0

    
    def _2_statistical_parity(self, g0, g1=None):
        if g1 is not None: 
            total_g0 = self.d[self.d[self.g] == g0].shape[0]
            total_g1 = self.d[self.d[self.g] == g1].shape[0]
            success_g0 = self.d[(self.d[self.g] == g0) & self.d[self.h]].shape[0]
            success_g1 = self.d[(self.d[self.g] == g1) & self.d[self.h]].shape[0]
        else:
            total_g0 = self.d[self.d[self.g] == g0].shape[0]
            total_g1 = self.d[self.d[self.g] != g0].shape[0]
            success_g0 = self.d[(self.d[self.g] == g0) & self.d[self.h]].shape[0]
            success_g1 = self.d[(self.d[self.g] != g0) & self.d[self.h]].shape[0]
            
        if (total_g0 * total_g1 != 0):
            return success_g0/total_g0 - success_g1/total_g1
        else:
            return 0

    
    def _2_total_accuracy(self, g0, g1=None):
        if g1 is not None: 
            total_g0 = self.d[self.d[self.g] == g0].shape[0]
            total_g1 = self.d[self.d[self.g] == g1].shape[0]
            tp_g0 = self.d[(self.d[self.g] == g0) & self.d[self.y] & self.d[self.h]].shape[0]
            tp_g1 = self.d[(self.d[self.g] == g1) & self.d[self.y] & self.d[self.h]].shape[0]
            tn_g0 = self.d[(self.d[self.g] == g0) & ~self.d[self.y] & ~self.d[self.h]].shape[0]
            tn_g1 = self.d[(self.d[self.g] == g1) & ~self.d[self.y] & ~self.d[self.h]].shape[0]
        else:
            total_g0 = self.d[self.d[self.g] == g0].shape[0]
            total_g1 = self.d[self.d[self.g] != g0].shape[0]
            tp_g0 = self.d[(self.d[self.g] == g0) & self.d[self.y] & self.d[self.h]].shape[0]
            tp_g1 = self.d[(self.d[self.g] != g0) & self.d[self.y] & self.d[self.h]].shape[0]
            tn_g0 = self.d[(self.d[self.g] == g0) & ~self.d[self.y] & ~self.d[self.h]].shape[0]
            tn_g1 = self.d[(self.d[self.g] != g0) & ~self.d[self.y] & ~self.d[self.h]].shape[0]

        if (total_g0 * total_g1 != 0):
            return ((tp_g0 + tn_g0)/total_g0) - ((tp_g1 + tn_g1)/total_g1)
        else:
            return 0
        
    def _2_calibration(self, g0, g1=None):
        c = 0
        t = 0
        rs = list(map(int, sorted(self.d[self.r].unique())))

        for single_r in rs:
            subset = self.d[self.d[self.r] == single_r]
            if g1 is not None:
                total_g0 = subset[subset[self.g] == g0].shape[0]
                total_g1 = subset[subset[self.g] == g1].shape[0]
                success_g0 = subset[(subset[self.g] == g0) & subset[self.y]].shape[0]
                success_g1 = subset[(subset[self.g] == g1) & subset[self.y]].shape[0]
            else:
                total_g0 = subset[subset[self.g] == g0].shape[0]
                total_g1 = subset[subset[self.g] != g0].shape[0]
                success_g0 = subset[(subset[self.g] == g0) & subset[self.y]].shape[0]
                success_g1 = subset[(subset[self.g] != g0) & subset[self.y]].shape[0]

            w = subset.shape[0]
            d = (single_r > 3) * 2 - 1

            gs0 = success_g0/total_g0 if total_g0 > 0 else 0
            gs1 = success_g1/total_g1 if total_g1 > 0 else 0

            c += d * w * (gs0 - gs1)
            t += w
        return c/t if t != 0 else 0
    
    if __name__ == "__main__":
        main()