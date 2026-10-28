import numpy as np

import pandas as pd


class fm:
    """
    A class to compute fairness measures for a dataset.
    """
    def __init__(self, dataset: pd.DataFrame, sensible_attribute: str, true_success: str, prediction: str, score=None, success_func=None, sigma=1/3, tau=1/3, gamma=1/3):
        
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

        # The name of the attribute in d representing r function
        self.s = success_func

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
    
    def measure(self, measure:str):        
        dict_measures = {'sp': 'self._statistical_parity(self.gs[0], self.gs[1])', 
                         'ta': 'self._total_accuracy(self.gs[0], self.gs[1])', 
                         'ca': 'self._calibration(self.gs[0], self.gs[1])', 
                         'ts': 'self._true_statistical_parity(self.gs[0], self.gs[1])'
                        }
        return eval(dict_measures[measure])

    def _true_statistical_parity(self, g0, g1):
        total_g0 = self.d[self.d[self.g] == g0].shape[0]
        total_g1 = self.d[self.d[self.g] == g1].shape[0]
        success_g0 = self.d[(self.d[self.g] == g0) & self.d[self.y]].shape[0]
        success_g1 = self.d[(self.d[self.g] == g1) & self.d[self.y]].shape[0]

        if (total_g0 * total_g1 != 0):
            return success_g0/total_g0 - success_g1/total_g1
        else:
            return 0


    def _statistical_parity(self, g0, g1):
        total_g0 = self.d[self.d[self.g] == g0].shape[0]
        total_g1 = self.d[self.d[self.g] == g1].shape[0]
        success_g0 = self.d[(self.d[self.g] == g0) & self.d[self.h]].shape[0]
        success_g1 = self.d[(self.d[self.g] == g1) & self.d[self.h]].shape[0]

        if (total_g0 * total_g1 != 0):
            return success_g0/total_g0 - success_g1/total_g1
        else:
            return 0


    def _total_accuracy(self, g0, g1):
        total_g0 = self.d[self.d[self.g] == g0].shape[0]
        total_g1 = self.d[self.d[self.g] == g1].shape[0]
        tp_g0 = self.d[(self.d[self.g] == g0) & self.d[self.y] & self.d[self.h]].shape[0]
        tp_g1 = self.d[(self.d[self.g] == g1) & self.d[self.y] & self.d[self.h]].shape[0]
        tn_g0 = self.d[(self.d[self.g] == g0) & ~self.d[self.y] & ~self.d[self.h]].shape[0]
        tn_g1 = self.d[(self.d[self.g] == g1) & ~self.d[self.y] & ~self.d[self.h]].shape[0]


        if (total_g0 * total_g1 != 0):
            return ((tp_g0 + tn_g0)/total_g0) - ((tp_g1 + tn_g1)/total_g1)
        else:
            return 0

    def _calibration(self, g0, g1):
        c = 0
        rs = list(map(int, sorted(self.d[self.r].unique())))

        for single_r in rs:
            subset = self.d[self.d[self.r] == single_r]
            total_g0 = subset[subset[self.g] == g0].shape[0]
            total_g1 = subset[subset[self.g] == g1].shape[0]
            success_g0 = subset[(subset[self.g] == g0) & subset[self.y]].shape[0]
            success_g1 = subset[(subset[self.g] == g1) & subset[self.y]].shape[0]

            w = subset.shape[0] / self.d.shape[0]
            d = self.s(single_r) * 2 - 1

            gs0 = success_g0/total_g0 if (total_g0 != 0) else 0
            gs1 = success_g1/total_g1 if (total_g1 != 0) else 0

            c += d * w * (gs0 - gs1)
        return c
    
    if __name__ == "__main__":
        main()