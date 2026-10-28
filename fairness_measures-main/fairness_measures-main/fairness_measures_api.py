
class fairness_measures_api:
    def __init__(self, d, g, y, r, h):
        # The pandas dataset
        self.d = d
        
        # The name of the attribute in d representing the sensible group
        self.g = g

        # array of sensible groups
        
        self.gs = sorted(self.d[self.g].unique())

        # The name of the attribute in d representing r
        self.r = r

        # The name of the attribute in d representing y
        self.y = y

        # The name of the attribute in d representing y_hat
        self.h = h

    def main():
        pass

    def _generate_permutations(self, measure):
        matrix = [[0 for _ in range(len(self.gs))] for _ in range(len(self.gs))]
        array = [0 for _ in range(len(self.gs))]
        i = 0
        j = 0
        n = 0
        min = 2
        max = 0
        for g0 in self.gs:
            for g1 in self.gs:
                if i != j:
                    matrix[i][j] = measure(g0, g1)
                    if abs(matrix[i][j]) < min:
                        min = matrix[i][j]
                        g0min = g0
                        g1min = g1
                    if abs(matrix[i][j]) > max:
                        max = matrix[i][j]
                        g0max = g0
                        g1max = g1
                j = j + 1
                if j == len(self.gs):
                    j = 0
                    i = i + 1
        i = 0
        for g0 in self.gs:
            array[i] = measure(g0)
            i = i + 1
        
        mean = 0
        mean_array = 0
        variance = 0
        variance_array = 0
        n = 0
        l = len(self.gs)
        
        for j in range(l):
            for i in range(j+1,l):
                mean += abs(matrix[i][j])
                n += 1
        mean /= n

        for j in range(l):
            for i in range(j+1,l):
                variance += (abs(matrix[i][j]) - mean)**2
        variance /= n

        for i in range(l):
            mean_array += abs(array[i])

        mean_array /= l

        for i in range(l):
            variance_array += (abs(array[i]) - mean_array)**2

        variance_array /= l
        
        return (matrix, array, 
                mean, variance, 
                mean_array, variance_array, 
                min, max, 
                g0min, g1min, 
                g0max, g1max)
    
    
    def true_statistical_parity(self):
        return self._generate_permutations(self._2_true_statistical_parity)
    
    def statistical_parity(self):
        return self._generate_permutations(self._2_statistical_parity)
    
    def total_accuracy(self):
        return self._generate_permutations(self._2_total_accuracy)
    
    def calibration(self):
        return self._generate_permutations(self._2_calibration)
    
    def ofi(self):
        return self._generate_permutations(self._2_ofi)
    

    def _2_true_statistical_parity(self, g0, g1=None):
        if g1 is not None:
            total_g0 = len(self.d[self.d[self.g] == g0])
            total_g1 = len(self.d[self.d[self.g] == g1])
            success_g0 = len(self.d[(self.d[self.g] == g0) & self.d[self.y]])
            success_g1 = len(self.d[(self.d[self.g] == g1) & self.d[self.y]])
        else:
            total_g0 = len(self.d[self.d[self.g] == g0])
            total_g1 = len(self.d[self.d[self.g] != g0])
            success_g0 = len(self.d[(self.d[self.g] == g0) & self.d[self.y]])
            success_g1 = len(self.d[(self.d[self.g] != g0) & self.d[self.y]])
        return success_g0/total_g0 - success_g1/total_g1

    
    def _2_statistical_parity(self, g0, g1=None):
        if g1 is not None: 
            total_g0 = len(self.d[self.d[self.g] == g0])
            total_g1 = len(self.d[self.d[self.g] == g1])
            success_g0 = len(self.d[(self.d[self.g] == g0) & self.d[self.h]])
            success_g1 = len(self.d[(self.d[self.g] == g1) & self.d[self.h]])
        else:
            total_g0 = len(self.d[self.d[self.g] == g0])
            total_g1 = len(self.d[self.d[self.g] != g0])
            success_g0 = len(self.d[(self.d[self.g] == g0) & self.d[self.h]])
            success_g1 = len(self.d[(self.d[self.g] != g0) & self.d[self.h]])
            
        return success_g0/total_g0 - success_g1/total_g1
    
    def _2_total_accuracy(self, g0, g1=None):
        if g1 is not None: 
            total_g0 = len(self.d[self.d[self.g] == g0])
            total_g1 = len(self.d[self.d[self.g] == g1])
            tp_g0 = len(self.d[(self.d[self.g] == g0) & self.d[self.y] & self.d[self.h]])
            tp_g1 = len(self.d[(self.d[self.g] == g1) & self.d[self.y] & self.d[self.h]])
            tn_g0 = len(self.d[(self.d[self.g] == g0) & ~self.d[self.y] & ~self.d[self.h]])
            tn_g1 = len(self.d[(self.d[self.g] == g1) & ~self.d[self.y] & ~self.d[self.h]])
        else:
            total_g0 = len(self.d[self.d[self.g] == g0])
            total_g1 = len(self.d[self.d[self.g] != g0])
            tp_g0 = len(self.d[(self.d[self.g] == g0) & self.d[self.y] & self.d[self.h]])
            tp_g1 = len(self.d[(self.d[self.g] != g0) & self.d[self.y] & self.d[self.h]])
            tn_g0 = len(self.d[(self.d[self.g] == g0) & ~self.d[self.y] & ~self.d[self.h]])
            tn_g1 = len(self.d[(self.d[self.g] != g0) & ~self.d[self.y] & ~self.d[self.h]])

        return ((tp_g0 + tn_g0)/total_g0) - ((tp_g1 + tn_g1)/total_g1) 
        
    def _2_calibration(self, g0, g1=None):
        c = 0
        t = 0
        rs = list(map(int, sorted(self.d[self.r].unique())))

        for single_r in rs:
            subset = self.d[self.d[self.r] == single_r]
            if g1 is not None:
                total_g0 = len(subset[subset[self.g] == g0])
                total_g1 = len(subset[subset[self.g] == g1])
                success_g0 = len(subset[(subset[self.g] == g0) & subset[self.y]])
                success_g1 = len(subset[(subset[self.g] == g1) & subset[self.y]])
            else:
                total_g0 = len(subset[subset[self.g] == g0])
                total_g1 = len(subset[subset[self.g] != g0])
                success_g0 = len(subset[(subset[self.g] == g0) & subset[self.y]])
                success_g1 = len(subset[(subset[self.g] != g0) & subset[self.y]])

            if (total_g0 * total_g1 != 0):
                w = len(subset)

                gs0 = success_g0/total_g0 
                gs1 = success_g1/total_g1

                # TODO modify the dir (f) function
                dir = -1 if single_r < 10 else 1
                c += dir * w * (gs0 - gs1)
                t += w
        return c/t if t != 0 else 0
        
    def _2_ofi(self, w_sp = 1/3, w_ta = 1/3, w_ca = 1/3):
        return w_sp * self.statistical_parity() + w_ta * self.total_accuracy() + w_ca * self.calibration()
    
    if __name__ == "__main__":
        main()