### Fairness measures API
The aim of this API is to give an accessible and practical way to calculate fairness both in dataset and in classification algorithms.

## Usage
`from fairness_measures_api import fairness_measures_api`

`fairness_measures_api(self, d, r, t, y, h, g0, g1)`

where 
- *d* is a pandas dataset
- *r* is the name of the score variable
- *t* is the value of the threshold for positive score
- *y* is the name of the ground truth success variable
- *h* is the name of the predicted success variable
- *g0* is first group
- *g1* is second group

methods are 
- true_statistical_parity
- statistical_parity
- total_accuracy
- calibration
- ofi

all results range from -1 to 1

A usage example is provided: `fairness_api_example.py`
