import numpy as np

class CalibrationDataGen:
    def __init__(self, npy_path="/Users/alpha/Downloads/selfRepo/lodcp/data/calibration_samples.npy"):
        self.data = np.load(npy_path)
        self.num_samples = self.data.shape[0]
        self.idx=0

    def __call__(self):
        for i in range(self.num_samples):
            yield self.data[i]
