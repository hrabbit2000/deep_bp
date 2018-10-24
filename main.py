import numpy as np
import deep_bp as dbp

def main():
# display some lines
    bp = dbp.DeepBP([2, 3, 1])
    bp.init_parameters()
    bp.weights = [np.array([(0.1, 0.2), (0.2, 0.3), (0.3, 0.4)]), np.array([(0.5, 0.6, 0.7)])]
    bp.biases = [np.array([(0.3, 0.4, 0.5)]), np.array([(0.2,)])]
    inputs = [[1, 2]]
    ds = [[0.5]]
    bp.batch_process(inputs, ds)


if __name__ == "__main__": main()
