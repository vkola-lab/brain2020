import numpy as np 

def normalization(scan):
    scan = (scan - np.mean(scan)) / np.std(scan)
    return scan

def clip(scan):
    return np.clip(scan, -1, 2.5)

if __name__ == "__main__":
    data = np.load("sample.npy")
    data = normalization(data)
    data = clip(data)
    


