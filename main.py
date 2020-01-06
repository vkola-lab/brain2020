from utils import read_json, data_split


config = read_json('./config.json')
print(config)

data_split(repe_time=5)

for repe_idx in range(repe_time):
    FCN.train()
    FCN.test()
    MLP.train()
    MLP.test()

