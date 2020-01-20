import subprocess
import json
import os

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config

def function(fil_num1, fil_num2, alpha, batch_size, learning_rate, train_epochs, roi_threshold):

    filename = './mlp_configuration.json'
    data = read_json(filename)
    data['fil_num1'] = fil_num1[0]
    data['fil_num2'] = fil_num2[0]
    data['alpha'] = alpha[0]
    data['batch_size'] = batch_size[0]
    data['learning_rate'] = learning_rate[0]
    data['train_epochs'] = train_epochs[0]
    data['roi_threshold'] = roi_threshold[0]

    os.remove(filename)

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

    command = ['/home/sq/.conda/envs/RL/bin/python', 'sklearn_classifiers.py', filename]

    output = subprocess.Popen(command, stdout=subprocess.PIPE)
    val = str(output.communicate()[0])
    print(val)
    val = val[val.index('$')+1:val.index('$$')]

    return float(val)

def main(job_id, params):
    print(params)
    return function(params['fil_num1'],
                    params['fil_num2'],
                    params['alpha'],
                    params['batch_size'],
                    params['learning_rate'],
                    params['train_epochs'],
                    params['roi_threshold'])
