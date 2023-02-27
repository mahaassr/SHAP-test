import matplotlib.pyplot as plt

import config_file as cfg

def loadLabels(labels_file):

    labels = []
    with open(labels_file, 'r') as lfile:
        for line in lfile.readlines():
            labels.append(line.replace('\n', ''))

    return labels
def new_name(output_prediction):
    output = output_prediction.reshape(output_prediction.shape[1])
    print(output.shape)
    labels = loadLabels(cfg.LABELS_FILE)
    #print(labels.shape)
    plt.bar(labels, output, width = 3)
    plt.xticks(labels, output, rotation=90)
    plt.show()