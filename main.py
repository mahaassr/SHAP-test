import os
import sys
import argparse
import traceback
import audio
import config_file as cfg
import operator
from keras.models import load_model
from load_model import *
from split_model import *
import utilss
import shap
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from omnixai.explainers.vision import ShapImage
matplotlib.use("Agg")
#matplotlib.use('module://backend_interagg')
#set global seed
cfg.RANDOM_SEED
#np.random.seed(42)

def parseInputFiles(path, allowed_filetypes=['wav', 'flac', 'mp3', 'ogg', 'm4a']):

    # Add backslash to path if not present
    if not path.endswith(os.sep):
        path += os.sep

    # Get all files in directory with os.walk
    files = []
    for root, dirs, flist in os.walk(path):
        for f in flist:
            if len(f.rsplit('.', 1)) > 1 and f.rsplit('.', 1)[1].lower() in allowed_filetypes:
                files.append(os.path.join(root, f))

    print('Found {} files to analyze'.format(len(files)))

    return sorted(files)


def getRawAudioFromFile(fpath):

    # Open file
    sig, rate = audio.openAudioFile(fpath, cfg.SAMPLE_RATE)

    # Split into raw audio chunks
    chunks = audio.splitSignal(sig, rate, cfg.SIG_LENGTH, cfg.SIG_OVERLAP, cfg.SIG_MINLEN)

    return chunks
def analyzeFile(item, model):
    # item is FILE_LIST
    fpath = item[0]
    # Status
    print('Analyzing {}'.format(fpath), flush=True)
    chunks = getRawAudioFromFile(fpath)

    try:
        start, end = 0, cfg.SIG_LENGTH
        samples = []

        for c in range(len(chunks)):
            samples.append(chunks[c])
            start += cfg.SIG_LENGTH - cfg.SIG_OVERLAP
            end = start + cfg.SIG_LENGTH
            print('#####################',np.array(samples).shape)
            p = model.predict(samples)
            for i in range(len(samples)):

                # Get prediction
                pred = p[i]
                labels = utilss.loadLabels(LABELS_FILE)

                # Assign scores to labels
                p_labels = dict(zip(labels, pred))
                # Sort by score
                p_sorted = sorted(p_labels.items(), key=operator.itemgetter(1), reverse=True)

                # Store top 5 results and advance indicies
                results = p_sorted
            samples =[]
    except:
        print(traceback.format_exc(), flush=True)

        # Write error log
        msg = 'Error: Cannot analyze audio file {}.\n{}'.format(fpath, traceback.format_exc())
        print(msg, flush=True)
        return False

def predict_on_spec(spec):
    spec = spec[:,:, :,1]
    pred = shapModel.predict(spec)
    return pred

if __name__ == '__main__':

    LABELS_FILE = cfg.LABELS_FILE
    INPUT_PATH = cfg.INPUT_PATH


    FILE_LIST = cfg.INPUT_PATH
    LABELS = utilss.loadLabels(LABELS_FILE)


    # prepare data for the model
    chunks = getRawAudioFromFile(FILE_LIST)
    samples = []
    sample1 =[]
    model = load_model(cfg.MODEL_PATH,
                       custom_objects={'LinearSpecLayer': LinearSpecLayer})
    spec, shapModel = split_birdnet_keras_model(model)
    #for c in range(len(chunks)):
    samples.append(chunks[2])

    #preprocess_func = lambda x: np.expand_dims(x.to_numpy() / 255, axis=-1)

    spectrogram = spec.predict(tf.convert_to_tensor(samples))
    spec = np.repeat(spectrogram, 3, axis=3)
    spec = np.repeat(spectrogram, 3, axis=3)

    masker = shap.maskers.Image("inpaint_telea",spec[0].shape)
    print('masker', masker.shape,type(LABELS))
    explainer = shap.Explainer(predict_on_spec, masker, output_names=LABELS)

    shap_values = explainer(spec[0:1], max_evals=500, batch_size=50, outputs=shap.Explanation.argsort.flip[:1])
    shap.image_plot(shap_values)
    #shap image save to file
    plt.savefig(cfg.RESULTS_PATH+'shap2.png')
    print(sys.executable)



