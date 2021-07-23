# this script contains functions used to visualise the relevant words for predicting cognitive presence phases as heatmaps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, transforms
import json

sta = 1

# create analysis results from the analyzers of SmoothGrad method in the iNNvestigate lib
def generateanalisys(xdata,ydata,modelnoactiv,modelactiv,analyzers):

  test_sample_preds = [None]*len(xdata)

  # a variable to store analysis results.
  analysis = []

  for i in range(len(xdata)):

      x, y = xdata[i], ydata[i]
      x = np.expand_dims(x, axis=0)    

      presm = modelnoactiv.predict_on_batch(x)[0] #forward pass without softmax
      prob = modelactiv.predict_on_batch(x)[0] #forward pass with softmax
      y_hat = prob.argmax()
      test_sample_preds[i] = y_hat
      
      a = np.squeeze(analyzers.analyze(x))
      a = np.sum(a, axis=1).flatten()
      analysis.append(a)
  
  return analysis


# plot heatmaps of highlighted words
def plot_text_heatmap(words, scores, number, title="", width=10, height=0.2, verbose=0, max_word_per_line=20):
    fig = plt.figure(figsize=(width, height))
    
    ax = plt.gca()

    ax.set_title(title, loc='left')
    tokens = words
    if verbose > 0:
        print('len words : %d | len scores : %d' % (len(words), len(scores)))

    cmap = plt.cm.ScalarMappable(cmap=cm.bwr)
    cmap.set_clim(0, 1)
    
    canvas = ax.figure.canvas
    t = ax.transData

    # normalize scores to the followings:
    # - low relevance scores in [0, 0.5]
    # - high relevance scores in (0.5, 1]
    normalized_scores = 0.5 * scores / np.max(np.abs(scores)) + 0.5
    
    if verbose > 1:
        print('Raw score')
        print(scores)
        print('Normalized score')
        print(normalized_scores)

    # make sure the heatmap doesn't overlap with the title
    loc_y = -0.2

    for i, token in enumerate(tokens):
        *rgb, _ = cmap.to_rgba(normalized_scores[i], bytes=True)
        color = '#%02x%02x%02x' % tuple(rgb)
        
        text = ax.text(0.0, loc_y, token,
                       bbox={
                           'facecolor': color,
                           'pad': 5.0,
                           'linewidth': 1,
                           'boxstyle': 'round,pad=0.5'
                       }, transform=t)

        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        
        # create a new line if the line exceeds the length
        if (i+1) % max_word_per_line == 0:
            loc_y = loc_y -  2.5
            t = ax.transData
        else:
            t = transforms.offset_copy(text._transform, x=ex.width+15, units='dots')

    if verbose == 0:
        ax.axis('off')
    
    # save the plot as pngs in the xai dir
    plt.savefig('xai_sample/' + str(number) + '.png')
    
    # plt.show()  #can try to return plt and save in the main.py as pngs 