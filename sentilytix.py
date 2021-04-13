import sys
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from transformers import pipeline
from bs4 import BeautifulSoup
import glob

print('Evaluating sentiments in HTML documents in folder: '+sys.argv[1])

s = pipeline('sentiment-analysis')

fileNames = glob.glob(sys.argv[1]+"/*.htm*")

for fileName in fileNames:

    html = open(fileName)
    html = html.read()
    html = BeautifulSoup(html)
    ps = html.findAll('p')

    average_score = 0
    for p in ps:
        p = p.text
        score = s(p)
        score = score[0]
        if score['label'] == 'NEGATIVE':
            score = -score['score']
        else:
            score = score['score']
        average_score += score
    average_score = average_score/len(ps)
    print(fileName, average_score)
