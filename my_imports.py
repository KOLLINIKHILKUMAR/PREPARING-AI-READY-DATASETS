import os
import re
import json
import nltk
import gensim
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVC
import PIL
import seaborn as sns
import pytesseract
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.shortcuts import render
from gensim.models import KeyedVectors
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import VotingClassifier
from multiprocessing.sharedctypes import Value
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask,request,render_template,Response,send_file
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
import sys
import os
import pandas as pd
from src.exception import CustomException
from src.components.data_transformation import know_type,Preprocessing,WordEmbedding,Classifier
from nltk.stem import SnowballStemmer
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
stemmer=SnowballStemmer("english")
import nltk
# nltk.download('stopwords')
nltk_stopwords = set(nltk.corpus.stopwords.words('english'))