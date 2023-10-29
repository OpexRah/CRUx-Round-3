#importing all necessay libs

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time 
import schedule
import joblib
import time
import numpy as np
import pandas as pd
import itertools
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re
import string
from nltk.stem import WordNetLemmatizer
from deep_translator import GoogleTranslator

DRIVER_PATH = 'chromedriver.exe'
wd = webdriver.Chrome()
wd.get("https://www.youtube.com/")



filename = "PAC_Multi_Data_model_1.sav"
pac = pickle.load(open(filename, 'rb'))

tfidf_vectorizer = joblib.load("tfidf_multi.pkl")

stopword_list = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


def predictor(inp):
  data = [str(inp)]
  _test = re.sub(r'[^a-zA-Z]', ' ', data[0])
  _test = _test.lower()
  _test = _test.split()
  _test = [lemmatizer.lemmatize(word) for word in _test if not word in set(stopword_list)]
  text_processed = [' '.join(_test)]
  print(text_processed)
  tfidf_reviews=tfidf_vectorizer.transform(text_processed)
  result_probs = pac._predict_proba_lr(tfidf_reviews[0])
  print("Educational : ", result_probs[0][0]*100)
  print("Music : ", result_probs[0][1]*100)
  print("Sports : ", result_probs[0][2]*100)
  print("Gaming : ", result_probs[0][3]*100)
  print("Movies : ", result_probs[0][4]*100,'\n\n')
  result = pac.predict(tfidf_reviews[0])
  #print(result)
  if result_probs[0][result][0]>=0.4:
    if result == 0:
      print("Educational", result_probs[0][result][0]*100,"% confidence")
    elif result == 1:
      print("Music", result_probs[0][result][0]*100,"% confidence")
    elif result == 2:
      print("Sports", result_probs[0][result][0]*100,"% confidence")
    elif result == 3:
      print("Gaming", result_probs[0][result][0]*100,"% confidence")
    elif result == 4:
      print("Movies", result_probs[0][result][0]*100,"% confidence")
  else:
    print("Unable to classify\n Most close prediction: ", ResToClass(result), result_probs[0][result][0]*100)

def url_check():
    title = wd.title
    #perform translation to english
    title = GoogleTranslator(source='auto', target='en').translate(text=title)
    print(title)
    if title[-7:] == "YouTube":
        if wd.current_url == "https://www.youtube.com/" or wd.current_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ&ab_channel=RickAstley" or wd.current_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ":
            pass
        else:
            predictor(title[:-7])
            
def ResToClass(result):
  if result == 0:
    return("Educational")
  elif result == 1:
    return("Music")
  elif result == 2:
    return("Sports")
  elif result == 3:
    return("Gaming")
  elif result == 4:
    return("Movies")


schedule.every(2).seconds.do(url_check)
  
while True:
    schedule.run_pending()
    time.sleep(1)



