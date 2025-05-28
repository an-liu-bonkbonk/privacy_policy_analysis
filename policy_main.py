## Import
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textstat
import textblob
from textblob import TextBlob
import re
import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.stats
from scipy.stats import ttest_rel
import seaborn as sns

"""
---------------------------------------------------------------------------------
Part 1: Computing readability metrics:
Flesch Reading Ease score
SMOG index
Coleman-Liau index
Flesch-Kincaid Grade
Dale-Chall readability score
"""

## Load the files (the folder)
base_path = "/policies_fromseg_custom/"
files = os.listdir(base_path)

## Compute the required readability metrics

# Define a function that gets all required metrics
def get_metrics(files):
    metrics = [] # Store all the results (nested dictionaries) here
    for file in files:
        complete_path = base_path + file

        try:
            with open(complete_path, "r") as rd_file:
                file_content = rd_file.read()

            FRE = textstat.flesch_reading_ease(file_content)
            SMOG = textstat.smog_index(file_content)
            CL = textstat.coleman_liau_index(file_content)
            FKG = textstat.flesch_kincaid_grade(file_content)
            DC = textstat.dale_chall_readability_score(file_content)

            # Add the dictionary of the metrics of one text file to metrics list
            metrics.append({
                "filename" : os.path.splitext(file)[0], # Split the path name into a pair root and extension; only use root
                "flesch" : FRE,
                "cl" : CL,
                "smog" : SMOG,
                "flesch-kincaid" : FKG,
                "dc" : DC
            })

        except Exception as e:
            print(f"Problem with computing the metrics of {file}: {e}")

    my_metrics = pd.DataFrame(metrics)
    
    return my_metrics

# Call the get_metrics function
my_metrics = get_metrics(files)


## Visualisation: Boxplots for each readability metric

metrics = ['flesch', 'cl', 'smog', 'flesch-kincaid', 'dc']

for metric in metrics:
    plt.figure(figsize = (6, 5))
    plt.boxplot(my_metrics[metric], patch_artist=True)
    plt.title("Boxplot of computed readability metrics")
    plt.xlabel(metric)
    plt.ylabel("Metric values")

plt.show()




"""
---------------------------------------------------------------------------------
Part 2: For each policy text file, conduct a sentiment analysis
including polarity and subjectivity

"""

## Define a function that loops through every file to conduct sentiment analysis 
# and store results in a list, which will be converted to pd dataframe

def get_sentiment(files):
    
    # Create an empty list
    sentiment_res = []

    for file in files:
    
        complete_path = base_path + file

        try:
            with open(complete_path, "r") as rd_file:
                file_content = rd_file.readlines() 
                # Reads all the lines and return them as each line a a string element in a list

            l_polarity = []
            l_subjectivity = []
        
            for line in file_content:

                # Text processing
                line = re.sub(r'[^a-zA-Z\s]', '', line) # Drop non-alphabetical characters
                line = line.lower() # Convert to lower cases - necessary for TF-IDF later

                # For every line in a txt file, conduct sentiment analysis using TextBlob
                blob = TextBlob(line)
                l_polarity.append(blob.sentiment.polarity)
                l_subjectivity.append(blob.sentiment.subjectivity)
            
            
            # Calculate the mean polarity and subjectivity scores of a txt file 
            # (rounded to 4 digits)
            mean_polarity = round(np.mean(l_polarity), 4)
            mean_subjectivity = round(np.mean(l_subjectivity), 4)

            # Classify sentiment as positive or negative based on polarity
            if mean_polarity > 0:
                sent_class = 'positive'
            elif mean_polarity < 0:
                sent_class = 'negative'
            else:
                sent_class = 'neutral'


            # Add scores and sentiment class into sentiment_res list
            sentiment_res.append({
                "Filename" : os.path.splitext(file)[0],
                "Polarity" : mean_polarity,
                "Subjectivity" : mean_subjectivity,
                "Sentiment class" : sent_class
            })

        except Exception as e:
            print(f"Problem with analysing {file}: {e}")

    df = pd.DataFrame(sentiment_res)
    
    return df


## Call the get_sentiment function
sentiment_df = get_sentiment(files)


## Visualisation of the distributions of both sentiment scores (only for the first 1000 files)

smaller_df = get_sentiment(files[0:1000])

# Generate Boxplots for Polarity and Subjectivity scores
fig, axs = plt.subplots(1, 2, figsize = (16, 8))

# Polarity
axs[0].boxplot(smaller_df["Polarity"], patch_artist = True)
axs[0].set_title("Polarity Score")
axs[0].set_ylabel("Score")

# Boxplot for Subjectivity
axs[1].boxplot(smaller_df["Subjectivity"], patch_artist = True)
axs[1].set_title("Subjectivity Score")
axs[1].set_ylabel("Score")

plt.show()




"""
---------------------------------------------------------------------------------
Part 3: Correlation analysis between readability metrics sentiment scores

"""

# Extract columns my_metrics and store as features/predictive variables
x_metrics = my_metrics.drop(columns = ['filename'])

# Extract columns of sentiment_df and store as targets/response variables
y_sentiment = sentiment_df[['Polarity', 'Subjectivity']]

# Pearson's correlation analysis between each metric and each sentiment score
corr_w_polarity = x_metrics.corrwith(y_sentiment["Polarity"])
corr_w_subjectivity = x_metrics.corrwith(y_sentiment["Subjectivity"])

heatmap_df = pd.DataFrame([corr_w_polarity,
                           corr_w_subjectivity])
heatmap_df.index = ["Polarity", "Subjectivity"]

sns.heatmap(heatmap_df, annot=True, cmap="PiYG", center = 0)
