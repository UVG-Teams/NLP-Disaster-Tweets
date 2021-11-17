import os
import re
import nltk
import string
import folium
import sparknlp
import numpy as np
import pandas as pd
import transformers
import seaborn as sns
import tensorflow as tf
from sparknlp.base import *
import plotly.express as px
from tqdm.notebook import tqdm
from pyspark.ml import Pipeline
import matplotlib.pyplot as plt
from sparknlp.annotator import *
from nltk.corpus import stopwords
import pyspark.sql.functions as F
from keras.models import Sequential
from geopy.geocoders import Nominatim
from pyspark.sql.functions import col
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Model
from wordcloud import WordCloud, STOPWORDS
from tensorflow.keras.optimizers import Adam
from tokenizers import BertWordPieceTokenizer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Input
from geopy.extra.rate_limiter import RateLimiter
from sparknlp.pretrained import PretrainedPipeline
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import (precision_score, recall_score, f1_score, classification_report, accuracy_score)
from keras.layers import (LSTM, Embedding, BatchNormalization, Dense, TimeDistributed, Dropout, Bidirectional, Flatten, GlobalMaxPool1D)

import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output


app = dash.Dash(__name__)
server = app.server

data = pd.read_csv('train.csv')

duplicated_data = data['text'].duplicated().sum()

data = data.drop_duplicates(subset=['text'], keep='first')

data['location'].replace('', np.nan, inplace=True)
data = data.dropna(subset=['location'])

def clean_data(text):
    text = text.lower()
    text = re.sub(r'http[s]?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    text = ' '.join(text.split())
    return text

def clean_data_locations(text):
    text = str(text)
    text = re.sub(r'http[s]?:\/\/.*[\r\n]*', '', text)
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    text = " ".join(text.split())
    return text

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )

    return emoji_pattern.sub(r'', text)

data['text'] = data['text'].apply(clean_data)

data['text'] = data['text'].apply(remove_emojis)

data['location'] = data['location'].apply(clean_data_locations)

data['location'] = data['location'].apply(remove_emojis)

data['location'].replace('', np.nan, inplace=True)
data = data.dropna(subset=['location'])

data['location'] = data['location'].replace({
    "uk": "UK",
    "US": "USA",
    "England": "UK",
    "london": "London",
    "London.": "London",
    "London UK": "London",
    "United Kingdom": "UK",
    "United States": "USA",
    "London, UK": "London",
    "London England": "London",
    "London, England": "London",
    "City of London London": "London",
    "Manchester England": "Manchester",
    "Manchester UK": "Manchester",
    "Atlanta, GA": "Atlanta",
    "NEW YORK": "New York",
    "New York City": "New York",
    "New York USA": "New York",
    "NYC": "New York",
    "NY": "New York",
    "New Your": "New York",
    "new york ny": "New York",
    "New York NY": "New York",
    "New York ATL": "New York",
    "New York, NY": "New York",
    "New York NYC": "New York",
    "New York 2099": "New York",
    "New York New York": "New York",
    "New York Worldwide": "New York",
    "New York United States": "New York",
    "nyc": "New York",
    "new york": "New York",
    "NYC New York": "New York",
    "New Jersey DR": "New Jersey",
    "New Jersey USA": "New Jersey",
    "New York City NY": "New York",
    "New JerseyNew York": "New York",
    "New Jersey usually": "New Jersey",
    "New York / Worldwide": "New York",
    "San Francisco, CA": "San Francisco",
    "San Francisco CA": "San Francisco",
    "California, USA": "California",
    "California USA": "California",
    "California, United States": "California",
    "California United States": "California",
    "CA": "California",
    "Chicago, IL": "Chicago",
    "Chicago IL": "Chicago",
    "Chicago Illinois": "Chicago",
    "Los Angeles, CA": "Los Angeles",
    "Los Angeles CA": "Los Angeles",
    "L.A.": "Los Angeles",
    "Los Angeles New York": "Los Angeles",
    "Everywhere": "Worldwide",
    "Earth": "Worldwide",
    "Washington, DC": "Washington",
    "Washington DC": "Washington",
    "Washington DC NATIVE": "Washington",
    "Washington state": "Washington",
    "Washington State": "Washington",
    "Washington USA": "Washington",
    "WASHINGTONDC": "Washington",
    "WashingtonState Seattle": "Washington",
    "Western Washington": "Washington",
    "Washington DC area": "Washington",
    "Washington DC 20009": "Washington",
    "washington dc": "Washington",
    "Washington, D.C.": "Washington",
    "Florida, USA": "Florida",
    "Florida USA": "Florida",
    "Miami, FL": "Miami",
    "Miami Florida": "Miami",
    "Dallas, TX": "Dallas",
    "Dallas TX": "Dallas",
    "Austin, Texas": "Austin",
    "Austin Texas": "Austin",
    "Austin TX": "Austin",
    "Houston TX": "Houston",
    "Arlington TX": "Arlington",
    "Atlanta GA": "Atlanta",
    "Atlanta Georgia": "Atlanta",
    "San Diego CA": "San Diego",
    "Seattle WA": "Seattle",
    "Seattle Washington": "Seattle",
    "SEATTLE WA USA": "Seattle",
    "WashingtonState Seattle": "Seattle",
    "seattle wa": "Seattle",
    "Boston MA": "Boston",
    "Denver CO": "Denver",
    "Pennsylvania USA": "Pennsylvania",
    "Brooklyn NY": "Brooklyn",
    "Brooklyn New York": "Brooklyn",
    "New York Brooklyn": "Brooklyn",
    "New York Connecticut": "Connecticut",
    "Portland OR": "Portland",
    "Portland Oregon": "Portland",
    "Orlando FL": "Orlando",
    "Tampa FL": "Tampa",
    "Texas USA": "Texas",
    "indiana": "Indiana",
    "Indiana USA": "Indiana",
    "Indianapolis IN": "Indiana",
    "canada": "Canada",
    "Berlin Germany": "Berlin",
    "Las Vegas Nevada": "Las Vegas",
    "Massachusetts USA": "Massachusetts",
    "Georgia USA": "Georgia",
    "ARGENTINA": "Argentina",
    "china": "China",
    "india": "India",
    "INDIA": "India",
    "Bangalore INDIA": "Bangalore",
    "Bangalore India": "Bangalore",
    "Bangalore City India": "Bangalore",
    "italy": "Italy",
    "anzioitaly": "Italy",
    "ITALY": "Italy",
    "Rome Italy": "Rome",
    "CasertaRoma Italy": "Caserta",
    "Manhattan NY": "Manhattan",
    "Mumbai India": "Mumbai",
    "Mumbai india": "Mumbai",
    "Stockholm Sweden": "Stockholm",
    "Virginia USA": "Virginia",
    "Auckland New Zealand": "Auckland",
    "Buenos Aires Argentina": "Buenos Aires",
    "Lagos Nigeria": "Lagos",
    "nigeria": "Nigeria",
    "NIGERIA": "Nigeria",
    "Nigeria Global": "Nigeria",
    "Nigeria WORLDWIDE": "Nigeria",
    "Charlotte NC": "Charlotte",
    "Charlotte County Florida": "Charlotte",
    "Nashville TN": "Nashville",
    "Sacramento CA": "Sacramento",
    "Melbourne Australia": "Melbourne",
    "Memphis TN": "Memphis",
    "Calgary Alberta": "Calgary",
    "Calgary AB Canada": "Calgary",
    "Calgary Alberta Canada": "Calgary",
    "Calgary Canada": "Calgary",
    "CalgaryAB Canada": "Calgary",
    "caNADA": "Canada",
    "Canada BC": "Canada",
    "Canada Eh": "Canada",
    "Toronto Canada": "Toronto",
    "Toronto ON Canada": "Toronto",
    "TorontoCitizen of Canada US": "Toronto",
    "Vancouver BC Canda": "Vancouver",
    "Vancouver Canada": "Vancouver",
    "Victoria BC Canada": "Victoria",
    "Victoria Canada": "Victoria",
    "Oklahoma City OK": "Oklahoma",
    "Ontario Canada": "Ontario",
    "Newcastle": "Newcastle Upon Tyne",
    "Newcastle England": "Newcastle Upon Tyne",
    "Newcastle OK": "Newcastle Upon Tyne",
    "Newcastle upon Tyne": "Newcastle Upon Tyne",
    "Newcastle Upon Tyne England": "Newcastle Upon Tyne",
    "NewcastleuponTyne UK": "Newcastle Upon Tyne",
    "The Netherlands": "Netherlands",
    "New Delhi Delhi": "New Delhi",
    "New Delhi India": "New Delhi",
    "New DelhiIndia": "New Delhi",
    "Adelaide Australia": "Adelaide",
    "Adelaide South Australia": "Adelaide",
    "Brisbane Australia": "Brisbane",
    "brisbane australia": "Brisbane",
    "Gold Coast Australia": "Gold Coast",
    "Gold Coast Qld Australia": "Gold Coast",
    "Sydney Australia": "Sydney",
    "sydney australia": "Sydney",
})


keywords = data['keyword'].value_counts().nlargest()
fig = px.bar(data, x=keywords.tolist(), y=keywords.index)


# BERT MODEL

# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# def bert_encode(data, maximum_length):
#     input_ids = []
#     attention_masks = []

#     for text in data:
#         encoded = tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=maximum_length,
#             pad_to_max_length=True,
#             return_attention_mask=True,
#         )
#         input_ids.append(encoded['input_ids'])
#         attention_masks.append(encoded['attention_mask'])

#     return np.array(input_ids), np.array(attention_masks)

# text = data['text']
# target = data['target']

# train_input_ids, train_attention_masks = bert_encode(text, 60)

# def create_model(bert_model):

#     input_ids = tf.keras.Input(shape=(60,),dtype='int32')
#     attention_masks = tf.keras.Input(shape=(60,),dtype='int32')

#     output = bert_model([input_ids,attention_masks])
#     output = output[1]
#     output = tf.keras.layers.Dense(32,activation='relu')(output)
#     output = tf.keras.layers.Dropout(0.2)(output)
#     output = tf.keras.layers.Dense(1,activation='sigmoid')(output)

#     model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)
#     model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# bert_model = transformers.TFBertModel.from_pretrained('bert-base-uncased')
# model = create_model(bert_model)
# model.summary()
# history = model.fit(
#     [train_input_ids, train_attention_masks],
#     target,
#     validation_split=0.2,
#     epochs=1,
#     batch_size=10
# )

# def plot_learning_curves(history, arr):
#     fig, ax = plt.subplots(1, 2, figsize=(20, 5))
#     for idx in range(2):
#         ax[idx].plot(history.history[arr[idx][0]])
#         ax[idx].plot(history.history[arr[idx][1]])
#         ax[idx].legend([arr[idx][0], arr[idx][1]],fontsize=18)
#         ax[idx].set_xlabel('A ',fontsize=16)
#         ax[idx].set_ylabel('B',fontsize=16)
#         ax[idx].set_title(arr[idx][0] + ' X ' + arr[idx][1],fontsize=16)

# plot_learning_curves(history, [['loss', 'val_loss'],['accuracy', 'val_accuracy']])





# LSTM MODEL


# SPARK NLP MODEL


# spark = sparknlp.start()

# df_train = spark.read.option("header", True).csv("clean_data.csv")

# df_train = df_train.na.drop(how="any")
# df_train.groupby("target").count().orderBy(col("count")).show()

# document = DocumentAssembler().setInputCol("text").setOutputCol("document")

# use = UniversalSentenceEncoder.pretrained().setInputCols(["document"]).setOutputCol("sentence_embeddings")

# classsifierdl = ClassifierDLApproach().setInputCols(["sentence_embeddings"]).setOutputCol("class").setLabelColumn("target").setMaxEpochs(10).setEnableOutputLogs(True).setLr(0.004)

# nlpPipeline = Pipeline(
#     stages = [
#         document,
#         use,
#         classsifierdl
#     ]
# )

# (train_set, test_set)= df_train.randomSplit([0.8, 0.2], seed=100)

# use_model = nlpPipeline.fit(train_set)

# # !cd ~/annotator_logs && ls -l

# # !cat ~/annotator_logs/ClassifierDLApproach_4c93d2227f55.log

# prediction = use_model.transform(train_set)
# prediction.select("target", "text", "class.result").show(5, truncate=False)

# df = use_model.transform(train_set).select("target", "document", "class.result").toPandas()
# df["result"]= df["result"].apply(lambda x: x[0])
# print(classification_report(df["target"], df["result"]))









app.layout = html.Div(
    [
        html.H1('Disaster Tweets'),
        html.I("Hola compadre, pone un tweet y te digo si es desastre o no"),
        html.Br(),
        dcc.Input(
            id="inp_tweet",
            type="text",
            placeholder="Enter a tweet...",
            debounce=True,
            style={
                'width': '98.2%',
                'margin-bottom': '15px',
                'padding': '10px',
                'border-radius': '10px',
                'border-style': 'solid',
            }
        ),
        html.Div(id="output"),
        html.Div([
            html.Div([
                html.H4("Bert")
            ],
                id="output_bert",
                style={
                    'background': 'green',
                    'width': '30%',
                    'font-size': '20px',
                    'text-align': 'center',
                    'border-radius': '20px',
                    'padding-top': '15px',
                    'padding-bottom': '15px',
                }
            ),
            html.Div(
                id="output_lstm",
                style={
                    'background': 'green',
                    'width': '30%',
                    'font-size': '20px',
                    'text-align': 'center',
                    'border-radius': '20px',
                    'padding-top': '15px',
                    'padding-bottom': '15px',
                }
            ),
            html.Div(
                id="output_sparknlp",
                style={
                    'background': 'green',
                    'width': '30%',
                    'font-size': '20px',
                    'text-align': 'center',
                    'border-radius': '20px',
                    'padding-top': '15px',
                    'padding-bottom': '15px',
                }
            ),
        ], style={
            'display':'flex',
            'justify-content':'space-between'
        }),
        # dcc.Graph(
        #     id='example-graph',
        #     figure=fig
        # )
    ]
)


@app.callback(
    Output("output", "children"),
    Output("output_bert", "children"),
    Output("output_lstm", "children"),
    Output("output_sparknlp", "children"),
    Input("inp_tweet", "value"),
)
def update_output(input1):
    print(input1)
    # if input1 != None:
    #     columns = ["id", "text"]
    #     data = [("0", input1)]
    #     rdd = spark.sparkContext.parallelize(data)
    #     l = spark.createDataFrame(rdd).toDF(*columns)
    #     # l.show()
    #     prediction = use_model.transform(l)
    #     # prediction.select("id", "text", "class.result").show(truncate=False)
    #     pred = prediction.select("class.result").collect()
    #     res_sparkNLP = re.sub(r'[^0-9 ]+', '', str(pred[0]))
    #     # print(res_sparkNLP)
    #     return (input1, 'Bert: \n {}'.format(input1), 'LSTM: \n {}'.format(input1), 'SparkNLP: \n {}'.format(res_sparkNLP))
    return (input1, 'Bert: \n {}'.format(input1), 'LSTM: \n {}'.format(input1), 'SparkNLP: \n {}'.format(input1))



if __name__ == '__main__':
    app.run_server(debug=True, host='127.0.0.1')
