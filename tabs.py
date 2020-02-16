
import time
import tensorflow as tf
import vis_lstm_model
import data_loader
import argparse
from os.path import isfile, join
import utils
import re
import os
import warnings
import datetime
from google_speech import Speech
import speech_recognition as sr
import base64
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
filen=""
app = dash.Dash(external_stylesheets=[dbc.themes.DARKLY])
vqaapp=html.Div([
    html.Hr(),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            
            html.A('Select Image')
        ]),
        style={
            'width': '60px',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'align':'center'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload'),
    html.H5("Ask a question associated to the image"),
    dcc.Input(id="question", type="text",style={'width':'50%','height':'60px','textAlign':'right','borderRadius':'5px','margin':'10px'}),
    html.Button("ASK",id="submit"),
    html.Div(id='output-container',
             children="enter a value and press submit",style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        })
    
])





###################################################################
###VQA TRAINER##################

vqatrainer=html.Div([
    html.Hr(),
    dcc.Upload(
        id='upload-dataset',
        children=html.Div([
            
            html.A('Select Image dataset')
        ]),
        style={
            'width': '160px',
            'height': '100px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'align':'center'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    dcc.Upload(
        id='upload-train-question',
        children=html.Div([
            
            html.A('Select training questions JSON file')
        ]),
        style={
            'width': '160px',
            'height': '160px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'align':'center'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    dcc.Upload(
        id='upload-val-question',
        children=html.Div([
            
            html.A('Select validation questions JSON file')
        ]),
        style={
            'width': '160px',
            'height': '160px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'align':'center'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Hr(),
    dcc.Upload(
        id='upload-train-annotations',
        children=html.Div([
            
            html.A('Select train annotations JSON file ')
        ]),
        style={
            'width': '160px',
            'height': '160px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'align':'center'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Hr(),
    dcc.Upload(
        id='upload-val-annotations',
        children=html.Div([
            
            html.A('Select validation annotations JSON file')
        ]),
        style={
            'width': '160px',
            'height': '160px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'align':'center'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Br(),
    html.Button("TRAIN",id='traindata')
    
])

##############################################################333
app.config['suppress_callback_exceptions'] = True
app.layout = dbc.Container(
    [
        dcc.Store(id="store"),
        html.Br(),
        html.H1("VISUAL QUESTION ANSWERING",style={'width': '1160px',
            'height': '160px',
            'lineHeight': '160px',
            'borderWidth': '2px',
            'borderStyle': 'solid',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'align':'center','color':'white'
                                                   }),
        html.Br(),html.Hr(),
        dbc.Tabs(
            [
                dbc.Tab(label="ABOUT", tab_id="about"),
                dbc.Tab(label="VQA APP", tab_id="vqa"),
                dbc.Tab(label="VQA TRAINER (in BETA)", tab_id="trainer"),
                dbc.Tab(label="INFO", tab_id="info"),
            ],
            id="tabs",
            style={'align':'center','width':'100%'}
        ),
        html.Div(id="tab-content", className="p-4"),
    ],
    style={'textAlign':'center'}
)
def parse_contents(contents, filename, date):
    global filen
    filen=filename
    return html.Div([
        html.H5("Filename:"+filen),
        #html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Hr()
        
    ])

@app.callback(
    dash.dependencies.Output('output-container', 'children'),
    [dash.dependencies.Input('submit', 'n_clicks')],
    [dash.dependencies.State('question', 'value')])

def update_output(n_clicks, value):
    vocab_data = data_loader.get_question_answer_vocab("2")
    qvocab = vocab_data['question_vocab']
    q_map = { vocab_data['question_vocab'][qw] : qw for qw in vocab_data['question_vocab']}
    print('filename::',filen)
    print(value)
    fc7_features = utils.extract_fc7_features(filen, 'Data/vgg16.tfmodel')
    model_options = {
		'num_lstm_layers' : 2,
		'rnn_size' : 512,
		'embedding_size' : 512,
		'word_emb_dropout' : 0.5,
		'image_dropout' : 0.5,
		'fc7_feature_length' : 4096,
		'lstm_steps' : vocab_data['max_question_length'] + 1,
		'q_vocab_size' : len(vocab_data['question_vocab']),
		'ans_vocab_size' : len(vocab_data['answer_vocab'])
	}
    
    question_vocab = vocab_data['question_vocab']
    word_regex = re.compile(r'\w+')
    question_ids = np.zeros((1, vocab_data['max_question_length']), dtype = 'int32')
    print('qst',value)
    question_words = re.findall(word_regex, value)
    base = vocab_data['max_question_length'] - len(question_words)
    for i in range(0, len(question_words)):
        if question_words[i] in question_vocab:
            question_ids[0][base + i] = question_vocab[ question_words[i] ]
        else:
            question_ids[0][base + i] = question_vocab['UNK']
    ans_map = { vocab_data['answer_vocab'][ans] : ans for ans in vocab_data['answer_vocab']}
    model = vis_lstm_model.Vis_lstm_model(model_options)
    input_tensors, t_prediction, t_ans_probab = model.build_generator()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, 'Data/Models/modelnew99.ckpt')
    pred, answer_probab = sess.run([t_prediction, t_ans_probab], feed_dict={
        input_tensors['fc7']:fc7_features,
        input_tensors['sentence']:question_ids,
    })
    print ("Ans:", ans_map[pred[0]])
    answer_probab_tuples = [(-answer_probab[0][idx], idx) for idx in range(len(answer_probab[0]))]
    answer_probab_tuples.sort()
    print ("Top Answers")
    for i in range(1):
        print (ans_map[ answer_probab_tuples[0][1] ])
        #ans=(ans_map[answer_probab_tuples[i][1] ])
        lang = "en"
        text="This is a "+ans_map[ answer_probab_tuples[0][1] ]
        speech = Speech(text, lang)

        sox_effects = ("speed", "0.8")
        speech.play(sox_effects)
        
    return ans_map[answer_probab_tuples[0][1]]
    
@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')],
              [State('upload-image', 'filename'),
               State('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children




##########################################








##########################################

@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("store", "data")],
)




def render_tab_content(active_tab, data):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    
    """
    test_png = 'vqadiagram.png'
    test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')
    if active_tab == "about":
        return dbc.Row([html.H3("What is VQA?", className="card-text"),
            html.P("VQA stands for Visual Question Answering. It is the artificial form of a human which is able to understand an image and question given to it and responds the answer in natural language", className="card-text"),
            html.Br(),
                        
            html.Img(src='data:image/png;base64,{}'.format(test_base64),style={'width':'75%','height':'80%'}),
            ])
    elif active_tab == "vqa":
        return dbc.Col(
            [
            
            html.H3("VQA App!", className="card-text"),
            html.H4("Give an image and question as an input", className="card-text"),
            vqaapp
           
            ]
        )
    elif active_tab == "trainer":
        return dbc.Col(
            [
            html.H3("VQA Trainer "),html.Hr(),
            html.H4("Train your own dataset with the help of our model. Find the dataset format below"),
            html.Hr(),
            vqatrainer,
            html.Br(),
            html.Iframe(src='assets/FORMAT.pdf',style={'width':'100%','height':'500px'})
            ]
        )
    elif active_tab == "info":
        return dbc.Col(
            [
            html.H3("Below is the implementation paper of the VQA task"),
            html.Iframe(src='assets/VQApaper.pdf',style={'width':'100%','height':'1200px'})
            ]
        )    
    return data




if __name__ == "__main__":
    app.run_server(debug=False)
