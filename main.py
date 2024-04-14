import re
import string
from heapq import nlargest
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI
from pandasai import SmartDataframe

import dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import tabulate
from dash import callback_context, dcc, html
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.preprocessing import StandardScaler


import os
import random
import re
import string
from heapq import nlargest

import dash
import duckdb
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tabulate
from GeneratedData import DataGenerator
from dash.dependencies import Input, Output, State
from faker import Faker
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import \
    create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI, OpenAI
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
# Download stopwords and punkt if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')
fake = Faker()

class VisualizationDashboard:
    def __init__(self):
        self.data_generator = DataGenerator()
        self.data = self.data_generator.generate_fake_data()        
        self.dropdown_options = [
            {"label": col, "value": col} for col in ["Monthly Revenue", "Opportunity Amount", "Support Tickets Open",
                                                     "Support Tickets Closed", "Lead Score", "Age", "Contract Type",
                                                     "Gender", "Lead Status"]
        ]


    def create_scatter_layout(self):
        return html.Div([
            html.H1('Scatter Plot'),
            dcc.Dropdown(
                id='scatter-dropdown-x',
                options=self.dropdown_options,
                value='Monthly Revenue',
                style={'color': 'black'}
            ),
            dcc.Dropdown(
                id='scatter-dropdown-y',
                options=self.dropdown_options,
                value='Opportunity Amount',
                style={'color': 'black'}
            ),
            dcc.Graph(
                id='scatter-plot',
                style={'width': '100%', 'height': 'calc(100vh - 150px)'}  # Adjusted height for full screen
            )
        ])

    def create_pie_chart_layout(self):
        return html.Div([
            html.H1("World GDP Distribution by Category"),
            dcc.Dropdown(
                id='pie-dropdown-category',
                options=[
                    {'label': 'Age Group', 'value': 'Age'},
                    {'label': 'Lead Status', 'value': 'Lead Status'},
                    {'label': 'Contract Type', 'value': 'Contract Type'},
                    {'label': 'Con3tinent', 'value': 'Continent'},
                    {'label': 'Gender', 'value': 'Gender'},
                ],
                value='Continent',
                clearable=False
            ),
            dcc.Dropdown(
                id='pie-dropdown-year',
                options=[{'label': year, 'value': year} for year in range(1980, 2021, 5)],
                value=range(1980, 2021, 5)[-1],
                clearable=False
            ),
            dcc.Graph(id='gdp-pie-chart')
        ])

    def create_time_series_layout(self):
        return html.Div([
            html.H1('Time Series Plot'),
            dcc.Dropdown(
                id='time-dropdown-x',
                options=[{'label': col, 'value': col} for col in ['Last Email Sent Date','Last Interaction Date','Last Phone Call Date','Last Meeting Date']],
                value='Last Email Sent Date',
                style={'color':'black'}
            ),
            dcc.Dropdown(
                id='time-dropdown-y',
                options=[{'label': col, 'value': col} for col in ['Monthly Revenue','Opportunity Amount','Probability of Close']],
                value='Opportunity Amount',
                style={'color':'black'}
            ),
            dcc.Graph(id='line-chart',style={'width': '100%', 'height': 'calc(100vh - 150px)'})
        ])



    def create_bar_chart_layout(self):
        return html.Div([
            html.H1("Bar Plot Distribution", style={'marginBottom': '20px'}),
            html.Div([
                html.Label("Select Column:", style={'marginRight': '10px'}),
                dcc.Dropdown(
                    id='bar-dropdown-column',
                    options=[{'label': col, 'value': col} for col in self.data.columns],
                    value='Gender'
                ),
                dcc.Graph(id='bar-chart'),
                html.Div(id='stats', style={'marginTop': '20px'})
            ], style={'width': '95%', 'margin': 'auto'}),
            html.Div([
                dcc.Slider(
                    id='bar-slider-points',
                    min=5,
                    max=50,
                    step=10,
                    value=50,
                    marks={i: str(i) for i in range(10, 101, 10)},
                    tooltip={'placement': 'top'}
                )
            ], style={'width': '95%', 'margin': 'auto', 'marginTop': '50px'})
        ], style={'textAlign': 'center', 'fontFamily': 'Arial, sans-serif'})

    def create_choropleth_layout(self):
        return html.Div([
            html.H1("Country Data"),
            html.Div([
                dcc.Dropdown(
                    id='choropleth-dropdown-column',
                    options=[
                        {'label': 'Population', 'value': 'Population'},
                        {'label': 'Area (sq km)', 'value': 'Area (sq km)'},
                        {'label': 'GDP (USD)', 'value': 'GDP (USD)'}
                    ],
                    value=['Population'],  # Default value
                    multi=True,
                    
                    )
            ]),
            html.Div(id='choropleth-container')
        ])

    def create_histogram_layout(self):
        return html.Div([
            html.H1("Fascinating Histogram"),
            html.Div([
                html.Label("Select Data Column:"),
                dcc.Dropdown(
                    id='hist-dropdown-column',
                    options=[{'label': col, 'value': col} for col in self.data.keys()],
                    value='Lead Score'
                ),
                dcc.Graph(id='histogram',
                          config={'displayModeBar': False}),
                html.Div(id='explanation', style={'padding': 10, 'fontSize': 18}),
                html.Label("Number of Bins:"),
                dcc.Slider(
                    id='hist-slider-bins',
                    min=5,
                    max=50,
                    step=1,
                    value=20,
                    marks={i: str(i) for i in range(5, 51, 5)}
                ),
            ], style={'width': '95%', 'margin': 'auto','marginTop':'50px'}),
       
        ], style={'textAlign': 'center'})
    





app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
dashboard = VisualizationDashboard()

# Function for text summarization
def text_summarization(text, num_sentences=3):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    stop_words = set(stopwords.words('english'))
    word_freq = {}
    for word in words:
        if word not in stop_words and word.isalnum():
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1

    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                if len(sentence.split(' ')) < 30:
                    if sentence not in sentence_scores:
                        sentence_scores[sentence] = word_freq[word]
                    else:
                        sentence_scores[sentence] += word_freq[word]

    summarized_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summarized_sentences)
    return summary





# Function to analyze sentiment using TextBlob
def analyze_sentiment(comment):
    analysis = TextBlob(comment)
    return analysis.sentiment.polarity

# Function to create layout
def create_layout(df):
    layout = html.Div([
        html.H1("Sentiment Analysis Dashboard"),
        html.Div([
            dcc.DatePickerRange(
                id='date-range-picker',
                min_date_allowed=df['Timestamp'].min(),
                max_date_allowed=df['Timestamp'].max(),
                initial_visible_month=df['Timestamp'].min(),
                start_date=df['Timestamp'].min(),
                end_date=df['Timestamp'].max(),
                display_format='YYYY-MM-DD'
            ),
        ], style={'margin-bottom': '20px', 'margin-top': '50px'}),  # Adjust margin-top and margin-bottom here
        html.Div([
            dcc.Graph(id='sentiment-graph'),
            html.Div(id='description')
        ], style={'margin-top': '20px'})  # Adjust margin-top here
    ])
    return layout

# Generate fake comments
df=dashboard.data

# Assuming df2 already exists
df2 = df[['Timestamp', 'comment']]

# Analyze sentiment and create DataFrame
data = []
for index, row in df2.iterrows():
    sentiment = analyze_sentiment(row['comment'])
    data.append((row['Timestamp'], sentiment, row['comment']))

df = pd.DataFrame(data, columns=['Timestamp', 'Sentiment', 'Comment'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
# Sort DataFrame by Timestamp
df.sort_values(by='Timestamp', inplace=True)



scatter_button=dbc.Button("Scatter Plot", id='btn-scatter', outline=True, color="primary",className="me-1", style={"margin-left": "25px", "margin-right": "10px", "margin-top": "5px", "margin-bottom": "5px", "width": "80px","fontSize": "13px", "padding": "2px 1px"})
pie_button=dbc.Button("Pie Plot", id='btn-pie', outline=True, color="primary", className="me-1", style={"margin-left": "10px", "margin-right": "10px", "margin-top": "5px", "margin-bottom": "5px", "width": "80px","fontSize": "13px", "padding": "2px 1px"})
time_button=dbc.Button("Time Series", id='btn-time', outline=True, color="primary", className="me-1", style={"margin-left": "10px", "margin-right": "10px", "margin-top": "5px", "margin-bottom": "5px", "width": "80px","fontSize": "13px", "padding": "2px 1px"})
bar_button=dbc.Button("Bar Plot", id='btn-bar', outline=True, color="primary", className="me-1", style={"margin-left": "25px", "margin-right": "10px", "margin-top": "5px", "margin-bottom": "15px", "width": "80px","fontSize": "13px", "padding": "2px 1px"})
choro_button=dbc.Button("Geo Plot", id='btn-choro', outline=True, color="primary", className="me-1", style={"margin-left": "10px", "margin-right": "10px", "margin-top": "5px", "margin-bottom": "15px", "width": "80px","fontSize": "13px", "padding": "2px 1px"})
histo_button=dbc.Button("Histogram", id='btn-histo', outline=True, color="primary", className="me-1", style={"margin-left": "10px", "margin-right": "10px", "margin-top": "5px", "margin-bottom": "15px", "width": "80px","fontSize": "13px", "padding": "2px 1px"})




app.layout = html.Div([
    html.Div(
        html.H1(id='page-title', children="Master Visualization", style={'text-align': 'center'}),
        style={'margin-top': '20px'}
    ),
    dcc.Location(id='url', refresh=True),
    html.Div(id='page-content', style={'text-align': 'center'})
])

# Define layout
def smartdata_layout():
    return html.Div(children=[
    dcc.Input(id='user-input', type='text', placeholder='Enter your query...', style={'width': '95%', 'padding': '15px', 'fontSize': '18px', 'marginBottom': '20px', 'borderRadius': '8px', 'border': '1px solid #ccc', 'outline': 'none','margin-top': '30px'}),
    html.Button('Analyse', id='analyse-button', n_clicks=0, style={'backgroundColor': '#4CAF50', 'border': 'True', 'color': 'white', 'padding': '10px 20px', 'textAlign': 'center', 'textDecoration': 'none', 'display': 'inline-block', 'fontSize': '17px', 'marginBottom': '20px', 'cursor': 'pointer', 'borderRadius': '8px','margin-bottom': '10px','marginRight':'17px'}),
    html.Button('Refresh', id='refresh-button', n_clicks=0, style={'backgroundColor': '#008CBA', 'border': 'True', 'color': 'white', 'padding': '10px 20px', 'textAlign': 'center', 'textDecoration': 'none', 'display': 'inline-block', 'fontSize': '17px', 'marginBottom': '20px', 'marginLeft': '17px', 'cursor': 'pointer', 'borderRadius': '8px'}),
    html.Div(id='output-container', style={'width': '95%', 'padding': '15px', 'fontSize': '16px', 'marginBottom': '20px', 'borderRadius': '8px', 'border': '1px solid #ccc', 'outline': 'none', 'height': '200px', 'overflowY': 'scroll', 'margin': 'auto'})
        ])

# Callback to render different plots and change H1 title based on URL path
@app.callback([Output('page-content', 'children'),
               Output('page-title', 'children')],
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/customer_seg':
        series_visualization = SeriesVisualization()
        layout = series_visualization.generate_layout()
        return layout, "Customer Segmentation"
    elif pathname == '/timeseries':
        return update_graph2(), "Time Series Analysis"
    elif pathname == '/talk_to_data':
        return smartdata_layout(), "SmartDataFrame Chat"  # Fixed pathname
    elif pathname == '/other_tools':
        return update_graph3(), "Text Analyzer"
    else:
        return generate_default_plot(), "Master Visualization"

user_queries = []

llm = ChatGroq(
    temperature=0,
    model_name="mixtral-8x7b-32768",
    api_key=os.environ["GROQ_API_KEY"]  # Replace with your API key
)
df_smart = SmartDataframe(dashboard.data, config={"llm": llm})

# Define callback to handle user input and display output
@app.callback(
    Output('output-container', 'children'),
    [Input('analyse-button', 'n_clicks'),
     Input('refresh-button', 'n_clicks')],
    [State('user-input', 'value')]
)
def update_output(analyse_clicks, refresh_clicks, user_input):
    global user_queries
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None

    if triggered_id == 'analyse-button' and analyse_clicks > 0 and user_input:
        output = df_smart.chat(user_input)
        if 'plot' in user_input.lower():
            try:
                # Assuming df.plot() returns a plot object
                fig = df_frame.plot()  
                user_queries.append({'input': user_input, 'output': fig})
            except Exception as e:
                user_queries.append({'input': user_input, 'output': str(e)})
        else:
            user_queries.append({'input': user_input, 'output': output})

    elif triggered_id == 'refresh-button' and refresh_clicks > 0:
        user_queries = []

    return [html.Div([
        html.P(query['output']) if isinstance(query['output'], str) else query['output']
    ]) for query in user_queries]



def update_graph3():
    return html.Div([
        dmc.Stack(children=[dmc.Textarea(id='input-text', label="Enter text to analyze:", style={"width": 300, 'margin': '0 auto','color':'blue'}, error="Message can't be empty!",className='blue-text')]),

        html.Button('Analyze', id='analyze-button', n_clicks=0, style={'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'padding': '5px 10px', 'border-radius': '5px','margin-top': '5px','marginRight': '7px'}),
        html.Button('Summary', id='summary-button', n_clicks=0, style={'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'padding': '5px 10px', 'border-radius': '5px', 'margin-top': '5px','marginLeft': '7px'}),
        html.Button('Translate', id='translate-button', n_clicks=0, style={'backgroundColor': '#007bff', 'color': 'white', 'border': 'none', 'padding': '5px 10px', 'border-radius': '5px', 'margin-top': '5px','marginLeft': '13px'}),
        html.Div(id='output-div', style={'text-align': 'center'}),
        dcc.Graph(id='frequency-plot', style={'margin': '0 auto'})
    ], style={'text-align': 'center'})




# Define callback to update output and frequency plot
@app.callback(
    [Output('output-div', 'children'),
     Output('frequency-plot', 'figure')],
    [Input('analyze-button', 'n_clicks'),
     Input('summary-button', 'n_clicks'),
     Input('translate-button', 'n_clicks')],
    [State('input-text', 'value')]
)

def update_output(analyze_clicks, summary_clicks, translate_clicks, text):
    ctx = callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if not text:
        return "No text to analyze.", {}

    if button_id == 'analyze-button':
        # Tokenize and filter out stop words, punctuation, and non-alphabetic characters
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if (word not in stop_words and word.isalpha())]

        freq_dist = nltk.FreqDist(filtered_tokens)
        insight_result = freq_dist.most_common(10)  # Default value set to 10

        # Sentiment analysis
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity

        # Determine sentiment label and color
        sentiment_label = "Happy 😊" if sentiment_score >= 0 else "Sad 😢"
        sentiment_color = '#28a745' if sentiment_score >= 0 else '#dc3545'


        dfr = pd.DataFrame({'Word': [word[0] for word in insight_result], 'Frequency': [word[1] for word in insight_result]})
        fig = px.bar(dfr, x='Word', y='Frequency', title='Top 10 Most Common Words')
        fig.update_layout(plot_bgcolor='#f9f9f9', paper_bgcolor='#f9f9f9', font_color='#333333', width=305, margin=dict(b=1, l=1, r=1))
        return [
            html.Div(f"Sentiment: {sentiment_label}", style={'color': sentiment_color, 'margin-bottom': '10px'}),
            html.Div([html.Span(f"{word[0]}: {word[1]}", style={'margin-right': '10px'}) for word in insight_result], style={'margin-bottom': '10px'}),
            html.Div(f"Total Words: {len(filtered_tokens)}", style={'margin-bottom': '10px'})
        ], fig

    elif button_id == 'summary-button':
        summary = text_summarization(text)
        # Tokenize and filter out stop words, punctuation, and non-alphabetic characters
        tokens = word_tokenize(summary.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if (word not in stop_words and word.isalpha())]

        freq_dist = nltk.FreqDist(filtered_tokens)
        insight_result = freq_dist.most_common(10)  # Default value set to 10

        # Generate frequency 


        dfr2 = pd.DataFrame({'Word': [word[0] for word in insight_result], 'Frequency': [word[1] for word in insight_result]})
        fig = px.bar(dfr2, x='Word', y='Frequency', title='Top 10 Most Common Words')
        fig.update_layout(plot_bgcolor='#f9f9f9', paper_bgcolor='#f9f9f9', font_color='#333333', width=305, margin=dict(b=1, l=1, r=1))
        return [
            html.Div([
                html.H3("Summary:", style={'color': 'blue'}),
                html.P(summary)
            ]),
            fig
        ]

    elif button_id == 'translate-button':
        translated_text = GoogleTranslator(source='auto', target='en').translate(text)
        # Tokenize and filter out stop words, punctuation, and non-alphabetic characters
        tokens = word_tokenize(translated_text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if (word not in stop_words and word.isalpha())]

        freq_dist = nltk.FreqDist(filtered_tokens)
        insight_result = freq_dist.most_common(10)  # Default value set to 10

        # Generate fr

        dfr3 = pd.DataFrame({'Word': [word[0] for word in insight_result], 'Frequency': [word[1] for word in insight_result]})
        fig = px.bar(dfr3, x='Word', y='Frequency', title='Top 10 Most Common Words')
        fig.update_layout(plot_bgcolor='#f9f9f9', paper_bgcolor='#f9f9f9', font_color='#333333', width=305, margin=dict(b=1, l=1, r=1))
        return [
            html.Div([
                html.H3("Translated Text:", style={'color': 'blue'}),
                html.P(translated_text)
            ]),
            fig
        ]


    
    
# Generate default plot layout
def generate_default_plot():
    return html.Div([
        scatter_button,
        pie_button,
        time_button,
        bar_button,
        choro_button,
        histo_button,
        html.Div(id='plot')  # Changed from dcc.Graph to a simple div
    ])


    

# Callback to update the plot based on button click
@app.callback(
    Output('plot', 'children'),  # Changed to 'children' instead of 'figure' as 'plot' is a Div
    [Input('btn-scatter', 'n_clicks'),
     Input('btn-pie', 'n_clicks'),
     Input('btn-time', 'n_clicks'),
     Input('btn-bar', 'n_clicks'),
     Input('btn-choro', 'n_clicks'),
     Input('btn-histo','n_clicks')]
)
def update_plot(btn_scatter, btn_pie, btn_time, btn_bar, btn_choro, btn_histo):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'btn-scatter':
        return dashboard.create_scatter_layout()
    elif button_id == 'btn-pie':
        return dashboard.create_pie_chart_layout()
    elif button_id == 'btn-time':
        return dashboard.create_time_series_layout()
    elif button_id == 'btn-bar':
        return dashboard.create_bar_chart_layout()
    elif button_id == 'btn-choro':
        return dashboard.create_choropleth_layout()
    elif button_id == 'btn-histo':
        return dashboard.create_histogram_layout()
    else:
        return dashboard.create_scatter_layout()  # Added return statement for the default case



    




class SeriesVisualization:
    
    def customer_seg(self):
        self.data_generator = DataGenerator()
        df = self.data_generator.generate_fake_data()
        X = df[['Monthly Revenue', 'Opportunity Amount', 'Support Tickets Open',
                'Support Tickets Closed', 'Lead Score', 'Age', 'Size',
                'Population', 'Area (sq km)', 'GDP (USD)', 'Probability of Close']]

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Fit KMeans clustering algorithm
        k = 3  # Number of clusters
        kmeans = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = kmeans.fit_predict(X_scaled)

        # Map clusters to meaningful categories
        cluster_mapping = {0: "Active", 1: "Inactive", 2: "Lead"}

        df['Cluster'] = df['Cluster'].map(cluster_mapping)

        return df
    
    def generate_layout(self):
        scatter_plot = self.customer_seg()
        fig_3d = px.scatter_3d(scatter_plot, x='Monthly Revenue', y='Opportunity Amount', z='Support Tickets Open',
                            color='Cluster', symbol='Cluster', opacity=0.7,
                            hover_data=['Age', 'Size', 'Population', 'Area (sq km)', 'GDP (USD)', 'Probability of Close'])
        fig_3d.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        fig_3d.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                             margin=dict(l=2, r=5, t=90, b=40),
                             xaxis=dict(showgrid=False, zeroline=False),
                             yaxis=dict(showgrid=False, zeroline=False))

        fig_2d = px.scatter(scatter_plot, x='Monthly Revenue', y='Opportunity Amount', color='Cluster',
                                 hover_data=['Age', 'Size', 'Population', 'Area (sq km)', 'GDP (USD)', 'Probability of Close'])
        fig_2d.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                             margin=dict(l=2, r=5, t=90, b=40),
                             xaxis=dict(showgrid=False, zeroline=False),
                             yaxis=dict(showgrid=False, zeroline=False))

        tabs = dcc.Tabs(id='tabs', value='scatter-plot-3d', children=[
            dcc.Tab(label='3D Scatter Plot', value='scatter-plot-3d',children=[
                dcc.Graph(id='scatter-plot-3d', figure=fig_3d)
            ]),
            dcc.Tab(label='2D Scatter Plot', value='scatter-plot-2d', children=[
                dcc.Graph(id='scatter-plot-2d', figure=fig_2d)
            ])

            
        ], vertical=False)


        return html.Div([
            html.Div(tabs, style={'margin-bottom': '20px'}),
            html.Div(id='statistics', style={'margin-top': '20px', 'text-align': 'center', 'font-size': '16px'})
        ])

    @staticmethod
    def calculate_statistics(df):
        basic_stats = {
            'Clusters': len(df['Cluster'].unique()),
            'Total Customers': len(df),
            'Average Monthly Revenue': round(df['Monthly Revenue'].mean(), 2),
            'Average Opportunity Amount': round(df['Opportunity Amount'].mean(), 2),
        }
        return basic_stats


@app.callback(
    [Output('scatter-plot-3d', 'figure'),
     Output('statistics', 'children')],
    [Input('tabs', 'value')]
)
def update_chart(tab):
    series_visualization = SeriesVisualization()  # Instantiate SeriesVisualization here
    
    if tab == 'scatter-plot-3d':
        chart_type = 'scatter-plot-3d'
    else:
        chart_type = 'scatter-plot-2d'

    df = series_visualization.customer_seg()
    
    if chart_type == 'scatter-plot-3d':
        fig = px.scatter_3d(df, x='Monthly Revenue', y='Opportunity Amount', z='Support Tickets Open',
                            color='Cluster', symbol='Cluster', opacity=0.7,
                            hover_data=['Age', 'Size', 'Population', 'Area (sq km)', 'GDP (USD)', 'Probability of Close'])
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0)),
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                             margin=dict(l=5, r=5, t=90, b=5),
                             xaxis=dict(showgrid=False, zeroline=False),
                             yaxis=dict(showgrid=False, zeroline=False))
        description = """
            Key points about the visualization:
            - 3D Scatter Plot: Visualizes the relationship between each data points.
            - Each data point represents a customer, color-coded by their cluster.
        """
    else:
        fig = px.scatter(df, x='Monthly Revenue', y='Opportunity Amount', color='Cluster',
                         hover_data=['Age', 'Size', 'Population', 'Area (sq km)', 'GDP (USD)', 'Probability of Close'])
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                             margin=dict(l=5, r=5, t=90, b=5),
                             xaxis=dict(showgrid=False, zeroline=False),
                             yaxis=dict(showgrid=False, zeroline=False))
        description = """
            Key points about the visualization:
            - 2D Scatter Plot: Visualizes the relationship between each data points.
            - Each data point represents a customer, color-coded by their cluster.
        """

    statistics = series_visualization.calculate_statistics(df)
    return fig, [
        html.P(description, style={'white-space': 'pre-line'}),
        html.Table([
            html.Tr([html.Th(col) for col in statistics.keys()]),
            html.Tr([html.Td(value) for value in statistics.values()])
        ])
    ]







    





# Define callback to update graph and description
@app.callback(
    [Output('sentiment-graph', 'figure'),
     Output('description', 'children')],
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_graph2():
    filtered_df = df  # You can filter the DataFrame here as needed
    
    # Create plot
    trace = go.Scatter(
        x=filtered_df['Timestamp'],
        y=filtered_df['Sentiment'],
        mode='lines+markers',
        name='Sentiment',
        marker=dict(color=['blue' if x > 0 else ('red' if x < 0 else 'green') for x in filtered_df['Sentiment']])
    )

    layout = go.Layout(
        title='Sentiment Analysis Over Time',
        xaxis=dict(title='Timestamp', rangeslider_visible=True),
        yaxis=dict(),
        height=350,
        margin=dict(l=25, r=1, t=105, b=70),
        hovermode='x',  # Add margin for better display
        xaxis_rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        showlegend=False,
    )

    # Additional details
    num_comments = len(filtered_df)
    positive_sentiment_count = (filtered_df['Sentiment'] > 0).sum()
    negative_sentiment_count = (filtered_df['Sentiment'] < 0).sum()
    total_sentiments = positive_sentiment_count + negative_sentiment_count
    positive_sentiment_percentage = (positive_sentiment_count / total_sentiments) * 100
    negative_sentiment_percentage = (negative_sentiment_count / total_sentiments) * 100
    sentiment_distribution = [("Positive", positive_sentiment_percentage), ("Negative", negative_sentiment_percentage)]
    positive_sentiments = filtered_df[filtered_df['Sentiment'] > 0]['Sentiment'].mean()
    negative_sentiments = filtered_df[filtered_df['Sentiment'] < 0]['Sentiment'].mean()
    average_sentiment = (positive_sentiments + negative_sentiments) / 2

    description = [
        html.P(f"Total number of comments: {num_comments}"),
        html.P(f"Average sentiment score: {average_sentiment:.2f}"),
        html.P("Sentiment distribution:"),
        html.Ul([html.Li(f"{sentiment}: {percentage:.2f}%") for sentiment, percentage in sentiment_distribution])
    ]

    return html.Div([
        dcc.Graph(figure={'data': [trace], 'layout': layout}),
        html.Div(description)
    ])




# Scatter Plot Callback
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('scatter-dropdown-x', 'value'),
     Input('scatter-dropdown-y', 'value')]
)
def update_scatter_plot(x_column, y_column):
    fig = px.scatter(dashboard.data, x=x_column, y=y_column, size="Size", color="Continent",
                     log_x=True, size_max=21, title="Scatter Plot")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                      margin=dict(l=2, r=5, t=110, b=40),
                                      xaxis=dict(showgrid=True, zeroline=True),
                                      yaxis=dict(showgrid=True, zeroline=True) )
        
    fig.update_traces(marker=dict(sizemin=9))  # Set minimum size for markers
    return fig

# Pie Chart Callback
@app.callback(
    Output('gdp-pie-chart', 'figure'),
    [Input('pie-dropdown-category', 'value'),
     Input('pie-dropdown-year', 'value')]
)
def update_pie_chart(selected_category, selected_year):
    df_grouped = dashboard.data.groupby(selected_category)['GDP (USD)'].sum()

    # Create a pie chart
    fig = go.Figure(data=[go.Pie(labels=df_grouped.index, values=df_grouped.values, hole=0.4)])

    # Update layout
    title = f"World GDP Distribution by {selected_category.capitalize()}"
    fig.update_layout(title=title,
                      margin=dict(t=10, b=10, l=10, r=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      uniformtext_minsize=12, uniformtext_mode='hide')
    

    return fig

# Time Series Callback
@app.callback(
    Output('line-chart', 'figure'),
    [Input('time-dropdown-x', 'value'),
     Input('time-dropdown-y', 'value')]
)
def update_graph(x_value, y_value):
    fig = px.line(dashboard.data, x=x_value, y=y_value, title='Time Series Plot')
    fig.update_xaxes(rangeslider_visible=True)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=1, r=5, t=90, b=40),
            xaxis=dict(showgrid=True, zeroline=True),
            yaxis=dict(showgrid=True, zeroline=True)
        )
    return fig




# Bar Chart Callback
@app.callback(
    Output('bar-chart', 'figure'),
    [Input('bar-dropdown-column', 'value'),
     Input('bar-slider-points', 'value')]
)

def update_bar_chart(selected_column, num_points):
    counts = dashboard.data[selected_column].value_counts().head(num_points)  # Limit the number of points
    x = counts.index
    y = counts.values

    bar_chart = go.Bar(x=x, y=y, marker=dict(color='royalblue', opacity=0.7))
    layout = go.Layout(title=f'{selected_column} Distribution',
                       
                       xaxis=dict(title=selected_column, showgrid=True, zeroline=True),
                       yaxis=dict(title='Count', showgrid=True, zeroline=True),
                       margin=dict(l=1, r=1, t=190, b=40))
    return {'data': [bar_chart], 'layout': layout}


# Choropleth Callback
@app.callback(
    Output('choropleth-container', 'children'),
    [Input('choropleth-dropdown-column', 'value')]
)
def update_choropleth(selected_columns):
    fig = px.choropleth(
        dashboard.data,
        locations='Country',
        locationmode='country names',
        color=selected_columns[0],  # Take only the first selected column for now
        title='Country Data',
        color_continuous_scale=px.colors.sequential.Plasma,
        labels={selected_columns[0]: selected_columns[0]},
    )

    if len(selected_columns) > 1:
        for column in selected_columns[1:]:
            fig.add_trace(px.choropleth(
                dashboard.data,
                locations='Country',
                locationmode='country names',
                color=column,
                color_continuous_scale=px.colors.sequential.Plasma,
                labels={column: column}
            ).data[0])

    # Adjust the color scale legend orientation to horizontal
    fig.update_layout(coloraxis_colorbar=dict(orientation='h'),
                      margin=dict(l=2, r=2, t=150, b=0),
                      xaxis=dict(showgrid=True, zeroline=True),
                      yaxis=dict(showgrid=True, zeroline=True),
                      autosize=True
                      )

    fig.update_layout(autosize=True)
    return dcc.Graph(figure=fig)

# Histogram Callback
@app.callback(
    [Output('histogram', 'figure'),
     Output('explanation', 'children')],
    [Input('hist-dropdown-column', 'value'),
     Input('hist-slider-bins', 'value')]
)
def update_histogram(column, bins):
    x_data = dashboard.data[column]  # Renamed to x_data to avoid conflict with data variable
    histogram_data = [go.Histogram(x=x_data, nbinsx=bins, marker=dict(color='royalblue'))]

    layout = go.Layout(
                       xaxis=dict(title=column, showgrid=True, zeroline=True),
                       yaxis=dict(title='Frequency', showgrid=True, zeroline=True),
                       margin=dict(l=30, r=1, t=40, b=80),
                       bargap=0.05)

    explanation_text = f"The histogram above displays the distribution of {column.lower()} with {bins} bins."

    return {'data': histogram_data, 'layout': layout}, explanation_text

if __name__ == '__main__':
    try:
        app.run_server(debug=False)
    except Exception as e:
        print("Error:", e)
        print("Server busy, please try again later.")
