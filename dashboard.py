# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pandas as pd
import gc
import json
import joblib
import dash_bootstrap_components as dbc
from utils import utils
from utils import preprocess

Threshold = 0.276229

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# load model
lGBMclf = joblib.load('LGBM_BestNF.pkl')

# Building the pre-processed data during the first time the Dashboard is launched
# For next runs of the dashboard, the csv file featureDF10K.csv will be read directly
if not os.path.isfile('featureDF10K.csv'):
    preprocess.BuildDataFromZipFile(lGBMclf, Threshold)

DashBoardDF = pd.read_csv("featureDF10K.csv")

# Reading rhe Shap values that have been preprocessed earlier.
# Could be put in preprocessing step BuildDataFromZipFile as well
shapValues1 = np.load("shapValues10K.npy")

available_features = DashBoardDF.columns

def getCustomerFeatures(CustId, NbFeatures = 12):
    maxFeatureId = sorted(range(len(shapValues1[CustId])),
                          key=lambda x: abs(shapValues1[CustId][x]))[-NbFeatures:]
    FeatureNames = np.empty(NbFeatures, dtype=object)
    FeatureShapValues = np.empty(NbFeatures, dtype=float)
    FeatureStdValues = np.empty(NbFeatures, dtype=float)
    for i, Id in enumerate(maxFeatureId):
        FeatureNames[i] = DashBoardDF.columns[Id]
        FeatureShapValues[i] = shapValues1[CustId][Id]
        FeatureStdValues[i] = DashBoardDF.iloc[CustId][Id]
    positive = FeatureShapValues > 0
    colors = list(map(lambda x: 'red' if x else 'blue', positive))
    return (FeatureNames, FeatureShapValues, colors)


def sortSecond(val):
    return val[1]

# Building the list of 12 most important features to be reported in a figure
NbFeatures = 12
listFeaturesI = [(x, int(y)) for x,y in zip(DashBoardDF.columns,lGBMclf.feature_importances_)]
listFeaturesI.sort(key = sortSecond, reverse=True)

# Coefficient Figure
coef_fig = px.bar(
    y=[x[0] for x in reversed(listFeaturesI[:NbFeatures])],
    x=[x[1] for x in reversed(listFeaturesI[:NbFeatures])],
    orientation="h",
    labels={"x": "Weight on Prediction", "y": "Features"},
    title="Mean Feature Importance",
)
coef_fig.update_traces(marker_color='orange')
coef_fig.update_layout(width=700, height=300, bargap=0.05,
                       margin=dict(l=100, r=100, t=50, b=50))

# Data for probability distribution
NbBins = 500
dist = pd.cut(DashBoardDF['Proba'], bins=NbBins).value_counts()
dist.sort_index(inplace=True)
ticks = np.linspace(0, 1, NbBins)
DashBoradTH=int(Threshold*len(dist))

# Layout of the Dashboard
app.layout = html.Div([
    html.Div([
        html.Div([
            html.H6("Customer Selection"),
            html.Div([
                   html.P("Custumer Id: "),
                   dbc.Input(id="customer-id", value=0, type="number", min=0, max=DashBoardDF.shape[0]),
                   dbc.Card(
                     [
                        html.H3(id='cust-answer', className="card-title"),
                        html.H6(id='cust-score', className="card-title"),
                     ]),
                   html.Br(),
                  ], style={'width': '48%', 'display': 'inline-block'}),
                  html.Div([dcc.Graph(id='proba-score')
                  ], style={'width': '48%', 'display': 'inline-block'}),

                  html.Div([
                    dcc.Graph(id='feature-graphic1')],
                    style={'width': '100%', 'display': 'inline-block'}),

    ],
    style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Graph(figure=coef_fig),
    ],
    style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            # dcc.Dropdown(
                # id='xaxis-column',
                # options=[{'label': i, 'value': i} for i in available_features],
                # value='EXT_SOURCE_1'
            # ),
            utils.OptionMenu(id="xaxis-column", label="X axis", values=available_features,
                       value='EXT_SOURCE_1'),
            dcc.Graph(id='xaxis-distribution')

        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            # dcc.Dropdown(
                # id='yaxis-column',
                # options=[{'label': i, 'value': i} for i in available_features],
                # value='EXT_SOURCE_2'
            # ),
            utils.OptionMenu(id="yaxis-column", label="Y axis", values=available_features,
                       value='EXT_SOURCE_2'),
            dcc.Graph(id='yaxis-distribution')
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),

    dcc.Graph(id='indicator-graphic'),
    html.Div(["Zoom: ",
        dcc.Slider(
                id='xy-range-slider',
                min=0,
                max=100,
                value=20
        ),
             ],
             style={'width': '48%', 'display': 'inline-block'}),
])

# Callbacks functions

# Callback for the customer feature importance
@app.callback(
    Output('feature-graphic1', 'figure'),
    [Input('customer-id', 'value')])
def update_graph(customerId):
    FeatureNames, FeatureShapValues, colors = getCustomerFeatures(customerId)
    cust_coef_fig = px.bar(
        y=FeatureNames,
        x=FeatureShapValues,
        orientation="h",
#        color=colors,
        labels={"x": "Weight on Prediction", "y": "Features"},
        title="Customer Feature Importance",
    )
    cust_coef_fig.update_traces(marker_color=colors)
    cust_coef_fig.update_layout(width=700, height=300, bargap=0.05,
                                margin=dict(l=100, r=100, t=50, b=50))

    return cust_coef_fig

# Callback for the prediction and score of the selected customer
@app.callback(
    Output('cust-answer', 'children'),
    [Input('customer-id', 'value')])
def update_score(customerId):
    Score = DashBoardDF.loc[customerId, 'Proba']
    Answer = 'Accepted' if Score <= Threshold else 'Refused'
    return f"{Answer}"

@app.callback(
    Output('cust-score', 'children'),
    [Input('customer-id', 'value')])
def update_score(customerId):
    Score = DashBoardDF.loc[customerId, 'Proba']
    return f"Score: {Score:.2f}"

# Callback for the probability distribution 
@app.callback(
    Output('proba-score', 'figure'),
    [Input('customer-id', 'value')])
def update_graph(customerId):
    fig = go.Figure(data = go.Scatter(x=ticks[:DashBoradTH],
                                      y=dist[:DashBoradTH],
                                      mode='lines',
                                      marker=dict(color='blue'),
                                      name='Success'))
    fig.add_trace(go.Scatter(x=ticks[DashBoradTH:],
                             y=dist[DashBoradTH:],
                             mode='lines',
                             marker=dict(color='red'),
                             name='Default'))
    Score = DashBoardDF.loc[customerId, 'Proba']
    rank = int(Score*len(dist))-1
    fig.add_trace(go.Scatter(x=[Score], y=[dist[rank]], mode='markers',
                             marker=dict(size=15, color='black'),
                             name='Customer'))
    fig.update_layout(
        title="Prediction Distribution",
        margin=dict(l=20, r=20, t=40, b=20),
        width=600, height=150,
        paper_bgcolor="LightSteelBlue",
    )

    return fig

# Callback for the X axis feature distribution for accepted and refused customers 
@app.callback(
    Output('xaxis-distribution', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('customer-id', 'value')])
def update_graph(xaxis_column_name,
                 customerId):
    dff = DashBoardDF[[xaxis_column_name, 'Predict']].dropna()
    hist_data = [dff[dff['Predict'] == 1][xaxis_column_name], \
                 dff[dff['Predict'] == 0][xaxis_column_name]]
    group_labels = ['Default', 'Success']
    colors = ['red', 'blue']
    fig = ff.create_distplot(hist_data, group_labels,
                             show_hist=False, colors=colors,
                             show_rug=False)
    title="{} Feature Distribution".format(xaxis_column_name)
    fig.update_layout(shapes=[
        dict(
            type= 'line',
            yref= 'paper',
            y0= 0,
            y1= 1,
            xref= 'x',
            x0= dff.iloc[customerId][xaxis_column_name],
            x1= dff.iloc[customerId][xaxis_column_name],
        )],
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        width=400, height=200,
        paper_bgcolor="LightSteelBlue",
    )

    return fig

# Callback for the Y axis feature distribution for accepted and refused customers 
@app.callback(
    Output('yaxis-distribution', 'figure'),
    [Input('yaxis-column', 'value'),
     Input('customer-id', 'value')])
def update_graph(yaxis_column_name,
                 customerId):
    dff = DashBoardDF[[yaxis_column_name, 'Predict']].dropna()
    hist_data = [dff[dff['Predict'] == 1][yaxis_column_name], \
                 dff[dff['Predict'] == 0][yaxis_column_name]]
    group_labels = ['Default', 'Success']
    colors = ['red', 'blue']
    fig = ff.create_distplot(hist_data, group_labels,
                             show_hist=False, colors=colors,
                             show_rug=False)
    title="{} Feature Distribution".format(yaxis_column_name)
    fig.update_layout(shapes=[
        dict(
            type= 'line',
            yref= 'paper',
            y0= 0,
            y1= 1,
            xref= 'x',
            x0= dff.iloc[customerId][yaxis_column_name],
            x1= dff.iloc[customerId][yaxis_column_name],
        )],
        title=title,
        margin=dict(l=20, r=20, t=40, b=20),
        width=400, height=200,
        paper_bgcolor="LightSteelBlue",
    )

    return fig


# Callback for the scatter figure with neighborhood customers, taking into account
# the zoom factor
@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('xy-range-slider', 'value'),
     Input('customer-id', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name,
                 xyrangeValues, customerId):
    dff = DashBoardDF[[xaxis_column_name, yaxis_column_name, 'Predict']].dropna()
    RangeX = xyrangeValues*(dff[xaxis_column_name].max()-dff[xaxis_column_name].min())/100
    RangeY = xyrangeValues*(dff[yaxis_column_name].max()-dff[yaxis_column_name].min())/100
    ValX = dff.iloc[customerId][xaxis_column_name]
    ValY = dff.iloc[customerId][yaxis_column_name]
    dff = dff[(dff[xaxis_column_name] > ValX-RangeX) & \
              (dff[xaxis_column_name] < ValX+RangeX)]
    dff = dff[(dff[yaxis_column_name] > ValY-RangeY) & \
              (dff[yaxis_column_name] < ValY+RangeY)]

    fig = go.Figure(data = go.Scatter(x=dff[dff['Predict'] == 1][xaxis_column_name],
                     y=dff[dff['Predict'] == 1][yaxis_column_name],
                     mode='markers',
                     marker=dict(size=6, color='red'),
                     name='Default'))
    fig.add_trace(go.Scatter(x=dff[dff['Predict'] == 0][xaxis_column_name],
                     y=dff[dff['Predict'] == 0][yaxis_column_name],
                     mode='markers',
                     marker=dict(size=6, color='blue'),
                     name='Success'))
    fig.add_trace(go.Scatter(x=[ValX], y=[ValY], mode='markers',
                             marker=dict(size=15, color='black'),
                             name='Customer'))

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig.update_xaxes(title=xaxis_column_name, 
                     type='linear') 

    fig.update_yaxes(title=yaxis_column_name, 
                     type='linear') 

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)