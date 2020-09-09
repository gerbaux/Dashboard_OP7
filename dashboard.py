# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

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
from sklearn.externals import joblib

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# load model
lGBMclf = joblib.load('LGBM_BestNF.pkl')

DashBoardDF = pd.read_csv("featureDF1K.csv")
shapValues1 = np.load("shapValues1K.npy")

# with open('dtypes.txt') as json_file:
    # dtypesR = json.load(json_file)
    
# df = pd.read_csv("code/FeatureDF.csv", dtype=dtypesR)
# def cleanKaggleTest(df):
    # train_df = df[df['TARGET'].notnull()]
    # test_df = df[df['TARGET'].isnull()]
    # del df
    # gc.collect()
    # return train_df, test_df

# df, testDF = cleanKaggleTest(df)
# del testDF
# gc.collect()

available_features = DashBoardDF.columns


def sortSecond(val):
    return val[1]

NbFeatures = 12
listFeaturesI = [(x, int(y)) for x,y in zip(DashBoardDF.columns,lGBMclf.feature_importances_)]
listFeaturesI.sort(key = sortSecond, reverse=True)

# Coefficient Figure
coef_fig = px.bar(
    y=[x[0] for x in reversed(listFeaturesI[:NbFeatures])],
    x=[x[1] for x in reversed(listFeaturesI[:NbFeatures])],
    orientation="h",
#    color=X_test.columns.isin(num_cols),
    labels={"color": "Is numerical", "x": "Weight on Prediction", "y": "Features"},
    title="Feature importance",
)

app.layout = html.Div([
    html.Div([
        html.H6("Customer Selection"),
        html.Div(["Id: ",
              dcc.Input(id='customer-id', value=0, type='number'),
              dcc.Graph(figure=coef_fig)]),

        html.Div([
            dcc.Dropdown(
                id='xaxis-column',
                options=[{'label': i, 'value': i} for i in available_features],
                value='EXT_SOURCE_1'
            ),
            dcc.Slider(
                id='xy-range-slider',
                min=0,
                max=100,
                value=20
            )

        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='yaxis-column',
                options=[{'label': i, 'value': i} for i in available_features],
                value='EXT_SOURCE_2'
            )
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),

    dcc.Graph(id='indicator-graphic'),

   dcc.Graph(id='indicator-graphic2')

])

@app.callback(
    Output('indicator-graphic2', 'figure'),
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
    fig.update_layout(shapes=[
        dict(
            type= 'line',
            yref= 'paper', 
            y0= dff[xaxis_column_name].min(), 
            y1= dff[xaxis_column_name].max(),
            xref= 'x', 
            x0= dff.iloc[customerId][xaxis_column_name], 
            x1= dff.iloc[customerId][xaxis_column_name],
        )]
    )
    
    return fig

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