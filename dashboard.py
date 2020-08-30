# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import gc
import json
from sklearn.externals import joblib

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv("df10Kx20.csv")

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

available_features = df.columns

app.layout = html.Div([
    html.Div([
        html.H6("Customer Selection"),
        html.Div(["Id: ",
              dcc.Input(id='customer-id', value='0', type='number')]),

        html.Div([
            dcc.Dropdown(
                id='xaxis-column',
                options=[{'label': i, 'value': i} for i in available_features],
                value='EXT_SOURCE_1'
            ),
            dcc.Slider(
                id='xy-range-slider',
                min=0,
                max=50,
                value=5
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

    dcc.Graph(id='indicator-graphic')

#    html.Div(id='output-container-range-slider')
])

@app.callback(
    Output('indicator-graphic', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('xy-range-slider', 'value'),
     Input('customer-id', 'value')])
def update_graph(xaxis_column_name, yaxis_column_name,
                 xyrangeValues, customerId):
    dff = df[[xaxis_column_name, yaxis_column_name, 'TARGET']].dropna()
    RangeX = xyrangeValues*(dff[xaxis_column_name].max()-dff[xaxis_column_name].min())/100
    RangeY = xyrangeValues*(dff[yaxis_column_name].max()-dff[yaxis_column_name].min())/100
    ValX = dff.iloc[customerId][xaxis_column_name]
    ValY = dff.iloc[customerId][yaxis_column_name]
    dff = dff[(dff[xaxis_column_name] > ValX-RangeX) & \
              (dff[xaxis_column_name] < ValX+RangeX)]
    dff = dff[(dff[yaxis_column_name] > ValY-RangeY) & \
              (dff[yaxis_column_name] < ValY+RangeY)]

    fig = px.scatter(x=dff[dff['TARGET'] == 1][xaxis_column_name],
                     y=dff[dff['TARGET'] == 1][yaxis_column_name],
                     color=dff[dff['TARGET'] == 1]['TARGET'])

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    fig.update_xaxes(title=xaxis_column_name, 
                     type='linear') 

    fig.update_yaxes(title=yaxis_column_name, 
                     type='linear') 

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)