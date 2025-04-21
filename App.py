import dash
from dash import html, dcc, dash_table, Input, Output, State
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from Prep_Data import load_student_data, engineer_features, prepare_data_for_training
from Evaluations import evaluate_classification_model

# Initialize app
app = dash.Dash(__name__)
server = app.server

# Load and prepare data
df = load_student_data("Student_performance_data.csv")
df = engineer_features(df)
X_train, X_test, y_train, y_test = prepare_data_for_training(df)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Generate classification report
report = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T.reset_index()
report.rename(columns={'index': 'Class'}, inplace=True)

# Layout
app.layout = html.Div([
    html.H1("BrightPath Academy - Student Performance Classifier", style={'textAlign': 'center'}),
    
    dcc.Tabs([
        dcc.Tab(label='Dashboard', children=[
            html.H2("Classification Report"),
            dash_table.DataTable(
                data=report.round(3).to_dict('records'),
                columns=[{"name": i, "id": i} for i in report.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'},
                page_size=10
            ),
            html.H2("GPA Distribution"),
            dcc.Graph(figure=px.histogram(df, x="GPA", nbins=20, title="Distribution of GPA")),
            html.H2("Feature Importance"),
            dcc.Graph(
                figure=px.bar(
                    x=model.feature_importances_,
                    y=X_train.columns,
                    orientation='h',
                    title="Feature Importances"
                )
            )
        ]),

        dcc.Tab(label='View Student Data', children=[
            html.H2("Full Dataset"),
            dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{"name": i, "id": i} for i in df.columns],
                page_size=15,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '5px'},
            )
        ]),

        dcc.Tab(label='Predict New Student', children=[
            html.H2("Enter Student Information to Predict Performance"),
            html.Div(id="input-fields", children=[
                html.Div([
                    html.Label(f"{col}"),
                    dcc.Input(id=f'input-{col}', type='number', placeholder=f'Enter {col}', step=0.01)
                ], style={'marginBottom': '10px'}) for col in X_train.columns
            ]),
            html.Button("Predict", id='predict-button', n_clicks=0, style={'marginTop': '10px'}),
            html.Div(id='prediction-output', style={'marginTop': '20px', 'fontSize': '20px'})
        ])
    ])
])

# Prediction Callback
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [State(f'input-{col}', 'value') for col in X_train.columns]
)
def make_prediction(n_clicks, *input_values):
    if n_clicks > 0:
        if None in input_values:
            return "Please fill in all input fields."
        input_df = pd.DataFrame([input_values], columns=X_train.columns)
        prediction = model.predict(input_df)[0]
        return f"Predicted Student Performance: **{prediction}**"
    return ""

# Run app
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    print("Launching Dash app...")
    app.run(debug=True, host="0.0.0.0", port=port)
