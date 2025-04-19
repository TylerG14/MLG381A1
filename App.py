import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from Prep_Data import load_student_data, engineer_features, prepare_data_for_training
from Evaluations import evaluate_classification_model

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server  # For deployment on Render

# Load and process data
df = load_student_data("Student_performance_data.csv")
df = engineer_features(df)
X_train, X_test, y_train, y_test = prepare_data_for_training(df)

# Train a baseline model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Classification Report as DataFrame
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().reset_index().rename(columns={'index': 'Class'})

# Layout
app.layout = html.Div([
    html.H1("BrightPath Academy - Student Performance Classifier", style={'textAlign': 'center'}),
    
    html.H2("Classification Report", style={'marginTop': '20px'}),
    dash_table.DataTable(
        data=report_df.round(3).to_dict('records'),
        columns=[{"name": i, "id": i} for i in report_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'},
        page_size=10,
    ),

    html.H2("GPA Distribution"),
    dcc.Graph(
        figure=px.histogram(df, x="GPA", nbins=20, title="Distribution of GPA")
    ),

    html.H2("Feature Importance"),
    dcc.Graph(
        figure=px.bar(
            x=model.feature_importances_,
            y=X_train.columns,
            orientation='h',
            title="Feature Importances from Random Forest"
        )
    )
])

# Run app
if __name__ == '__main__':
    app.run(debug=True)