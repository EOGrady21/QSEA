"""
Quality System Evaluation Application (QSEA)

This application provides a user interface for reviewing and quality-controlling
oceanographic data. It allows users to upload data files, visualize the data in
various ways, perform quality control checks, and download the reviewed data.

The application is built using Dash, a Python framework for building web applications.
"""

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table
import os

# Import callbacks to register them (even if not directly used in layout)
import callbacks

# --- App Initialization ---
# Use Dash Bootstrap Components for layout enhancements
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server # Expose server for deployment (e.g., Gunicorn)

# --- Reusable Components ---
def build_upload_section():
    """
    Creates the upload section of the application.

    This section allows users to:
    1. Enter their username (for logging purposes)
    2. Upload a CSV data file by dragging and dropping or selecting from their computer

    Returns:
        A Dash Bootstrap Card component containing the upload interface
    """
    return dbc.Card(
        dbc.CardBody([
            html.H4("1. Upload Data & Reviewer Info", className="card-title"),
            dcc.Input(
                id='input-sme-username',
                type='text',
                placeholder='Enter Your Username (for logging)',
                style={'marginBottom': '10px', 'width': '100%'}
            ),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select CSV File')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px 0'
                },
                multiple=False # Allow only single file upload
            ),
            html.Div(id='upload-status', style={'marginTop': '10px'}) # To show loading messages/errors
        ])
    )

def build_review_panel():
    """
    Creates the manual review panel of the application.

    This panel allows users to:
    1. View details of selected data points
    2. Change quality flags for selected data
    3. Add comments explaining their changes
    4. Submit their reviews

    The panel is initially hidden and becomes visible when data is selected.

    Returns:
        A Dash Bootstrap Card component containing the review interface
    """
    return dbc.Card(
        dbc.CardBody([
            html.H4("3. Manual Review Panel", className="card-title"),
            # This div will hold the text like "Select rows..." or selected point details
            html.Div(id='review-panel-content', children="Select rows in the table below to enable review."),
            # This div will show status messages after review submission
            html.Div(id='review-status', style={'marginTop': '10px'}),
            # --- Container for Review Controls (Initially Hidden) ---
            html.Div(id='review-controls-container', style={'display': 'none'}, children=[
                 dcc.Dropdown(
                     id='dropdown-final-flag',
                     # options are defined in callbacks.py, load them here if static or update dynamically
                     # For now, just define the component placeholder
                     options=[], # Placeholder options
                     placeholder="Select New Final Flag",
                     clearable=False,
                     style={'margin-bottom': '10px'}
                 ),
                 dcc.Textarea(
                     id='textarea-sme-comment',
                     placeholder="Enter mandatory comment explaining the change...",
                     style={'width': '100%', 'height': '80px', 'margin-bottom': '10px'}
                 ),
                 html.Button('Submit Review', id='button-submit-review', n_clicks=0, className='btn btn-primary')
            ])
            # --- End Review Controls Container ---
        ]),
        id='review-panel',
        style={'display': 'none'} # The entire panel is still hidden initially
    )

def build_output_section():
    """
    Creates the download results section of the application.

    This section allows users to:
    1. Download the updated CSV file with quality flags
    2. Download a summary report of the quality control process

    The section is initially hidden and becomes visible when data is loaded.

    Returns:
        A Dash Bootstrap Card component containing download buttons
    """
    return dbc.Card(
         dbc.CardBody([
             html.H4("4. Download Results", className="card-title"),
             html.Button("Download Updated CSV", id="button-download-csv", className="btn btn-success", style={'marginRight': '10px'}),
             dcc.Download(id="download-csv"),
             html.Button("Download Summary Report", id="button-download-report", className="btn btn-info"),
             dcc.Download(id="download-report"),
         ]),
         id='output-buttons',
         style={'display': 'none'} # Initially hidden
     )

# --- App Layout ---
app.layout = dbc.Container(fluid=True, children=[
    # --- Title ---
    dbc.Row(dbc.Col(html.H1("Quality System Evaluation Application"), width=12)),

    # --- Stores for Data Sharing ---
    dcc.Store(id='store-data'), # Main DataFrame store
    dcc.Store(id='store-filename'), # Input filename
    dcc.Store(id='store-sme-username'), # SME username

    # --- Top Row: Upload and Review Panels ---
    dbc.Row([
        dbc.Col(build_upload_section(), md=6), # Upload section takes half width on medium screens and up
        dbc.Col(build_review_panel(), md=6)  # Review panel takes the other half
    ], className="mb-3"), # Add margin below the row

    # --- Main Content Tabs ---
    dbc.Row(
        dbc.Col(
            dbc.Tabs(id="tabs-main", children=[
                dbc.Tab(label="Map View", tab_id="tab-map", children=[
                    dcc.Graph(id='map-graph', config={'scrollZoom': True})
                ]),
                 dbc.Tab(label="Profiles / Time Series", tab_id="tab-profiles", children=[
                    # Add controls here if needed (e.g., dropdown to select parameter)
                    dcc.Graph(id='profile-time-series-graph')
                 ]),
                 dbc.Tab(label="T-S / Nutrient Plots", tab_id="tab-ts-nut", children=[
                     dcc.Graph(id='ts-nutrient-graph')
                 ]),
                 dbc.Tab(label="QC Comparisons", tab_id="tab-qc-comp", children=[
                     dcc.Graph(id='qc-comparison-graph') # Placeholder
                 ]),
                 dbc.Tab(label="Data Table", tab_id="tab-table", children=[
                     html.Div(id='data-table-info', style={'margin': '10px 0'}), # Info text above table
                     html.Div(id='data-table-container', children=html.Div("Load data to view table."))
                 ]),
                 dbc.Tab(label="Summary", tab_id="tab-summary", children=[
                     # Placeholder for summary text output if desired in UI
                     html.Div("Summary report will be available for download.")
                 ]),
            ], style={'display': 'none'}) # Initially hidden until data loads
        )
    ),

    # --- Bottom Row: Output/Download Buttons ---
    dbc.Row(
        dbc.Col(build_output_section(), width=12),
        className="mt-3" # Add margin above the row
    )
])

# --- Run the App ---
if __name__ == '__main__':
    # Create qc_data directory if it doesn't exist
    if not os.path.exists('qc_data'):
        os.makedirs('qc_data')
        print("Created 'qc_data' directory for external files (ranges, climatology, etc.). Please populate it.")
    # Run app
    app.run(debug=True) # Turn debug=False for production
