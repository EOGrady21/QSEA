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
server = app.server # local dev

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
                # Hidden store to track the selected QC flag value
                dcc.Store(id='selected-qc-flag-store'),

                # Quick flag selection buttons
                html.Div([
                    html.Label("Select New QC Flag:", className="fw-bold mb-2"),
                    dbc.ButtonGroup([
                        dbc.Button("0: NoQC", id="flag-btn-0", color="secondary", outline=True, size="sm", n_clicks=0),
                        dbc.Button("1: Good", id="flag-btn-1", color="success", outline=True, size="sm", n_clicks=0),
                        dbc.Button("2: Inconsistent", id="flag-btn-2", color="info", outline=True, size="sm",
                                   n_clicks=0),
                        dbc.Button("3: Doubtful", id="flag-btn-3", color="warning", outline=True, size="sm",
                                   n_clicks=0),
                        dbc.Button("4: Bad", id="flag-btn-4", color="danger", outline=True, size="sm", n_clicks=0),
                        dbc.Button("5: Modified", id="flag-btn-5", color="primary", outline=True, size="sm",
                                   n_clicks=0),
                        dbc.Button("7: Investigate", id="flag-btn-7", color="dark", outline=True, size="sm",
                                   n_clicks=0),
                    ], className="mb-3")
                ]),

                # Display selected flag
                dbc.Alert(id="selected-flag-display", color="info", className="mb-3"),

                # Predefined comments section
                html.Div([
                    html.Label("Quick Comments:", className="fw-bold"),
                    dcc.Dropdown(
                        id="predefined-comments-dropdown",
                        options=[
                            {"label": "Spike detected", "value": "spike"},
                            {"label": "Drift observed", "value": "drift"},
                            {"label": "Sensor malfunction", "value": "sensor_malfunction"},
                            {"label": "Environmental interference", "value": "env_interference"},
                            {"label": "Data gap", "value": "data_gap"},
                            {"label": "Outlier - confirmed valid", "value": "valid_outlier"},
                            {"label": "Biofouling suspected", "value": "biofouling"},
                            {"label": "Questionable gradient", "value": "gradient"},
                            {"label": "Outside expected range", "value": "out_of_range"},
                            {"label": "Inconsistent with nearby measurements", "value": "inconsistent"},
                            {"label": "Instrument calibration issue", "value": "calibration"},
                            {"label": "Data manually verified", "value": "verified"}
                        ],
                        placeholder="Select predefined comments...",
                        multi=True,
                        clearable=True,
                        className="mb-2"
                    )
                ]),

                # Additional comment textarea
                dcc.Textarea(
                    id='textarea-sme-comment',
                    placeholder="Add additional comments or elaborate on selections...",
                    style={'width': '100%', 'height': '80px', 'margin-bottom': '10px'}
                ),

                # Submit button
                html.Button('Submit Review', id='button-submit-review', n_clicks=0, className='btn btn-primary')
            ])
            # --- End Review Controls Container ---
        ]),
        id='review-panel',
        style={'display': 'none'}  # The entire panel is still hidden initially
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

    # Flag Update Trigger Store
    # This store acts as a trigger to notify other components when a QC flag has been updated.
    # It stores information about the most recently updated row(s) to enable automatic selection
    # in the data table for easier review workflow.
    dcc.Store(id='flag-update-trigger', data={'row_indices': [], 'timestamp': None}),

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
