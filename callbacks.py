"""
Callbacks Module for Quality System Evaluation Application

This module contains all the callback functions that handle the interactive
elements of the application. Callbacks connect user actions (like clicking buttons
or uploading files) to changes in the application state and user interface.

Key functionality includes:
- Processing uploaded data files
- Running automated quality control checks
- Updating visualizations and data tables
- Handling manual data review and flag changes
- Generating and downloading reports and data files
"""

import dash
from dash import dcc, html, dash_table, callback, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import io
import datetime
from pathlib import Path

# Import functions from other modules
from file_handling import (
    parse_uploaded_file, load_external_data, format_dataframe_for_output,
    append_to_log, generate_summary_report, QC_FLAG_DEFINITIONS, EXPECTED_COLUMNS
)
from qc_functions import run_all_qc

# Define flag options for SME review dropdown (exclude 'Requires Investigation')
REVIEW_FLAG_OPTIONS = [{'label': f"{k} - {v}", 'value': k} for k, v in QC_FLAG_DEFINITIONS.items() if k != 7]

# Maximum number of rows to display directly in table (performance)
MAX_TABLE_ROWS = 5000 # Adjust as needed

# --- Callback for File Upload, Processing, and Auto QC ---
@callback(
    Output('store-data', 'data'),
    Output('store-filename', 'data'),
    Output('store-sme-username', 'data'),
    Output('upload-status', 'children'),
    Output('tabs-main', 'style'), # Show/hide tabs
    Output('review-panel', 'style', allow_duplicate=True), # Hide review panel initially
    Output('output-buttons', 'style'), # Hide output buttons initially
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('input-sme-username', 'value'),
    prevent_initial_call=True
)
def process_upload(contents, filename, sme_username):
    """
    Processes uploaded data files and runs automated quality control checks.

    This callback is triggered when a user uploads a file. It:
    1. Loads external reference data needed for QC checks
    2. Parses the uploaded file into a pandas DataFrame
    3. Runs automated quality control checks on the data
    4. Prepares the data for display and storage
    5. Updates the UI to show the data and enable review features

    Args:
        contents (str): Base64-encoded contents of the uploaded file
        filename (str): Name of the uploaded file
        sme_username (str): Username of the person uploading the file

    Returns:
        tuple: Multiple outputs that update various parts of the UI:
            - Processed data for storage
            - Filename for storage
            - Username for storage
            - Status message about the upload
            - Display settings for tabs
            - Display settings for review panel
            - Display settings for output buttons
    """
    if contents is None:
        raise PreventUpdate

    start_time = datetime.datetime.now()
    print(f"Processing upload for file: {filename}")

    # Load external QC data (ranges, climatology etc.)
    external_data, load_errors = load_external_data()
    error_messages = "\n".join(load_errors)
    status_prefix = f"External Data Load Status:\n{error_messages}\n\n" if error_messages else ""

    # Parse the uploaded file
    df, parse_message = parse_uploaded_file(contents, filename)

    if df is None:
        # Parsing failed
        return (None, None, None, html.Div(['Error loading file: ', html.Br(), parse_message], className="alert alert-danger"),
                {'display': 'none'}, {'display': 'none'}, {'display': 'none'})

    # Run Automated QC Checks
    try:
        df_processed = run_all_qc(df, external_data) # Pass external data here
        status_message = f"Successfully loaded and ran initial QC on '{filename}'."
        status_style = "alert alert-success"
    except Exception as e:
         # QC failed
         print(f"Error during automated QC: {e}")
         return (None, None, None, html.Div(['Error during automated QC: ', html.Br(), str(e)], className="alert alert-danger"),
                 {'display': 'none'}, {'display': 'none'}, {'display': 'none'})

    end_time = datetime.datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    print(f"File processing and QC took {processing_time:.2f} seconds.")

    # Store processed data, filename, and username
    # Convert non-serializable types before storing if necessary (e.g., lists in details col)
    # Dataframes are best stored as dicts for dcc.Store
    # Ensure auto_qc_details is suitable for dcc.Store/DataTable
    if 'auto_qc_details' in df_processed.columns:
        # Convert lists/dicts to strings, handle None/NaN
        df_processed['auto_qc_details'] = df_processed['auto_qc_details'].apply(
            lambda x: str(x) if isinstance(x, (list, dict)) else ('' if pd.isna(x) else x)
        )
        # If you ONLY need auto_qc_details_str for the table, you could even drop the original auto_qc_details column here
        # df_processed = df_processed.drop(columns=['auto_qc_details']) # Only if auto_qc_details_str is sufficient
    elif 'auto_qc_details_str' in df_processed.columns:
        # If only auto_qc_details_str exists, ensure it's always a string or None/''
        df_processed['auto_qc_details_str'] = df_processed['auto_qc_details_str'].apply(
            lambda x: '' if pd.isna(x) else str(x)
        )

    data_dict = df_processed.to_dict('records')

    return (data_dict, filename, sme_username,
            html.Div([status_prefix + status_message], className=status_style),
            {'display': 'block'}, # Show tabs
            {'display': 'none'},  # Keep review panel hidden
            {'display': 'block'}  # Show output buttons
           )


# --- Callback to Update Plots and Table when Data Store Changes ---
@callback(
    Output('map-graph', 'figure'),
    Output('profile-time-series-graph', 'figure'),
    Output('ts-nutrient-graph', 'figure'),
    Output('qc-comparison-graph', 'figure'), # Placeholder output
    Output('data-table-container', 'children'), # Update table container
    Output('data-table-info', 'children'), # Info about displayed rows
    Input('store-data', 'data'), # Triggered when data is loaded or updated
    State('store-filename', 'data'),
    prevent_initial_call=True
)
def update_visualizations_and_table(data_dict, filename):
    """
    Updates all visualizations and the data table when new data is loaded.

    This callback is triggered whenever the data store changes (e.g., after a file
    upload or after data is modified). It creates several visualizations:
    1. A map showing the geographic locations of data points
    2. Profile or time series plots of the data
    3. Temperature-salinity plots and nutrient plots
    4. Quality control comparison plots
    5. A data table showing the raw data with quality flags

    Args:
        data_dict (list): List of dictionaries representing the data (from DataFrame.to_dict('records'))
        filename (str): Name of the data file being displayed

    Returns:
        tuple: Multiple outputs that update various parts of the UI:
            - Map figure
            - Profile/time series figure
            - T-S/nutrient figure
            - QC comparison figure
            - Data table component
            - Information about the displayed data
    """
    if data_dict is None:
        print("Update Viz: No data in store.")
        # Return empty figures and table placeholder
        empty_fig = go.Figure()
        return empty_fig, empty_fig, empty_fig, empty_fig, html.Div("Load data to view table."), ""

    start_time = datetime.datetime.now()
    print("Updating visualizations and table...")
    df = pd.DataFrame(data_dict)

    # Ensure essential columns are numeric for plotting
    for col in ['DIS_HEADER_SLAT', 'DIS_HEADER_SLON', 'DIS_HEADER_START_DEPTH', 'DIS_DETAIL_DATA_VALUE', 'auto_qc_flag']:
         if col in df.columns:
              df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- 1. Map Figure ---
    map_fig = go.Figure()
    if not df.empty and 'DIS_HEADER_SLAT' in df.columns and 'DIS_HEADER_SLON' in df.columns:
        # Handle potential NaNs gracefully
        plot_df_map = df.dropna(subset=['DIS_HEADER_SLAT', 'DIS_HEADER_SLON', 'auto_qc_flag']).copy()
        if not plot_df_map.empty:
             plot_df_map['auto_qc_flag_str'] = plot_df_map['auto_qc_flag'].map(QC_FLAG_DEFINITIONS).fillna('Unknown')
             # Add hover text
             hover_texts = plot_df_map.apply(lambda row: f"Station: {row.get('EVENT_COLLECTOR_STN_NAME', 'N/A')}<br>" + \
                                                         f"Event ID: {row.get('EVENT_COLLECTOR_EVENT_ID', 'N/A')}<br>" + \
                                                         f"Lat: {row['DIS_HEADER_SLAT']:.4f}, Lon: {row['DIS_HEADER_SLON']:.4f}<br>" + \
                                                         f"Auto QC: {row['auto_qc_flag_str']} ({int(row['auto_qc_flag'])})", axis=1)

             map_fig = px.scatter_mapbox(
                 plot_df_map,
                 lat="DIS_HEADER_SLAT",
                 lon="DIS_HEADER_SLON",
                 color="auto_qc_flag",
                 color_continuous_scale=px.colors.sequential.Viridis_r, # Or choose a categorical scale if preferred
                 hover_name=hover_texts,
                 # hover_data=["EVENT_COLLECTOR_EVENT_ID", "auto_qc_flag"], # Add more hover data if needed
                 title="Station Map (Colored by Auto QC Flag)",
                 zoom=3,
                 height=500
             )
             map_fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":40,"l":0,"b":0})


    # --- 2. Determine Fixed vs. Multiple Stations (based on loaded data) ---
    is_fixed_station = False
    station_plot_title = "Depth Profiles (Multiple Stations)"
    if not df.empty and 'DIS_HEADER_SLAT' in df.columns and 'DIS_HEADER_SLON' in df.columns:
        unique_coords = df.dropna(subset=['DIS_HEADER_SLAT', 'DIS_HEADER_SLON'])[['DIS_HEADER_SLAT', 'DIS_HEADER_SLON']].drop_duplicates()
        if len(unique_coords) > 1:
            # Check if all coords are within a small radius (e.g., 100m) - simplified check here
            lat_range = unique_coords['DIS_HEADER_SLAT'].max() - unique_coords['DIS_HEADER_SLAT'].min()
            lon_range = unique_coords['DIS_HEADER_SLON'].max() - unique_coords['DIS_HEADER_SLON'].min()
            # Very rough check, assumes degrees ~ 111km at equator. Needs refinement.
            if lat_range < 0.001 and lon_range < 0.001: # Approx 100m
                 is_fixed_station = True
        elif len(unique_coords) == 1:
             is_fixed_station = True # Only one coordinate pair

        if is_fixed_station:
            station_plot_title = "Parameter Time Series (Fixed Station)"

    # --- 3. Profile / Time Series Plot ---
    station_fig = go.Figure()
    plot_df_station = df.dropna(subset=['DIS_HEADER_START_DEPTH', 'DIS_DETAIL_DATA_VALUE', 'auto_qc_flag', 'DATA_TYPE_METHOD', 'DATETIME']).copy()
    # Identify key parameters (adjust list as needed)
    key_parameters = plot_df_station['DATA_TYPE_METHOD'].unique()[:5] # Limit initial plot complexity
    if not plot_df_station.empty:
        if is_fixed_station:
             station_fig = px.scatter(
                 plot_df_station[plot_df_station['DATA_TYPE_METHOD'].isin(key_parameters)],
                 x="DATETIME",
                 y="DIS_DETAIL_DATA_VALUE",
                 color="auto_qc_flag",
                 facet_row="DATA_TYPE_METHOD", # Separate plot per parameter
                 labels={"DIS_DETAIL_DATA_VALUE": "Value", "DATETIME": "Time"},
                 title=station_plot_title
             )
             station_fig.update_yaxes(matches=None) # Unlink y-axes
        else: # Multiple stations -> Profile plot
             station_fig = px.scatter(
                 plot_df_station[plot_df_station['DATA_TYPE_METHOD'].isin(key_parameters)],
                 x="DIS_DETAIL_DATA_VALUE",
                 y="DIS_HEADER_START_DEPTH",
                 color="auto_qc_flag",
                 facet_col="DATA_TYPE_METHOD", # Separate plot per parameter
                 hover_data=['EVENT_COLLECTOR_EVENT_ID'],
                 labels={"DIS_DETAIL_DATA_VALUE": "Value", "DIS_HEADER_START_DEPTH": "Depth (m)"},
                 title=station_plot_title
             )
             station_fig.update_yaxes(autorange="reversed") # Depth increases downwards
             station_fig.update_xaxes(matches=None) # Unlink x-axes

        station_fig.update_layout(height=300 * len(key_parameters)) # Adjust height based on params


    # --- 4. T-S / Nutrient Plot ---
    ts_nut_fig = go.Figure()
    # Requires specific DATA_TYPE_METHOD for Temp, Sal, Nitrate, Phosphate
    temp_methods = [m for m in df['DATA_TYPE_METHOD'].unique() if 'TEMP' in m]
    sal_methods = [m for m in df['DATA_TYPE_METHOD'].unique() if 'PSAL' in m] # Assuming Practical Salinity
    nitrate_methods = [m for m in df['DATA_TYPE_METHOD'].unique() if 'NTRA' in m] # Adjust if needed
    phos_methods = [m for m in df['DATA_TYPE_METHOD'].unique() if 'PHOS' in m] # Adjust if needed

    if temp_methods and sal_methods:
        # Pivot or merge to get Temp and Sal on the same row (per profile/depth)
        # Simple approach: plot all points, may not be true T-S pairs if depths differ
        plot_df_ts = df[df['DATA_TYPE_METHOD'].isin(temp_methods + sal_methods)].copy()
        # TODO: A more robust approach involves merging/pivoting data based on
        # EVENT_COLLECTOR_EVENT_ID and DIS_HEADER_START_DEPTH (or interpolating)
        # to get actual T/S pairs. This is a simplified plot for now.
        if 'DIS_DETAIL_DATA_VALUE' in plot_df_ts.columns:
             ts_nut_fig = px.scatter(
                 plot_df_ts.dropna(subset=['DIS_DETAIL_DATA_VALUE', 'auto_qc_flag']),
                 x="DIS_DETAIL_DATA_VALUE", # This needs refining based on actual parameter
                 y="DIS_DETAIL_DATA_VALUE", # This needs refining based on actual parameter
                 color="auto_qc_flag",
                 title="Property-Property Plot (Placeholder - Requires T/S pairing)"
             )
             # Add specific T-S plot logic here when data is paired

    # --- 5. QC Comparison Plot (Placeholder) ---
    qc_comp_fig = go.Figure()
    qc_comp_fig.update_layout(title="QC Comparison Plot (Placeholder - Needs specific QC test viz)")
    # TODO: Implement plots comparing data against QC thresholds (e.g., global ranges, climatology)

    # --- 6. Data Table ---
    # Define columns to display (subset for performance/clarity)
    display_columns = [
        'MISSION_DESCRIPTOR', 'EVENT_COLLECTOR_EVENT_ID', 'DIS_HEADER_START_DEPTH',
        'DATA_TYPE_METHOD', 'DIS_DETAIL_DATA_VALUE',
        'auto_qc_flag', 'DIS_DETAIL_DATA_QC_CODE', 'auto_qc_details_str' # Use the string version for display
        # Add 'DIS_SAMPLE_KEY_VALUE' if needed for identification
    ]
    # Ensure all display columns exist in the dataframe
    cols_to_show = [col for col in display_columns if col in df.columns]

    # Limit rows for initial display in DataTable for performance
    table_df = df # Use the full dataframe for filtering/sorting
    display_rows = min(len(table_df), MAX_TABLE_ROWS)
    table_info_text = f"Displaying {display_rows} of {len(table_df)} rows. Use filters to narrow results." if len(table_df) > MAX_TABLE_ROWS else f"Total rows: {len(table_df)}"

    data_table = dash_table.DataTable(
        id='data-table',
        columns=[{"name": i, "id": i, "editable": (i == 'DIS_DETAIL_DATA_QC_CODE')} for i in cols_to_show],
        # Pass only a subset of data initially if df is very large
        data=table_df.head(MAX_TABLE_ROWS).to_dict('records'), # Use .head() for large datasets
        editable=True, # Only effective for columns marked editable
        row_selectable='multi',
        selected_rows=[],
        filter_action='native', # Enable backend filtering
        sort_action='native',   # Enable backend sorting
        page_action='native',   # Enable pagination if needed
        page_size=20,           # Number of rows per page
        style_table={'overflowX': 'auto', 'maxHeight': '500px', 'overflowY': 'auto'},
        style_cell={'minWidth': '100px', 'width': '150px', 'maxWidth': '200px', 'textAlign': 'left'},
        # Conditional formatting for flags
        style_data_conditional=[
            {
                'if': {'column_id': 'auto_qc_flag', 'filter_query': '{auto_qc_flag} = 4'}, # Bad
                'backgroundColor': 'tomato', 'color': 'white'
            },
            {
                'if': {'column_id': 'auto_qc_flag', 'filter_query': '{auto_qc_flag} = 3'}, # Doubtful
                'backgroundColor': 'orange', 'color': 'white'
            },
            {
                'if': {'column_id': 'auto_qc_flag', 'filter_query': '{auto_qc_flag} = 7'}, # Investigate
                'backgroundColor': 'yellow', 'color': 'black'
            },
             {
                'if': {'column_id': 'auto_qc_flag', 'filter_query': '{auto_qc_flag} = 2'}, # Inconsistent
                'backgroundColor': 'lightgrey', 'color': 'black'
            },
             {
                'if': {'column_id': 'DIS_DETAIL_DATA_QC_CODE', 'filter_query': '{DIS_DETAIL_DATA_QC_CODE} != {auto_qc_flag}'}, # Manually changed
                'backgroundColor': 'lightblue', 'fontWeight': 'bold'
            },
        ]
    )

    end_time = datetime.datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    print(f"Visualizations and table update took {processing_time:.2f} seconds.")

    return map_fig, station_fig, ts_nut_fig, qc_comp_fig, html.Div(data_table), table_info_text


# --- Callback to Update Review Panel based on Table Selection ---
@callback(
    # Update the text shown above the controls
    Output('review-panel-content', 'children'),
    # Control the visibility of the entire review panel
    Output('review-panel', 'style', allow_duplicate=True),
    # Control the visibility of the controls container
    Output('review-controls-container', 'style', allow_duplicate=True),
    # Update the dropdown value
    Output('dropdown-final-flag', 'value'),
    # Update the textarea value (clear it)
    Output('textarea-sme-comment', 'value'),
    # You might also need to update dropdown options if they are dynamic
    Output('dropdown-final-flag', 'options'), # Add this Output
    Input('data-table', 'selected_rows'),
    State('data-table', 'data'),
    State('store-data', 'data'),
    prevent_initial_call=True
)
def update_review_panel(selected_rows, table_data, stored_data_dict):
    """
    Updates the review panel based on selected rows in the data table.

    This callback is triggered when a user selects rows in the data table. It:
    1. Shows details of the selected data point(s)
    2. Enables controls for changing the quality flag
    3. Provides a text area for adding comments about the change

    If a single row is selected, detailed information about that data point is shown.
    If multiple rows are selected, only the count of selected rows is shown.

    Args:
        selected_rows (list): Indices of the selected rows in the data table
        table_data (list): Data currently displayed in the table
        stored_data_dict (list): Complete dataset stored in the application

    Returns:
        tuple: Multiple outputs that update the review panel:
            - Content to display above the controls
            - Display settings for the review panel
            - Display settings for the controls container
            - Current flag value for the dropdown
            - Empty string for the comment text area
            - Options for the flag dropdown
    """
    if selected_rows is None or len(selected_rows) == 0 or stored_data_dict is None:
        # No rows selected or no data loaded:
        # Hide the entire panel
        # Hide the controls container
        # Reset dropdown/textarea values
        # Reset text above controls
        return ("Select rows in the table to review.",
                {'display': 'none'},  # Hide panel
                {'display': 'none'},  # Hide controls
                None, "", REVIEW_FLAG_OPTIONS)  # Reset values, set options

    df_full = pd.DataFrame(stored_data_dict)
    # Map selected row indices (from potentially filtered/paged table data) back to the full dataframe index
    # This is crucial if the table data ('data') is different from the full stored data
    # Note: `selected_rows` gives indices relative to the *current* `data` property of the DataTable
    # If using native filtering/sorting, `data` reflects the filtered/sorted state.
    # If only showing a subset (e.g., .head(MAX_TABLE_ROWS)), selected_rows refers to that subset.
    # We need the identifier (e.g., original index or DIS_SAMPLE_KEY_VALUE) from the selected rows
    # to reliably find them in the full dataframe stored in dcc.Store.

    # Assuming `table_data` corresponds directly to the rows displayed,
    # and we have a unique identifier like 'DIS_SAMPLE_KEY_VALUE' or we use the original index if preserved.
    # If 'DIS_SAMPLE_KEY_VALUE' is unique:
    # selected_keys = [table_data[i]['DIS_SAMPLE_KEY_VALUE'] for i in selected_rows]
    # selected_df_rows = df_full[df_full['DIS_SAMPLE_KEY_VALUE'].isin(selected_keys)]
    # Alternative using index if table data is just a filtered view of stored data:
    # This assumes the order in table_data matches the order in stored_data_dict *after filtering*
    # A safer way is often to include the original index or a unique ID in the table data itself.
    # For simplicity here, we'll try and use the indices directly, assuming they map correctly for now.
    # WARNING: This mapping might be fragile with native filtering/pagination if not handled carefully.
    # A robust solution often involves getting `derived_virtual_indices` or `derived_viewport_indices`.
    # Let's fetch data based on selected row indices from the potentially filtered `table_data`
    selected_row_data = [table_data[i] for i in selected_rows]
    num_selected = len(selected_row_data)

    if num_selected == 1:
        row = selected_row_data[0]
        # Format the details text (using <br> for new lines in html.Div)
        details_elements = [
            html.Strong("Selected Point Details:"), html.Br(),
            f"Event ID: {row.get('EVENT_COLLECTOR_EVENT_ID', 'N/A')}", html.Br(),
            f"Depth: {row.get('DIS_HEADER_START_DEPTH', 'N/A')}", html.Br(),
            f"Parameter: {row.get('DATA_TYPE_METHOD', 'N/A')}", html.Br(),
            f"Value: {row.get('DIS_DETAIL_DATA_VALUE', 'N/A')}", html.Br(),
            f"Auto Flag: {row.get('auto_qc_flag', 'N/A')} ({QC_FLAG_DEFINITIONS.get(row.get('auto_qc_flag'), 'N/A')})",
            html.Br(),
            f"Auto Details: {row.get('auto_qc_details_str', 'N/A')}", html.Br(),
            html.Strong(
                f"Current Final Flag: {row.get('DIS_DETAIL_DATA_QC_CODE', 'N/A')} ({QC_FLAG_DEFINITIONS.get(row.get('DIS_DETAIL_DATA_QC_CODE'), 'N/A')})"),
            html.Br(),
        ]
        current_final_flag = row.get('DIS_DETAIL_DATA_QC_CODE')
        details_content = html.Div(details_elements)  # Wrap in a div

    else:
        details_content = html.Strong(f"{num_selected} rows selected.")
        current_final_flag = None  # Don't pre-select flag for multi-edit

    # Enable review controls
    review_controls = html.Div([
         dcc.Dropdown(
             id='dropdown-final-flag',
             options=REVIEW_FLAG_OPTIONS,
             placeholder="Select New Final Flag",
             value=current_final_flag, # Pre-select if single row
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

    # Rows are selected:
    # Show the entire panel
    # Show the controls container
    # Update dropdown value and clear textarea
    return (details_content,  # Update text above controls
            {'display': 'block'},  # Show panel
            {'display': 'block'},  # Show controls
            current_final_flag, "", REVIEW_FLAG_OPTIONS)  # Update values, set options


# --- Callback to Handle Review Submission ---
@callback(
    Output('store-data', 'data', allow_duplicate=True),
    Output('review-status', 'children'),
    Output('data-table', 'selected_rows'), # Clear selection after submit
    Output('review-panel', 'style', allow_duplicate=True), # Hide panel after submit
    Output('review-controls-container', 'style', allow_duplicate=True), # Hide controls after submit
    # Note: We don't need to clear dropdown/textarea here, update_review_panel will handle it
    # when selected_rows becomes empty after clearing.
    Input('button-submit-review', 'n_clicks'),
    # Add State for selected_rows here, matching its position in the function arguments
    State('data-table', 'selected_rows'),
    # These are now States referring to components in the initial layout
    State('dropdown-final-flag', 'value'),
    State('textarea-sme-comment', 'value'),
    State('store-data', 'data'),
    State('store-sme-username', 'data'),
    State('store-filename', 'data'),
    State('data-table', 'data'),
    prevent_initial_call=True
)
def submit_review(n_clicks, selected_rows, new_flag, sme_comment, stored_data_dict, sme_username, filename, table_data):
    """
    Processes the submission of a manual quality control review.

    This callback is triggered when a user clicks the 'Submit Review' button. It:
    1. Validates the input (selected rows, new flag, comment)
    2. Updates the quality flags for the selected data points
    3. Logs the review action for traceability
    4. Updates the stored data with the new flags
    5. Provides feedback about the review submission

    Args:
        n_clicks (int): Number of times the submit button has been clicked
        selected_rows (list): Indices of the selected rows in the data table
        new_flag (int): The new quality flag value selected by the user
        sme_comment (str): Comment explaining the reason for the flag change
        stored_data_dict (list): Complete dataset stored in the application
        sme_username (str): Username of the person submitting the review
        filename (str): Name of the data file being reviewed
        table_data (list): Data currently displayed in the table

    Returns:
        tuple: Multiple outputs that update the UI after submission:
            - Updated data for storage
            - Status message about the submission
            - Empty list to clear selected rows
            - Display settings to hide the review panel
            - Display settings to hide the controls container
    """
    if n_clicks == 0 or selected_rows is None or len(selected_rows) == 0 or stored_data_dict is None:
        raise PreventUpdate

    # --- Input Validation ---
    if new_flag is None:
        return dash.no_update, html.Div("Error: Please select a final flag.", className="alert alert-warning"), dash.no_update, dash.no_update
    if not sme_comment or sme_comment.strip() == "":
         return dash.no_update, html.Div("Error: Please enter a comment.", className="alert alert-warning"), dash.no_update, dash.no_update

    start_time = datetime.datetime.now()
    print("Submitting review...")

    df_full = pd.DataFrame(stored_data_dict)

    # --- Identify Rows to Update in the Full DataFrame ---
    # Use the selected rows from the potentially filtered `table_data` to get unique identifiers
    # Assuming 'DIS_SAMPLE_KEY_VALUE' is the unique key, or fall back to index if needed.
    # This part is critical and depends on having a reliable way to link table rows to the main df.
    try:
        # Attempt to use DIS_SAMPLE_KEY_VALUE if it exists and is reliable
        if 'DIS_SAMPLE_KEY_VALUE' in df_full.columns and df_full['DIS_SAMPLE_KEY_VALUE'].is_unique:
            selected_keys = [table_data[i]['DIS_SAMPLE_KEY_VALUE'] for i in selected_rows]
            update_indices = df_full[df_full['DIS_SAMPLE_KEY_VALUE'].isin(selected_keys)].index
        else:
            # Fallback: Assume selected_rows directly index into df_full (Less Robust!)
            # This requires careful state management if table is heavily filtered/paged/sorted.
            # Consider adding the original index as a column to the table data if this fails.
            print("Warning: Using selected_rows index directly, may be unreliable with filtering/paging.")
            update_indices = selected_rows # Use selected_rows directly if they map 1:1 to df_full's index after filtering/sorting. Needs verification.
            # A better fallback might be needed depending on how data-table `data` is populated.

        if len(update_indices) == 0:
             raise ValueError("Could not map selected table rows to the main dataset.")

    except Exception as e:
        print(f"Error mapping selected rows: {e}")
        return dash.no_update, html.Div(f"Error mapping selected rows: {e}", className="alert alert-danger"), dash.no_update, dash.no_update

    # --- Apply Updates and Log Changes ---
    log_entries = []
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mission_descriptor = df_full['MISSION_DESCRIPTOR'].iloc[0] if not df_full.empty else "UnknownMission" # Get mission for log filename

    # Make a copy to modify - important for Pandas >= 1.5 CoW
    df_updated = df_full.copy()

    for idx in update_indices:
        try:
            # Ensure index exists in the copied dataframe
            if idx not in df_updated.index:
                print(f"Warning: Index {idx} not found in DataFrame during update, skipping.")
                continue

            original_auto_flag = df_updated.loc[idx, 'auto_qc_flag']
            # Use a unique identifier for logging if available
            data_point_id = df_updated.loc[idx, 'DIS_SAMPLE_KEY_VALUE'] if 'DIS_SAMPLE_KEY_VALUE' in df_updated.columns else f"Index_{idx}"

            # Update the final QC code
            df_updated.loc[idx, 'DIS_DETAIL_DATA_QC_CODE'] = new_flag

            # Create log entry
            log_data = {
                'Timestamp': timestamp,
                'SME_Username': sme_username if sme_username else 'Not Provided',
                'DataPoint_Identifier': data_point_id,
                'Original_Auto_Flag': original_auto_flag,
                'Assigned_Final_Flag': new_flag,
                'SME_Comment': sme_comment
            }
            # Append log entry to file
            append_to_log(log_data, mission_descriptor)

        except KeyError as ke:
            print(f"KeyError accessing index {idx} during review update: {ke}")
            # Decide whether to continue or raise an error
            continue # Skip this row and continue with others
        except Exception as e:
            print(f"Error processing review for index {idx}: {e}")
            # Decide whether to continue or raise an error
            return dash.no_update, html.Div(f"Error processing review update for index {idx}: {e}", className="alert alert-danger"), dash.no_update, dash.no_update

    end_time = datetime.datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    print(f"Review submission processed {len(update_indices)} rows in {processing_time:.2f} seconds.")

    # Return updated data, success message, clear selection, HIDE panel and controls
    return (df_updated.to_dict('records'),
            html.Div(f"Successfully updated {len(update_indices)} records.", className="alert alert-success"),
            [],  # Clear selected rows (this will trigger update_review_panel to hide the panel)
            {'display': 'none'},  # Explicitly hide panel here too for good measure
            {'display': 'none'})  # Explicitly hide controls container here too


# --- Callback for Downloading Updated CSV ---
@callback(
    Output('download-csv', 'data'),
    Output('review-status', 'children', allow_duplicate=True), # For error messages
    Input('button-download-csv', 'n_clicks'),
    State('store-data', 'data'),
    State('store-filename', 'data'),
    prevent_initial_call=True
)
def download_csv(n_clicks, stored_data_dict, filename):
    """
    Prepares and downloads the quality-controlled data as a CSV file.

    This callback is triggered when a user clicks the 'Download Updated CSV' button. It:
    1. Checks if all data points have been properly reviewed (no flag 7 remaining)
    2. Formats the data for output with all original columns in the correct order
    3. Generates a CSV file with the updated quality flags
    4. Provides the file for download

    Args:
        n_clicks (int): Number of times the download button has been clicked
        stored_data_dict (list): Complete dataset stored in the application
        filename (str): Name of the original data file

    Returns:
        tuple: (download data, status message)
            - If successful: (CSV file data, success message)
            - If failed: (None, error message)
    """
    if n_clicks == 0 or stored_data_dict is None:
        raise PreventUpdate

    df = pd.DataFrame(stored_data_dict)
    output_filename = f"{Path(filename).stem}_QC_updated.csv" if filename else "qc_output.csv"

    # --- Final Check: Ensure no Flag 7 remains ---
    if 7 in df['DIS_DETAIL_DATA_QC_CODE'].unique():
        count_flag_7 = (df['DIS_DETAIL_DATA_QC_CODE'] == 7).sum()
        error_msg = f"Error: Cannot download. {count_flag_7} records still have flag 7 (Requires Investigation). Please review these points before exporting."
        return None, html.Div(error_msg, className="alert alert-danger")

    try:
        # Ensure columns are in the original order
        output_df = format_dataframe_for_output(df.copy()) # Use copy

        # Generate CSV content
        csv_string = output_df.to_csv(index=False, encoding='utf-8')

        return dcc.send_data_frame(output_df.to_csv, filename=output_filename, index=False), html.Div(f"Prepared {output_filename} for download.", className="alert alert-info")

    except Exception as e:
        print(f"Error during CSV generation: {e}")
        return None, html.Div(f"Error generating CSV: {e}", className="alert alert-danger")


# --- Callback for Downloading Summary Report ---
@callback(
    Output('download-report', 'data'),
    Input('button-download-report', 'n_clicks'),
    State('store-data', 'data'),
    State('store-filename', 'data'),
    State('store-sme-username', 'data'),
    prevent_initial_call=True
)
def download_report(n_clicks, stored_data_dict, filename, sme_username):
    """
    Generates and downloads a summary report of the quality control process.

    This callback is triggered when a user clicks the 'Download Summary Report' button. It:
    1. Creates a text report summarizing the quality control process and results
    2. Includes statistics about automatic and manual quality flags
    3. Provides the report as a downloadable text file

    Args:
        n_clicks (int): Number of times the download button has been clicked
        stored_data_dict (list): Complete dataset stored in the application
        filename (str): Name of the original data file
        sme_username (str): Username of the person who reviewed the data

    Returns:
        dict or None: Dictionary with report content and filename for download,
                     or None if an error occurs (with PreventUpdate)
    """
    if n_clicks == 0 or stored_data_dict is None:
        raise PreventUpdate

    df = pd.DataFrame(stored_data_dict)
    report_filename = f"{Path(filename).stem}_QC_summary.txt" if filename else "qc_summary.txt"

    try:
        report_content = generate_summary_report(df.copy(), filename, sme_username)

        # Remove the trailing comma
        return dict(content=report_content, filename=report_filename)

    except Exception as e:
        print(f"Error during Summary Report generation: {e}")
        # Optionally display error to user via review-status or another component
        raise PreventUpdate # Or return an error message in a Div


# --- Placeholder Callback for linking Map Clicks to Table ---
@callback(
    Output('data-table', 'filter_query'),
    Input('map-graph', 'clickData'),
    prevent_initial_call=True
)
def filter_table_on_map_click(clickData):
    """
    Filters the data table based on a point clicked on the map.

    This callback is triggered when a user clicks on a point in the map. It:
    1. Extracts the latitude and longitude of the clicked point
    2. Creates a filter query to show only data points at that location
    3. Applies the filter to the data table

    Note: This is a placeholder implementation that is currently disabled.
    A more robust implementation would be needed for production use.

    Args:
        clickData (dict): Data about the clicked point on the map

    Returns:
        str or None: Filter query string for the data table,
                    or None with PreventUpdate if filtering is not implemented
    """
    if clickData is None:
        raise PreventUpdate

    # Extract information from clickData (e.g., station ID or coordinates)
    point_info = clickData['points'][0]
    # Example: filter by event ID if available in customdata or hover text
    # This depends on how hover/custom data was set up in the map figure
    # event_id = point_info.get('customdata', [None])[0] # If stored in customdata
    # A simpler approach might be to filter by lat/lon, but requires precise matching
    lat = point_info['lat']
    lon = point_info['lon']

    # Construct a filter query for dash_table
    # Be cautious with floating point comparisons - use ranges if needed
    filter_query = f"{{DIS_HEADER_SLAT}} = {lat} && {{DIS_HEADER_SLON}} = {lon}"
    # Example using event ID if parsed from hover text (more robust)
    # hover_text = point_info.get('hovertext', '')
    # match = re.search(r"Event ID: (\S+)", hover_text)
    # if match:
    #      event_id = match.group(1)
    #      filter_query = f"{{EVENT_COLLECTOR_EVENT_ID}} = {event_id}" # Careful if Event ID is numeric vs string

    print(f"Map click filter query: {filter_query}")
    # return filter_query
    print("Map click filtering needs robust implementation based on available point data.")
    raise PreventUpdate # Disable until robust filtering is implemented
