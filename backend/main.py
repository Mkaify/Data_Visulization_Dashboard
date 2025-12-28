from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from fastapi.responses import StreamingResponse
import io
from fastapi.responses import JSONResponse

app = FastAPI(title="Data Viz Dashboard API")

# --- Global Storage (Simple In-Memory) ---
# In production, use Redis or a Database
data_store = {
    "df": None,        # Holds the Pandas DataFrame
    "filename": None
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
# This defines the structure of the JSON sent from Frontend
class CleaningRequest(BaseModel):
    method: str          # "dropna" or "fillna"
    column: Optional[str] = None
    fill_value: Optional[str] = None # For filling specific values

# --- New Pydantic Model for Plotting ---
class PlotRequest(BaseModel):
    x_axis: str
    y_axis: Optional[str] = None
    chart_type: str  # "bar", "line", "scatter" or "pie"

class FilterRequest(BaseModel):
    column: str
    operation: str  # "gt" (>), "lt" (<), "eq" (==), "contains"
    value: str      # We'll parse this to float/int if possible

def get_df_info(df):
    """Helper to format response"""
    return {
        "filename": data_store["filename"],
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(), # Useful for UI
        "preview": df.head().fillna("").to_dict(orient="records")
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Save to global store
        data_store["df"] = df
        data_store["filename"] = file.filename

        return get_df_info(df)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clean")
def clean_data(request: CleaningRequest):
    """
    Applies data cleaning operations on the stored DataFrame.
    """
    if data_store["df"] is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")

    df = data_store["df"]

    # 1. Drop Missing Values
    if request.method == "dropna":
        if request.column and request.column != "all":
            # Drop rows where specific column is missing
            df.dropna(subset=[request.column], inplace=True)
        else:
            # Drop rows if ANY column is missing
            df.dropna(inplace=True)

    # 2. Fill Missing Values
    elif request.method == "fillna":
        if not request.fill_value:
             raise HTTPException(status_code=400, detail="Fill value required.")
        
        if request.column and request.column != "all":
            df[request.column].fillna(request.fill_value, inplace=True)
        else:
            df.fillna(request.fill_value, inplace=True)
    
    # Save the cleaned version back to store
    data_store["df"] = df

    return get_df_info(df)

@app.post("/plot")
def generate_plot_data(request: PlotRequest):
    """
    Prepares data for visualization.
    Auto-aggregates data if there are duplicate X values.
    """
    if data_store["df"] is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")

    df = data_store["df"].copy()
    
    # Check if columns exist
    if request.x_axis not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column {request.x_axis} not found.")

    data = {}

    try:
        # Scenario 1: Categorical Bar/Line Chart (Aggregation needed)
        # Example: X="Country", Y="Sales". We need Sum of Sales per Country.
        if request.y_axis and request.chart_type in ["bar", "line", "pie"]:
            if request.y_axis not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column {request.y_axis} not found.")
            
            # Group by X and Sum Y (You can change 'sum' to 'mean' later)
            # We convert everything to strings for X-axis labels
            grouped = df.groupby(request.x_axis)[request.y_axis].sum().reset_index()
            
            data = {
                "labels": grouped[request.x_axis].astype(str).tolist(),
                "values": grouped[request.y_axis].tolist(),
                "label": f"Sum of {request.y_axis} by {request.x_axis}"
            }

        # Scenario 2: Scatter Plot (No Aggregation, Raw Data)
        elif request.chart_type == "scatter" and request.y_axis:
             data = {
                "labels": df[request.x_axis].tolist(), # X coordinates
                "values": df[request.y_axis].tolist(), # Y coordinates
                "label": f"{request.y_axis} vs {request.x_axis}"
            }
            
        # Scenario 3: Histogram / Count Plot (Only X provided)
        # Example: X="Category". We count how many times each category appears.
        elif not request.y_axis:
            counts = df[request.x_axis].value_counts()
            data = {
                "labels": counts.index.astype(str).tolist(),
                "values": counts.values.tolist(),
                "label": f"Count of {request.x_axis}"
            }

        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plotting error: {str(e)}")

# --- Feature 1: Statistical Summary ---
@app.get("/stats")
def get_stats():
    if data_store["df"] is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")
    
    df = data_store["df"]
    # Select only numerical columns for stats
    numeric_df = df.select_dtypes(include=['number'])
    
    if numeric_df.empty:
        return {"message": "No numerical columns found."}

    # Generate descriptive statistics (Mean, Std, Min, Max, etc.)
    stats = numeric_df.describe().T.reset_index()
    stats.columns = ["Column", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    
    return stats.to_dict(orient="records")

# --- Feature 2: Advanced Filtering ---
@app.post("/filter")
def filter_data(request: FilterRequest):
    if data_store["df"] is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")

    df = data_store["df"]
    
    try:
        column = request.column
        val = request.value
        
        # Attempt to convert value to number if the column is numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            val = float(val)

        if request.operation == "gt":    # Greater than
            df = df[df[column] > val]
        elif request.operation == "lt":  # Less than
            df = df[df[column] < val]
        elif request.operation == "eq":  # Equals
            df = df[df[column] == val]
        elif request.operation == "contains": # String contains
            df = df[df[column].astype(str).str.contains(val, case=False, na=False)]
            
        data_store["df"] = df  # Update global state
        return get_df_info(df)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Filtering error: {str(e)}")

# --- Feature 3: Download Processed Data ---
@app.get("/download")
def download_data():
    if data_store["df"] is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded.")
    
    df = data_store["df"]
    
    # Convert DataFrame to CSV in-memory
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=processed_data.csv"
    return response

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Serve the frontend folder
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def read_index():
    return FileResponse('frontend/index.html')