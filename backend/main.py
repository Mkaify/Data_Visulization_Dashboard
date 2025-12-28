from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict
import pandas as pd
import io
import uuid

app = FastAPI(title="Data Viz Dashboard Pro")

# --- MULTI-USER STORAGE ---
# Stores data as { "session_id": {"df": DataFrame, "filename": str} }
sessions: Dict[str, dict] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODELS ---
class BaseSessionRequest(BaseModel):
    session_id: str

class CleaningRequest(BaseSessionRequest):
    method: str
    column: Optional[str] = None
    fill_value: Optional[str] = None

class PlotRequest(BaseSessionRequest):
    x_axis: str
    y_axis: Optional[str] = None
    chart_type: str

class FilterRequest(BaseSessionRequest):
    column: str
    operation: str
    value: str

# --- HELPERS ---
def get_session_data(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session expired or not found. Please re-upload.")
    return sessions[session_id]

def format_response(session_id: str, df: pd.DataFrame):
    return {
        "session_id": session_id,
        "filename": sessions[session_id]["filename"],
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "preview": df.head().fillna("").to_dict(orient="records")
    }

# --- ENDPOINTS ---

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed.")
    try:
        session_id = str(uuid.uuid4())
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        sessions[session_id] = {"df": df, "filename": file.filename}
        return format_response(session_id, df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clean")
def clean_data(request: CleaningRequest):
    session = get_session_data(request.session_id)
    df = session["df"]

    if request.method == "dropna":
        if request.column and request.column != "all":
            df = df.dropna(subset=[request.column])
        else:
            df = df.dropna()
    elif request.method == "fillna":
        if not request.fill_value:
             raise HTTPException(status_code=400, detail="Fill value required.")
        if request.column and request.column != "all":
            df[request.column] = df[request.column].fillna(request.fill_value)
        else:
            df = df.fillna(request.fill_value)
    
    sessions[request.session_id]["df"] = df.reset_index(drop=True)
    return format_response(request.session_id, sessions[request.session_id]["df"])

@app.post("/plot")
def generate_plot(request: PlotRequest):
    session = get_session_data(request.session_id)
    df = session["df"]
    
    try:
        if request.chart_type in ["bar", "line", "pie"]:
            if request.y_axis:
                grouped = df.groupby(request.x_axis)[request.y_axis].sum().reset_index()
                return {
                    "labels": grouped[request.x_axis].astype(str).tolist(),
                    "values": grouped[request.y_axis].tolist(),
                    "label": f"Sum of {request.y_axis}"
                }
            else:
                counts = df[request.x_axis].value_counts()
                return {
                    "labels": counts.index.astype(str).tolist(),
                    "values": counts.values.tolist(),
                    "label": "Frequency"
                }
        elif request.chart_type == "scatter":
            return {
                "labels": df[request.x_axis].tolist(),
                "values": df[request.y_axis].tolist(),
                "label": f"{request.y_axis} vs {request.x_axis}"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plotting error: {str(e)}")

@app.get("/stats/{session_id}")
def get_stats(session_id: str):
    session = get_session_data(session_id)
    numeric_df = session["df"].select_dtypes(include=['number'])
    if numeric_df.empty:
        return []
    stats = numeric_df.describe().T.reset_index()
    stats.columns = ["Column", "Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    return stats.to_dict(orient="records")

@app.post("/filter")
def filter_data(request: FilterRequest):
    session = get_session_data(request.session_id)
    df = session["df"]
    try:
        col, val = request.column, request.value
        if pd.api.types.is_numeric_dtype(df[col]):
            val = float(val)

        if request.operation == "gt": df = df[df[col] > val]
        elif request.operation == "lt": df = df[df[col] < val]
        elif request.operation == "eq": df = df[df[col] == val]
        elif request.operation == "contains": df = df[df[col].astype(str).str.contains(str(val), case=False)]
        
        sessions[request.session_id]["df"] = df.reset_index(drop=True)
        return format_response(request.session_id, sessions[request.session_id]["df"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{session_id}")
def download(session_id: str):
    session = get_session_data(session_id)
    stream = io.StringIO()
    session["df"].to_csv(stream, index=False)
    return StreamingResponse(iter([stream.getvalue()]), media_type="text/csv", 
                             headers={"Content-Disposition": f"attachment; filename=cleaned_{session['filename']}"})

# --- FRONTEND SERVING ---
# Ensure your 'frontend' folder is in the same directory as main.py
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/")
async def read_index():
    return FileResponse('frontend/index.html')