import os
import tempfile
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import DiseaseEngine package
try:
    from app.inference import DiseaseEngine
except ImportError:
    from inference import DiseaseEngine

app = FastAPI(
    title="AgroIntelli AI Engine",
    description="Backend API for Offline Plant Pathology Inference",
    version="1.0.0"
)

# Enable CORS for local testing if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize deep learning inference engine
engine = DiseaseEngine()

@app.post("/predict")
async def predict_leaf(
    file: UploadFile = File(...),
    field_mode: bool = Query(True, description="Enable Field Mode confidence thresholding")
):
    # Determine file extension to prevent Pillow issues
    _, ext = os.path.splitext(file.filename or "")
    if not ext:
        ext = ".jpg"
        
    # Write incoming file to temporary location
    # Windows requires closing file descriptor before opening it elsewhere (like in PIL)
    fd, temp_path = tempfile.mkstemp(suffix=ext)
    try:
        with os.fdopen(fd, "wb") as tmp:
            content = await file.read()
            tmp.write(content)
            
        # Perform image validation, severity proxy, and neural inference
        result = engine.predict(temp_path, field_mode=field_mode)
        return result
        
    except Exception as e:
        return {"ok": False, "error": str(e)}
        
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

# Mount static files at root directory
# html=True automatically serves index.html at "/"
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    print("Starting AgroIntelli Web Server on http://localhost:8000 ...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
