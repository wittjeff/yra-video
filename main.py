from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import whisperx
import torch
import shutil
import logging
import traceback
from pathlib import Path
import os 
from dotenv import load_dotenv
import uuid
from pydantic import BaseModel
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is not set in the environment!")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16"
BATCH_SIZE = 8  # Reduced batch size for stability
TEMP_DIR = Path("temp_files")
TEMP_DIR.mkdir(exist_ok=True)
MAX_CONCURRENT_TASKS = 2  # Limiting concurrent tasks to improve stability

# Task queue for sequential processing
task_queue = queue.Queue()

# Thread pool for concurrent processing
thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_TASKS)

# Model locks to prevent concurrent access to models
model_lock = threading.Lock()

# Task status tracking
class TaskStatus(BaseModel):
    id: str
    status: str
    progress: Optional[float] = 0.0
    result: Optional[Dict] = None
    error: Optional[str] = None

# Store for all tasks
tasks: Dict[str, TaskStatus] = {}

def load_models():
    """Load WhisperX models with proper error handling"""
    try:
        # Ensure we're using GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            logger.info(f"Loading WhisperX model on GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available! Loading models on CPU, which will be extremely slow")
            
        whisper_model = whisperx.load_model("large-v2", device, compute_type=COMPUTE_TYPE)
        
        logger.info("Loading diarization pipeline...")
        diarize_pipeline = whisperx.DiarizationPipeline(
            use_auth_token=HF_TOKEN,
            device=device
        )
        
        logger.info(f"Models loaded successfully on {device}")
        return whisper_model, diarize_pipeline
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to load models: {str(e)}")

def process_transcription(task_id: str, file_path: Path):
    """Process transcription with individual model instances and save results"""
    try:
        # Update task status
        tasks[task_id].status = "processing"
        tasks[task_id].progress = 0.1
        
        # Load individual model instances for this task
        logger.info(f"[Task {task_id}] Loading models...")
        whisper_model, diarize_pipeline = load_models()
        tasks[task_id].progress = 0.15
            
        # Load audio
        logger.info(f"[Task {task_id}] Loading audio...")
        audio = whisperx.load_audio(str(file_path))
        tasks[task_id].progress = 0.2
        
        # Run initial transcription
        logger.info(f"[Task {task_id}] Running initial transcription...")
        result = whisper_model.transcribe(audio, batch_size=BATCH_SIZE)
        language = result["language"]
        logger.info(f"[Task {task_id}] Detected language: {language}")
        tasks[task_id].progress = 0.5
        
        # Load alignment model and align
        logger.info(f"[Task {task_id}] Loading alignment model...")
        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=DEVICE
        )
        
        logger.info(f"[Task {task_id}] Aligning transcription...")
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            DEVICE,
            return_char_alignments=False
        )
        tasks[task_id].progress = 0.7
        
        # Run diarization
        logger.info(f"[Task {task_id}] Running diarization...")
        diarize_segments = diarize_pipeline(
            audio
        )
        logger.info(f"[Task {task_id}] Diarization complete. Found segments: {diarize_segments}")
        tasks[task_id].progress = 0.9
        
        # Assign speaker labels
        logger.info(f"[Task {task_id}] Assigning speaker labels...")
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # Format response with proper speaker labels
        response_data = {
            "transcription": [
                {
                    "start": format(segment["start"], ".2f"),
                    "end": format(segment["end"], ".2f"),
                    "speaker": f"Speaker {segment.get('speaker', 'Unknown')}".replace("SPEAKER_", ""),
                    "text": segment["text"].strip()
                }
                for segment in result["segments"]
            ],
            "language": language
        }
        
        # Create results directory if it doesn't exist
        results_dir = Path("transcription_results")
        results_dir.mkdir(exist_ok=True)
        
        # Save transcription results with unique filename
        result_filename = results_dir / f"{task_id}.json"
        
        import json
        with open(result_filename, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"[Task {task_id}] Saved transcription results to {result_filename}")
        
        # Add file path to response data
        response_data["saved_file"] = str(result_filename)
        
        logger.info(f"[Task {task_id}] Processing complete. Found {len(response_data['transcription'])} segments")
        
        # Update task with results
        tasks[task_id].status = "completed"
        tasks[task_id].progress = 1.0
        tasks[task_id].result = response_data
        
        # Clean up model references to free CUDA memory
        del whisper_model
        del diarize_pipeline
        del model_a
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"[Task {task_id}] Error processing audio: {str(e)}")
        logger.error(traceback.format_exc())
        tasks[task_id].status = "failed"
        tasks[task_id].error = str(e)
    
    finally:
        # Cleanup temporary audio file
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"[Task {task_id}] Cleaned up {file_path}")
        except Exception as e:
            logger.error(f"[Task {task_id}] Error cleaning up: {str(e)}")

def task_worker():
    """Worker that processes tasks sequentially from the queue"""
    while True:
        task = task_queue.get()
        if task is None:
            break
            
        task_id, file_path = task
        try:
            process_transcription(task_id, file_path)
        except Exception as e:
            logger.error(f"Error in task worker: {str(e)}")
        finally:
            task_queue.task_done()

@app.post("/transcribe/")
async def transcribe_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not torch.cuda.is_available():
        logger.error("CUDA GPU is required but not available")
        raise HTTPException(status_code=500, detail="CUDA GPU is required but not available")

    # Check current CUDA memory usage
    if torch.cuda.is_available():
        current_mem = torch.cuda.memory_allocated(0)/(1024**3)
        total_mem = torch.cuda.get_device_properties(0).total_memory/(1024**3)
        logger.info(f"Current CUDA memory usage: {current_mem:.2f}GB / {total_mem:.2f}GB")
        
        # Generate a unique task ID
        task_id = str(uuid.uuid4())
        logger.info(f"Created new task {task_id} for file: {file.filename}")
        
        # Create temporary file path
        temp_file = TEMP_DIR / f"{task_id}_{file.filename}"
        
        try:
            # Save uploaded file
            with open(temp_file, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"File saved to {temp_file}")
            
            # Initialize task status
            status = "queued"
            message = "Transcription queued for processing"
            
            # If memory usage is high, mark as waiting
            if current_mem > total_mem * 0.15:
                status = "waiting"
                message = "Server is currently at capacity. Please wait."
                logger.warning(f"Task {task_id} queued but waiting due to high memory usage")
            
            tasks[task_id] = TaskStatus(
                id=task_id,
                status=status,
                progress=0.0
            )
            
            # Add task to the queue (will be processed in sequence)
            task_queue.put((task_id, temp_file))
            
            return JSONResponse(
                status_code=202,
                content={
                    "task_id": task_id, 
                    "status": status, 
                    "message": message
                }
            )
        
        except Exception as e:
            logger.error(f"Error submitting task: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Clean up
            if temp_file.exists():
                temp_file.unlink()
                
            raise HTTPException(status_code=500, detail=str(e))


    if not torch.cuda.is_available():
        logger.error("CUDA GPU is required but not available")
        raise HTTPException(status_code=500, detail="CUDA GPU is required but not available")

    # Check current CUDA memory usage
    if torch.cuda.is_available():
        current_mem = torch.cuda.memory_allocated(0)/(1024**3)
        total_mem = torch.cuda.get_device_properties(0).total_memory/(1024**3)
        logger.info(f"Current CUDA memory usage: {current_mem:.2f}GB / {total_mem:.2f}GB")
        
        # Prevent overloading GPU
        if current_mem > total_mem * 0.8:
            raise HTTPException(
                status_code=503, 
                detail="Server is currently at capacity. Please try again later."
            )

    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    logger.info(f"Created new task {task_id} for file: {file.filename}")
    
    # Create temporary file path
    temp_file = TEMP_DIR / f"{task_id}_{file.filename}"
    
    try:
        # Save uploaded file
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved to {temp_file}")
        
        # Initialize task status
        tasks[task_id] = TaskStatus(
            id=task_id,
            status="queued",
            progress=0.0
        )
        
        # Add task to the queue (will be processed in sequence)
        task_queue.put((task_id, temp_file))
        
        return JSONResponse(
            status_code=202,
            content={"task_id": task_id, "status": "queued", "message": "Transcription queued for processing"}
        )
    
    except Exception as e:
        logger.error(f"Error submitting task: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Clean up
        if temp_file.exists():
            temp_file.unlink()
            
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    task = tasks[task_id]
    response = {
        "id": task.id,
        "status": task.status,
        "progress": task.progress
    }
    
    if task.status == "completed":
        response["result"] = task.result
    elif task.status == "failed":
        response["error"] = task.error
        
    return JSONResponse(content=response)

@app.get("/tasks/")
async def list_tasks():
    return JSONResponse(content={
        "tasks": [
            {
                "id": task_id,
                "status": task.status,
                "progress": task.progress
            } for task_id, task in tasks.items()
        ]
    })

@app.get("/system/stats")
async def get_system_stats():
    """Get current system statistics"""
    if torch.cuda.is_available():
        return JSONResponse(content={
            "gpu": {
                "name": torch.cuda.get_device_name(0),
                "memory_used_gb": torch.cuda.memory_allocated(0)/(1024**3),
                "memory_total_gb": torch.cuda.get_device_properties(0).total_memory/(1024**3),
                "utilization": torch.cuda.utilization(0)
            },
            "queue_size": task_queue.qsize(),
            "active_tasks": len([t for t in tasks.values() if t.status == "processing"]),
            "completed_tasks": len([t for t in tasks.values() if t.status == "completed"]),
            "failed_tasks": len([t for t in tasks.values() if t.status == "failed"])
        })
    else:
        return JSONResponse(content={
            "gpu": "Not available",
            "queue_size": task_queue.qsize(),
            "active_tasks": len([t for t in tasks.values() if t.status == "processing"])
        })

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up application...")
    try:
        # Log CUDA info
        if torch.cuda.is_available():
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB")
            logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved(0)/(1024**2):.2f} MB")
        else:
            logger.warning("CUDA is not available!")
            
        # Start worker threads for processing tasks
        for _ in range(MAX_CONCURRENT_TASKS):
            thread = threading.Thread(target=task_worker, daemon=True)
            thread.start()
            
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        logger.error(traceback.format_exc())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down application...")
    
    # Signal worker threads to exit
    for _ in range(MAX_CONCURRENT_TASKS):
        task_queue.put(None)
        
    # Wait for all tasks to complete
    task_queue.join()
    thread_pool.shutdown(wait=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )