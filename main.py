from fastapi import FastAPI, File, UploadFile, HTTPException
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

# Global models
whisper_model = None
diarize_pipeline = None

def initialize_models():
    global whisper_model, diarize_pipeline
    try:
        logger.info("Loading WhisperX model...")
        whisper_model = whisperx.load_model("large-v2", DEVICE, compute_type=COMPUTE_TYPE)
        
        logger.info("Loading diarization pipeline...")
        diarize_pipeline = whisperx.DiarizationPipeline(
            use_auth_token=HF_TOKEN,
            device=DEVICE
        )
        
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Failed to load models: {str(e)}")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    if not torch.cuda.is_available():
        raise HTTPException(status_code=500, detail="CUDA GPU is required but not available")

    logger.info(f"Received file: {file.filename}")
    temp_file = TEMP_DIR / f"temp_{file.filename}"
    
    try:
        # Save uploaded file
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved to {temp_file}")

        # Initialize models if not already done
        if whisper_model is None or diarize_pipeline is None:
            initialize_models()

        # Load audio
        logger.info("Loading audio...")
        audio = whisperx.load_audio(str(temp_file))
        
        # Run initial transcription
        logger.info("Running initial transcription...")
        result = whisper_model.transcribe(audio, batch_size=BATCH_SIZE)
        language = result["language"]
        logger.info(f"Detected language: {language}")

        # Load alignment model and align
        logger.info("Loading alignment model...")
        model_a, metadata = whisperx.load_align_model(
            language_code=language,
            device=DEVICE
        )
        
        logger.info("Aligning transcription...")
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            DEVICE,
            return_char_alignments=False
        )

        # Run diarization
        logger.info("Running diarization...")
        diarize_segments = diarize_pipeline(
            audio
        )
        logger.info(f"Diarization complete. Found segments: {diarize_segments}")

        # Assign speaker labels
        logger.info("Assigning speaker labels...")
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

        logger.info(f"Processing complete. Found {len(response_data['transcription'])} segments")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup
        try:
            if temp_file.exists():
                temp_file.unlink()
                logger.info(f"Cleaned up {temp_file}")
        except Exception as e:
            logger.error(f"Error cleaning up: {str(e)}")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up application...")
    try:
        # Preload models
        initialize_models()
        
        # Log CUDA info
        if torch.cuda.is_available():
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MB")
            logger.info(f"CUDA memory cached: {torch.cuda.memory_reserved(0)/(1024**2):.2f} MB")
        else:
            logger.warning("CUDA is not available!")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )