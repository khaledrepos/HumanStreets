from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
import torch
import time
from threading import Thread
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager, contextmanager
import asyncio

# Global variables
model = None
tokenizer = None
model_id = "LiquidAI/LFM2-1.2B"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print(f"Loading Base Model: {model_id}...")
    try:
        # 1. Load Base Model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype="bfloat16",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # 2. Load Fine-Tuned LoRA Adapter
        # Path where the fine-tuned checkpoint is saved
        adapter_path = r"V:\MICS\Projects___IN_PROGRESS\DevPorj\BootCamp_Capstone_Project\idea1_walkabilityScoring\cloned\HumanStreets\fineTuning_PEFT\saudi-lora-test\checkpoint-113"
        
        print(f"Loading LoRA Adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        print("Model & Adapter loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    global model, tokenizer
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    try:
        prompt = request.message
        # Using chat template but simpler handling for demo
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        ).to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            do_sample=True,
            temperature=0.3,
            min_p=0.15,
            repetition_penalty=1.05,
            max_new_tokens=512,
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        async def response_generator():
            for text in streamer:
                yield text
                await asyncio.sleep(0.01) # Small sleep to yield control

        return StreamingResponse(response_generator(), media_type="text/plain")

    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

import psycopg2
import json

# ... (Previous code)

@contextmanager
def get_db_connection():
    """Yields a database connection context."""
    conn = None
    try:
        conn = psycopg2.connect(
            dbname="streets_eval",
            user="postgres",
            password="12345",
            host="localhost",
            port="5432"
        )
        yield conn
    except Exception as e:
        print(f"DB Connection Error: {e}")
        raise e
    finally:
        if conn:
            conn.close()

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/segmentations")
def get_segmentations():
    """Fetch all segmentation polygons as GeoJSON"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Query to get GeoJSON features directly
                query = """
                    SELECT json_build_object(
                        'type', 'FeatureCollection',
                        'features', json_agg(json_build_object(
                            'type', 'Feature',
                            'geometry', ST_AsGeoJSON(geom)::json,
                            'properties', json_build_object('class_id', class_id)
                        ))
                    )
                    FROM segmentations;
                """
                
                cur.execute(query)
                result = cur.fetchone()[0]
                
                if result is None:
                     return {"type": "FeatureCollection", "features": []}
                     
                return result
        
    except Exception as e:
        print(f"DB Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/segmentations/centroid")
def get_segmentations_centroid():
    """Fetch the centroid of all segmentation polygons"""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Calculate centroid of the union of all geometries
                query = """
                    SELECT ST_X(ST_Centroid(ST_Collect(geom)))::float as lon,
                           ST_Y(ST_Centroid(ST_Collect(geom)))::float as lat
                    FROM segmentations;
                """
                
                cur.execute(query)
                result = cur.fetchone()
                
                if result is None or result[0] is None:
                     raise HTTPException(status_code=404, detail="No segmentations found")
                     
                return {"longitude": result[0], "latitude": result[1]}
        
    except Exception as e:
        print(f"DB Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
