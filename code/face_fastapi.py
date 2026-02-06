"""
Face Recognition Service with FastAPI.

This implementation loads local ONNX/OM models and serves them via FastAPI.
"""

import os
import cv2
import time
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
import logging 
from face_model import FaceModelONNX
from face_ais_bench import FaceModelAISBench

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition Service",
    description="Implementation using local ONNX/OM models",
    version="1.0"
)

# Initialize face model implementation
model = None
model_type=os.environ.get("MODEL_TYPE", "onnx")

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global model
    try:
        logger.info("Model type: %s", model_type)
        if model_type=="om":
            model = FaceModelAISBench()
        else:
            model = FaceModelONNX()
        #model = FaceModel()
        logger.info(f"Finish loading face model. model type:{model_type}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        print("\n" + "="*60)
        print(f"ERROR: Failed to initialize face model: {e}")
        print("Please check your model dependencies.")
        print("You may need to install additional dependencies:")
        print("pip install onnxruntime-gpu")
        print("="*60 + "\n")


@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    """Serve the upload form HTML"""
    # Simple inline HTML form
    return HTMLResponse(
        """
        <html>
            <head>
                <title>Face Recognition Service</title>
            </head>
            <body>
                <h1>Face Recognition Service</h1>
                <form action="/" method="post" enctype="multipart/form-data">
                    <input type="file" name="file" accept="image/*" required>
                    <br><br>
                    <input type="checkbox" name="gender" value="1"> Gender
                    <input type="checkbox" name="age" value="1"> Age
                    <input type="checkbox" name="race" value="1"> Race
                    <br><br>
                    <input type="submit" value="Submit">
                </form>
            </body>
        </html>
        """
    )


@app.post("/")
async def process_image(
    file: UploadFile = File(...),
    gender: str = Form(None),
    age: str = Form(None),
    race: str = Form(None)
):
    """Process uploaded image and return face recognition results"""
    total_time_start = time.time()
    logger.info("--------------- START ---------------")
    timer = True
    logger.info("Request filename=%s content_type=%s", file.filename, file.content_type)

    # Check if model is initialized
    if model is None:
        logger.error("Model not initialized")
        return JSONResponse(
            status_code=503,
            content={"error": "Model not initialized"}
        )

    # Read and decode image
    try:
        image_bytes = await file.read()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Handle grayscale images
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        logger.info("Image shape=%s", img.shape)
            
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image file"}
        )

    # Parse flags
    gender_flag = gender == '1'
    age_flag = age == '1'
    race_flag = race == '1'
    logger.info("Flags gender=%s age=%s race=%s", gender_flag, age_flag, race_flag)
    face_info = []

    # Detect faces
    if timer:
        start = time.time()

    try:
        num_of_person, bbox, pts5, nimg = model.get_input(img)
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Face detection failed"}
        )
    logger.info("Detected faces: %s", num_of_person)

    if timer:
        end = time.time()
        detection_cost = end - start
        logger.info("Detection cost: %.4f s", detection_cost)

    # Extract features
    if timer:
        start = time.time()

    features = []
    extraction_cost_list = []
    for i in range(num_of_person):
        s = time.time()
        try:
            features.append(model.get_feature(nimg[i]).tolist())
        except Exception as e:
            logger.error(f"Feature extraction error for face {i}: {e}")
            features.append([])
        e = time.time()
        extraction_cost_list.append(e - s)

    if timer:
        end = time.time()
        extraction_cost = end - start
        logger.info("Feature extraction cost: %.4f s", extraction_cost)

    # Get gender, age, race if requested
    if timer:
        start = time.time()

    races = []
    genders = []
    ages = []
    race_cost_list = []
    if race_flag or gender_flag or age_flag:
        for i in range(num_of_person):
            s = time.time()
            try:
                gender_pred, age_pred, race_pred = model.get_gender_age_race(nimg[i])
                races.append(race_pred)
                genders.append(gender_pred)
                ages.append(age_pred)
            except Exception as e:
                logger.error(f"Attribute prediction error for face {i}: {e}")
                races.append("")
                genders.append("")
                ages.append(0)
            e = time.time()
            race_cost_list.append(e - s)

    if timer:
        end = time.time()
        race_cost = end - start
        logger.info("Attribute cost: %.4f s", race_cost)

    # Prepare response
    bbox = bbox.tolist() if num_of_person > 0 else []
    for i in range(num_of_person):
        person_info = {
            "bboxes": bbox[i] if i < len(bbox) else [],
            "feature": features[i] if i < len(features) else []
        }

        if race_flag or gender_flag or age_flag:
            person_info["race"] = races[i] if i < len(races) else ""
            person_info["gender"] = genders[i] if i < len(genders) else ""
            person_info["age"] = str(ages[i]) if i < len(ages) else str(0)

        face_info.append(person_info)

    # Log performance metrics
    total_time_end = time.time()
    if timer:
        average_extraction = (str(sum(extraction_cost_list) / len(extraction_cost_list))[:5]) \
            if extraction_cost_list else "None"
        average_race = str(sum(race_cost_list) / len(race_cost_list))[:5] \
            if race_cost_list else "None"
        logger.info(
            "total time cost: %.4f, detected %s faces, detection cost: %.4f, "
            "extraction cost: %.4f, average extract: %s, gender/age/race cost: %.4f, average attr: %s",
            total_time_end - total_time_start,
            num_of_person,
            detection_cost,
            extraction_cost,
            average_extraction,
            race_cost,
            average_race,
        )

    logger.info("---------------- END ----------------")
    return face_info


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/metrics")
def get_metrics():
    """Return service metrics"""
    return {
        "version": "1.0",
        "framework": "FastAPI",
        "engine": "onnx/om",
        "implementation": "local-models",
        "timestamp": time.time()
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Recognition Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=8112, help="Port to bind the server")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()
    
    logger.info("Starting FastAPI server on %s:%s", args.host, args.port)
    uvicorn.run(
        "face_fastapi:app", 
        host=args.host, 
        port=args.port, 
        reload=args.reload
    )
