
import time 
### ===============================
# Ultralytics SAM3 Integration
# Usage: inference, image and video
# ultralytics>=8.3.237
### ===============================

startTime = time.time() 
### === 01 - image inference (with text-prompt) ===
from ultralytics.models.sam.predict import SAM3SemanticPredictor

# Initialize SAM3 predictor with configuration
overrides = dict(
    conf=0.25,              # confidence threshold
    task="segment",         # task i.e. segment
    mode="predict",         # mode i.e. predict
    model="SAM3/sam3.pt",        # model file = sam3.pt
    half=True,              # Use FP16 for faster inference on GPU.
)
predictor = SAM3SemanticPredictor(overrides=overrides)

predictor.set_image("datasets/sidewalk_segmentation/images/train/tile_6144_13312.jpg")  # Inference on image.
results = predictor(text=["sidewalk"], save=True)
endTime = time.time()
print("Time taken: ", endTime - startTime)
## ================================================


## === 02 - image inference (with multiple text-prompt) ===
# from ultralytics.models.sam.predict import SAM3SemanticPredictor
#
# # Initialize SAM3 predictor with configuration
# overrides = dict(
#     conf=0.25,              # confidence threshold
#     show_conf=False,        # Enable/disable confidence display
#     task="segment",         # task i.e. segment
#     mode="predict",         # mode i.e. predict
#     model="sam3.pt",        # model file = sam3.pt
#     half=True,              # Use FP16 for faster inference on GPU.
# )
# predictor = SAM3SemanticPredictor(overrides=overrides, bpe_path="bpe_simple_vocab_16e6.txt.gz")
#
# predictor.set_image("cars.png")  # Inference on image.
# results = predictor(text=["red car", "black car"], save=True)
## ========================================================


## === 03 - video Inference - black cars example ===
## with text-prompt semantic feature, track objects.
# from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor
#
# # Initialize SAM3 predictor with configuration
# overrides = dict(
#     conf=0.25,              # confidence threshold
#     show_conf=False,        # Enable/disable confidence display
#     task="segment",         # task i.e. segment
#     mode="predict",         # mode i.e. predict
#     model="sam3.pt",        # model file = sam3.pt
#     half=True,              # Use FP16 for faster inference on GPU.
# )
# predictor = SAM3VideoSemanticPredictor(overrides=overrides)
# results = predictor(source="cars1.mp4", text=["black car"], stream=False) # large stream, use stream=True.
## =================================================


## === 04 - video Inference - white dog example ===
## with text-prompt semantic feature, track objects.
# from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor
#
# # Initialize SAM3 predictor with configuration
# overrides = dict(
#     conf=0.25,              # confidence threshold
#     show_conf=False,        # Enable/disable confidence display
#     task="segment",         # task i.e. segment
#     mode="predict",         # mode i.e. predict
#     model="sam3.pt",        # model file = sam3.pt
#     half=True,              # Use FP16 for faster inference on GPU.
# )
# predictor = SAM3VideoSemanticPredictor(overrides=overrides)
# results = predictor(source="dogs.mp4", text=["white dog"], stream=False) # large stream, use stream=True.
## ================================================


## === 05 - video Inference - penguins walking example ===
# text-prompt semantic feature, track objects
# from ultralytics.models.sam.predict import SAM3VideoSemanticPredictor
#
# # Initialize SAM3 predictor with configuration
# overrides = dict(
#     conf=0.25,              # confidence threshold
#     show_conf=False,        # Enable/disable confidence display
#     task="segment",         # task i.e. segment
#     mode="predict",         # mode i.e. predict
#     model="sam3.pt",        # model file = sam3.pt
#     half=True,              # Use FP16 for faster inference on GPU.
# )
# predictor = SAM3VideoSemanticPredictor(overrides=overrides)
# results = predictor(source="penguins.mp4", text=["penguin"], stream=False) # large stream, use stream=True.
## ================================================
