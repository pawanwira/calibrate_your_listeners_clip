
from calibrate_your_listeners.src.datasets import (
    shapeworld
)
import os
from transformers import CLIPTokenizer
import clip

ROOT_DIR=os.getcwd()
MAIN_REPO_DIR=os.getcwd()
DROPOUT_LISTENER_MODEL_DIR=os.path.join(MAIN_REPO_DIR, "src/models/checkpoints")
NORMAL_LISTENER_MODEL_DIR=os.path.join(MAIN_REPO_DIR, "src/models/checkpoints")
# self.clip_listener, self.preprocess = clip.load("ViT-B/32")
# self.clip_scorer = clip_listener_scores.CLIPListenerScores
# self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

MAX_SEQ_LEN= 10 # 77 # TODO: change back to 10 when ready
EPS=1e-5

NAME2DATASETS = {
    'shapeworld': shapeworld.Shapeworld
}
