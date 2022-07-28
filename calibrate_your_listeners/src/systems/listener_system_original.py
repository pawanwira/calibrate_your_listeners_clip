from calibrate_your_listeners import constants
from calibrate_your_listeners.src.systems import system
from calibrate_your_listeners.src.models import (
    dropout_listener,
    listener,
)

from transformers import GPT2Tokenizer
import torch.nn as nn
import torch
import os

class ListenerSystem(system.BasicSystem):

    def __init__(self, config):
        super().__init__(config=config)
        self.post_model_init()

    def post_model_init(self):
        self.train_dataset.listener_tokenize_f=self.model.tokenize
        self.val_dataset.listener_tokenize_f=self.model.tokenize
        self.test_dataset.listener_tokenize_f=self.model.tokenize

    def set_models(self):
        num_sw_tokens = len(self.train_dataset.vocab['w2i'].keys())
        num_gpt2_tokens = GPT2Tokenizer.from_pretrained('gpt2').vocab_size
        self.config.dataset_params.num_shapeworld_tokens = num_sw_tokens
        if self.config.model_params.type == "normal":
            l0 = listener.Listener(config=self.config)
        if self.config.model_params.type == "ensemble":
            l0 = listener.Listener(config=self.config)
        elif self.config.model_params.type == "dropout":
            l0 = dropout_listener.DropoutListener(config=self.config)
        self.model = l0
        print('LISTENER MODEL ARCHITECTURE')

    def save(self):
        l0_dir = constants.NORMAL_LISTENER_MODEL_DIR
        if self.config.model_params.vocab == "shapeworld":
            vocab_type = "small_vocab"
        elif self.config.model_params.vocab == "gpt2":
            vocab_type = "big_vocab"

        model_type = self.config.model_params.type

        listener_idx = self.config.model_params.listener_idx

        model_fname = os.path.join(
            l0_dir,
            vocab_type,
            f"{model_type}_listener_{listener_idx}.pt"
        )
        torch.save(self.model, model_fname)

    def get_losses_for_batch(self, batch, batch_idx):
        imgs, labels, utterances = (
            batch['imgs'].float(), batch['label'].argmax(-1).long(), batch['utterance'])
        utterance_lengths = self.model.get_length(utterances)
        lis_scores, _ = self.model(imgs, utterances, utterance_lengths)
        lis_pred = lis_scores.argmax(1)
        loss = nn.CrossEntropyLoss()
        losses = loss(lis_scores, labels)
        return {'loss': losses, 'acc': (lis_pred == labels).float().mean()}

