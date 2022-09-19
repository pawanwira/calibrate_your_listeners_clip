from calibrate_your_listeners.src.models import vision_clip
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import pandas as pd
from torch.nn import functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, CLIPTextConfig, GPT2Config
import clip

from calibrate_your_listeners.src.models import (
    vision_clip
)
from calibrate_your_listeners import constants

class Speaker(nn.Module):
    def __init__(self, config):
        super(Speaker, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self._is_old = (self.config.model_params.vocab == "shapeworld")
        self.gpt2_config = GPT2Config()

        self.initialize_modules()
        
    def initialize_modules(self):
        # GPT2
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]  # 768
        self.prefix_length = self.config.model_params.prefix_length
        self.prefix_size = 512

        # CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32")
        self.clip_listener.cuda().eval()  # CHECK: necessary or not?
        self.freeze_model(self.clip_listener)
        self.clip_project = nn.Linear(self.prefix_size, self.gpt_embedding_size * self.prefix_length)

    def _preprocess(self, image):
        return self.clip_preprocess(Image.fromarray(np.uint8(image.cpu())).convert('RGB'))
   
    def set_vocab(self):
        pass

    def embed_imgs(self, imgs_sets):
        # 1 img set (from one reference game) consists of 3 imgs
        imgs_preprocessed = torch.tensor(np.stack([self._preprocess(img_set[0]) for img_set in imgs_sets])).cuda()
        with torch.no_grad():
            prefix = self.clip_model.encode_image(imgs_preprocessed).to(
                self.device, dtype=torch.float32
            )
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)  # CHECK: first dim should be batch size?
        return prefix_projections

    def forward(self, imgs, utterances):        
        embedding_text = self.gpt.transformer.wte(utterances)  # CHECK: can we pass utterances directly into wte?
        prefix_projections = self.embed_imgs(imgs)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        outputs = self.gpt(inputs_embeds=embedding_cat)
        return outputs

def to_onehot(y, n=3):
    y_onehot = torch.zeros(y.shape[0], n).to(y.device)
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot

class OriginalSpeaker(nn.Module): # L_0
    def __init__(self, config):

        super(Speaker, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self._is_old = (self.config.model_params.vocab == "shapeworld")
        self._max_seq_len = constants.MAX_SEQ_LEN
        self.clip_text_config = CLIPTextConfig()

        self.set_vocab()
        self.initialize_modules()

    def set_vocab(self):
        self.vocab_size = self.clip_text_config.vocab_size
        self._set_tokens()


    def initialize_modules(self):
        self.hidden_size = self.config.model_params.hidden_size
        self.embedding = nn.Embedding(self.vocab_size, 50) 

        self.init_lang_model()
        self.init_image_feature_model()

        self.image_feat_size = self.feat_model.final_feat_dim
        self.n_images = self.config.dataset_params.n_images
        self.imgFeat2hidden = nn.Linear(
            self.n_images * (self.image_feat_size + 1),
            self.hidden_size)

    def init_lang_model(self):
        self.gru = nn.GRU(
            self.embedding.embedding_dim, self.hidden_size)
        self.hidden2vocab = nn.Linear(self.hidden_size, self.vocab_size)

    def init_image_feature_model(self):
        self.feat_model = vision_clip.Conv4() # f_L(I_t)

    @property
    def is_old(self):
        return self._is_old

    def _set_tokens(self):
        # TODO: replace magic numbers with constants
        self._start_token = 49406 # start token id
        self._end_token = 49407 # end token id
        self._pad_token = self._end_token # 0 # pad token id

    def get_trainable_parameters(self, freeze_mode):
        return self.parameters()

    def embed_features(self, feats, targets):
        batch_size = feats.shape[0]
        n_obj = feats.shape[1]
        rest = feats.shape[2:]
        feats_flat = feats.view(batch_size * n_obj, *rest)
        feats_emb_flat = self.feat_model(feats_flat)
        feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)
        # Add targets
        targets_onehot = to_onehot(targets)
        feats_and_targets = torch.cat((feats_emb, targets_onehot.unsqueeze(2)), 2)
        ft_concat = feats_and_targets.view(batch_size, -1)
        return ft_concat

    # lang is onehots (e.g. onehots from teacher_forcing_forward)
    def insert_sos_token(self, lang, batch_size):
        lang_perm = lang.permute(1, 0, 2)
        sos_onehot = torch.zeros(batch_size, 1, self.clip_text_config.vocab_size, device=lang.device)
        sos_onehot[:, 0, self._start_token] = 1.0
        pad_onehot_perm = sos_onehot.permute(1, 0, 2)
        lang_padded = lang_perm
        lang_padded = torch.cat((pad_onehot_perm, lang_padded))
        lang_padded_perm = lang_padded.permute(1, 0, 2)
        return lang_padded_perm
    
    # y is targets
    def teacher_forcing_forward(self, feats, seq, targets): # length, targets): # y):
        batch_size = seq.shape[0]
        feats_emb = self.embed_features(feats=feats, targets=targets) # y.long())

        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1).to(feats.device)

        # embed your sequences
        embed_seq = seq @ self.embedding.weight

        # feats_emb = self.init_h(feats_emb)
        feats_emb = self.imgFeat2hidden(feats_emb)  # replaces line above
        
        feats_emb = feats_emb.unsqueeze(0)

        # shape = (seq_len, batch, hidden_dim)
        output, _ = self.gru(embed_seq, feats_emb)  
        # feats_emb here is the basically like states in "forward" function below
        # embed_seq is the sequence of ground truth words in utterances, rather than the predicted_onehot (predicted next word in utterance) in the "forward" function below

        # reorder from (L,B,D) to (B,L,D)
        # (batch_size, max_sequence_length, hidden unit size)
        output = output.transpose(0, 1) 
        ### output_presoftmax = self.hidden2vocab(output) 
        ### onehots = F.gumbel_softmax(output_presoftmax, tau=1.0, hard=True)
        ### onehots_with_sos = self.insert_sos_token(onehots, batch_size)

        max_length = output.size(1)
        output_2d = output.reshape(batch_size * max_length, -1)
        # outputs_2d = self.outputs2vocab(output_2d)
        outputs_2d = self.hidden2vocab(output_2d)

        # Distribution over vocab for each batch, (batch_size, max_seq_length, vocab_size)
        lang_tensor = outputs_2d.reshape(batch_size, max_length, self.vocab_size)
        return lang_tensor 

    def forward(self, feats, targets, activation='gumbel', tau=1.0, length_penalty=False):
        batch_size = feats.size(0)
        feats_emb = self.embed_features(feats, targets)

        # initialize hidden states using image features
        states = self.imgFeat2hidden(feats_emb)
        states = states.unsqueeze(0)

        # This contains are series of sampled onehot vectors
        lang = []
        lang_prob = None
        if length_penalty:
            eos_prob = []

        # And vector lengths
        lang_length = np.ones(batch_size, dtype=np.int64)
        done_sampling = np.array([False for _ in range(batch_size)])

        # first input is SOS token
        # (batch_size, n_vocab)
        inputs_onehot = torch.zeros(batch_size, self.vocab_size, device=feats.device)

        # No start token for GPT - leave inputs as onehot

        # Start token for CLIP
        inputs_onehot[:, self._start_token] = 1.0  # edit jul 15

        # (batch_size, len, n_vocab)
        inputs_onehot = inputs_onehot.unsqueeze(1)

        # Add SOS to lang
        lang.append(inputs_onehot)

        # (B,L,D) to (L,B,D)
        inputs_onehot = inputs_onehot.transpose(0, 1)
        
        # compute embeddings
        # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
        inputs = inputs_onehot @ self.embedding.weight
        max_len = self._max_seq_len # - 2  # clip

        for i in range(max_len):  # Have room for EOS if never sampled
            # FIXME: This is inefficient since I do sampling even if we've
            # finished generating language.
            if all(done_sampling):
                break

            outputs, states = self.gru(inputs, states)  # outputs: (L=1,B,H)
            outputs = outputs.squeeze(0)                # outputs: (B,H)
            outputs = self.hidden2vocab(outputs)       # outputs: (B,V)

            if activation=='gumbel':
                predicted_onehot = F.gumbel_softmax(outputs, tau=tau, hard=True)
            else:
                raise NotImplementedError(activation)

            # Add to lang
            lang.append(predicted_onehot.unsqueeze(1))
            if length_penalty:
                idx_prob = F.log_softmax(outputs, dim = 1)
                eos_prob.append(idx_prob[:, self._end_token])

            # Update language lengths
            lang_length += ~done_sampling
            done_sampling = np.logical_or(
                done_sampling,
                (predicted_onehot[:, self._end_token] == 1.0).cpu().numpy())
            # assert activation in {'gumbel', 'multinomial'}, "check activation either gumbel or multinom"

            # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
            inputs = (predicted_onehot.unsqueeze(0)) @ self.embedding.weight

        # Add EOS if we've never sampled it
        if not all(done_sampling):
            eos_onehot = torch.zeros(batch_size, 1, self.vocab_size, device=feats.device)
            eos_onehot[:, 0, self._end_token] = 1.0
            lang.append(eos_onehot)

            # Cut off the rest of the sentences
            lang_length += (~done_sampling) 

        pad_onehot = torch.zeros(batch_size, 1, self.vocab_size, device=feats.device)
        pad_onehot[:, 0, self._pad_token] = 1.0
        # for i in range(self.clip_text_config.max_position_embeddings - max(lang_length)):
        for i in range(max_len + 2 - max(lang_length)):
            lang.append(pad_onehot)

        # Cat language tensors (batch_size, max_seq_length, vocab_size)
        # skip first element b/c it's just 0s
        # no SOS token for GPT
        """if self._is_old:
            lang_tensor = torch.cat(lang, 1)
        else:
            lang_tensor = torch.cat(lang[1:], 1)
            lang_length -= 1"""
        lang_tensor = torch.cat(lang, 1)  # clip

        """for i in range(lang_tensor.shape[0]):
            lang_tensor[i, lang_length[i]:] = 0"""

        # Trim max length
        # max_lang_len = lang_length.max()
        # lang_tensor = lang_tensor[:, :max_lang_len, :]

        if length_penalty:
            # eos prob -> eos loss
            eos_prob = torch.stack(eos_prob, dim = 1)
            for i in range(eos_prob.shape[0]):
                r_len = torch.arange(1,eos_prob.shape[1]+1,dtype=torch.float32)
                eos_prob[i] = eos_prob[i]*r_len.to(eos_prob.device)
                eos_prob[i, lang_length[i]:] = 0
            eos_loss = -eos_prob
            eos_loss = eos_loss.sum(1)/torch.tensor(
                lang_length,dtype=torch.float32, device=eos_loss.device)
            eos_loss = eos_loss.mean()
        else:
            eos_loss = 0

        # Sum up log probabilities of samples
        lang_length = torch.Tensor(lang_length)

        return lang_tensor, lang_length, eos_loss, self.embedding # , lang_prob
