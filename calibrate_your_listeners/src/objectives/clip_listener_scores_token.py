import torch
import numpy as np
from PIL import Image
from torch import nn
import clip
from transformers import CLIPTextConfig
from calibrate_your_listeners.src.models import speaker_clip
from calibrate_your_listeners import constants
from transformers import CLIPTokenizer

class CLIPListenerTokenScores(object):

    def __init__(self, listener, imgs, tokenizer, preprocess, vocab_type, lang, lang_length, embedding_module=None, config=None):
        self.config = config
        self.listener = listener
        self.imgs = imgs
        self.preprocess = preprocess
        self.vocab_type = vocab_type
        self.lang = lang
        self.lang_length = lang_length
        self.embedding = embedding_module
        self.clip_text_config = CLIPTextConfig()
        self._pad_token = 49407 # speaker_clip._pad_token # TODO: replace magic number
        self.lang_padded = self.pad_lang(self.lang)
        self.tokenizer = tokenizer
        self._construct_text_langs_list()
 
        self.token_losses = self._calculate_listener_scores()

    def freeze_model(self, model):
        for p in model.parameters():
            p.requires_grad = False
    
    def _preprocess(self, image):
        return self.preprocess(Image.fromarray(np.uint8(image.cpu())).convert('RGB'))

    def _process_gt_clip(self, gt):
        if isinstance(gt, dict):
            result = []
            for seq_id in range(gt['input_ids'].shape[0]):
                result.append(self.tokenizer.decode(gt['input_ids'][seq_id][1:]))  # [1:] to remove sos token
            return result
    
    def _construct_text_langs_list(self):
        text_tokens = {'input_ids': self.lang.argmax(-1)}
        self.text_langs_list = self._process_gt_clip(text_tokens)
    
    def _get_utterance_tokens(self, i):
        utterance = self.text_langs_list[i]
        utterance_tokens = clip.tokenize(utterance).cuda()
        return utterance_tokens

    def _get_utterance_tokens_from_df(self, i):
        if self.vocab_type == "gpt2":
            lang = self.df['speaker_utterance'][i][:-13]
        elif self.vocab_type == "shapeworld":
            lang = self.df['speaker_utterance'][i].replace("<sos>", "").replace("<eos>", "").replace("<UNK>", "").replace("<PAD>", "")
            lang = " ".join(lang.split())
        utterance = lang
        utterance_tokens = clip.tokenize(utterance).cuda()
        return utterance_tokens

    def pad_lang(self, lang):
        lang_perm = lang.permute(1, 0, 2)
        batch_size = self.imgs.size(0)
        pad_onehot = torch.zeros(batch_size, 1, self.clip_text_config.vocab_size, device=self.lang.device)
        pad_onehot[:, 0, self._pad_token] = 1.0
        pad_onehot_perm = pad_onehot.permute(1, 0, 2)
        lang_padded = lang_perm
        max_len = constants.MAX_SEQ_LEN + 2
        for i in range(self.clip_text_config.max_position_embeddings - max_len):
            lang_padded = torch.cat((lang_padded, pad_onehot_perm))
        lang_padded_perm = lang_padded.permute(1, 0, 2)
        return lang_padded_perm

    def _calculate_listener_scores(self):
        losses = []
        for i in range(len(self.imgs)):
            lis_scores = []
            states = self.imgs[i]
            images = torch.tensor(np.stack([self._preprocess(state) for state in states])).cuda()
            image_features = self.listener.encode_image(images).float()
            image_features = image_features.clone() / image_features.clone().norm(dim=-1, keepdim=True)
            
            lang = self.lang_padded[i].clone()
            partial_lang = lang
            eos_token_loc = int(self.lang_length[i].item() - 1)
            eos_onehot = torch.zeros(self.clip_text_config.vocab_size, device=self.lang.device)
            eos_onehot[self._pad_token] = 1.0
            num_tokens = int(self.lang_length[i].item()) - 2
            for k in range(0, num_tokens):
                partial_lang[eos_token_loc] = eos_onehot
                max_idx = torch.tensor(np.argmax([partial_lang[j].argmax().item() for j in range(int(self.lang_length[i].item()))])).unsqueeze(0)
                utterance_features = self.listener.encode_text(partial_lang, max_idx).float()
                utterance_features = utterance_features.clone() / utterance_features.clone().norm(dim=-1, keepdim=True)
                image_probs = (100.0 * utterance_features @ image_features.T).softmax(dim=-1)
                lis_scores.append(image_probs[0])
                eos_token_loc -= 1
            num_labels = num_tokens

            if num_tokens == 0:
                max_idx = torch.tensor(np.argmax([partial_lang[j].argmax().item() for j in range(int(self.lang_length[i].item()))])).unsqueeze(0)
                utterance_features = self.listener.encode_text(partial_lang, max_idx).float()
                utterance_features = utterance_features.clone() / utterance_features.clone().norm(dim=-1, keepdim=True)
                image_probs = (100.0 * utterance_features @ image_features.T).softmax(dim=-1)
                lis_scores.append(image_probs[0])
                num_labels = 1
                
            lis_scores_final = torch.stack(lis_scores)
            loss_f = nn.CrossEntropyLoss()
            # loss_f = nn.CrossEntropyLoss(reduction='sum')
            labels = torch.zeros(num_labels, device=self.lang.device, dtype=torch.int64)
            loss = loss_f(lis_scores_final, labels)
            losses.append(loss)
        
        token_losses = torch.stack(losses)
        return token_losses
        
    # def get_average_l0_score(self):
    #     return torch.mean(self.listener_scores, axis=0) # average across listeners
