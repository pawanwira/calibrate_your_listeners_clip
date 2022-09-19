import torch
import numpy as np
from PIL import Image
from torch import nn
import clip
from transformers import CLIPTextConfig
from calibrate_your_listeners.src.models import speaker_clip
from calibrate_your_listeners import constants
from transformers import CLIPTokenizer

class CLIPListenerScores(object):

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
        
        if self.config.training_params.ood_loss:
            self.listener_scores, self.ood_losses = self._calculate_listener_scores_with_ood_loss()
        else:
            self.listener_scores = self._calculate_listener_scores()

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

    def get_ood_loss(self, image_features, utterance_features):
        # id = in-distribution
        id_labels = ["a shape"]
        id_labels_tokens = clip.tokenize(id_labels).cuda()
        id_labels_features = self.listener.encode_text_original(id_labels_tokens).float()
        id_labels_features /= id_labels_features.norm(dim=-1, keepdim=True)

        # get image features of target image
        image_features = image_features[0]

        # features of all candidate labels (ID labels and speaker utterance)
        text_features = torch.cat((utterance_features, id_labels_features), dim=0)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        utterance_prob = text_probs[0]

        # ood loss 
        # TODO: adjust weight of ood loss via scaling, or pass it into a function, if appropriate
        ood_loss = 1 - utterance_prob
        return ood_loss
    
    # calculate listener scores with ood loss
    def _calculate_listener_scores_with_ood_loss(self):
        lis_scores = []
        ood_loss_list = []
        for i in range(len(self.imgs)):
            states = self.imgs[i]
            images = torch.tensor(np.stack([self._preprocess(state) for state in states])).cuda()
            max_idx = torch.tensor(np.argmax([self.lang_padded[i][j].argmax().item() for j in range(12)])).unsqueeze(0)  
            image_features = self.listener.encode_image(images).float()
            utterance_features = self.listener.encode_text(self.lang_padded[i], max_idx).float()
            image_features = image_features.clone() / image_features.clone().norm(dim=-1, keepdim=True)
            utterance_features = utterance_features.clone() / utterance_features.clone().norm(dim=-1, keepdim=True)
            ood_loss = self.get_ood_loss(image_features, utterance_features)
            ood_loss_list.append(ood_loss) 
            image_probs = (100.0 * utterance_features @ image_features.T).softmax(dim=-1)
            lis_scores.append(image_probs[0])

        lis_scores_final = torch.stack(lis_scores)
        ood_loss_final = torch.stack(ood_loss_list)
        return lis_scores_final, ood_loss_final
    
    def _calculate_listener_scores(self):
        lis_scores = []
        for i in range(len(self.imgs)):
            states = self.imgs[i]
            images = torch.tensor(np.stack([self._preprocess(state) for state in states])).cuda()
            max_idx = torch.tensor(np.argmax([self.lang_padded[i][j].argmax().item() for j in range(int(self.lang_length[i].item()))])).unsqueeze(0)
            image_features = self.listener.encode_image(images).float()
            utterance_features = self.listener.encode_text(self.lang_padded[i], max_idx).float()
            image_features = image_features.clone() / image_features.clone().norm(dim=-1, keepdim=True)
            utterance_features = utterance_features.clone() / utterance_features.clone().norm(dim=-1, keepdim=True)
            image_probs = (100.0 * utterance_features @ image_features.T).softmax(dim=-1)
            lis_scores.append(image_probs[0])

        lis_scores_final = torch.stack(lis_scores)
        return lis_scores_final
    
    def _calculate_listener_scores_test_with_cliptokenize(self):
        lis_scores = []
        batch_size = len(self.imgs)
        for i in range(batch_size):
            states = self.imgs[i]
            images = torch.tensor(np.stack([self._preprocess(state) for state in states])).cuda()
            utterance_tokens = self._get_utterance_tokens(i)
            image_features = self.listener.encode_image(images).float()
            utterance_features = self.listener.encode_text(utterance_tokens).float()
            image_features = image_features.clone() / image_features.clone().norm(dim=-1, keepdim=True)
            utterance_features = utterance_features.clone() / utterance_features.clone().norm(dim=-1, keepdim=True)
            image_probs = (100.0 * utterance_features @ image_features.T).softmax(dim=-1)
            lis_scores.append(image_probs[0])

        lis_scores_final = torch.stack(lis_scores)
        return lis_scores_final

    # def get_average_l0_score(self):
    #     return torch.mean(self.listener_scores, axis=0) # average across listeners
