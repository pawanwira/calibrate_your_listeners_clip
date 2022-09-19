from calibrate_your_listeners.src.systems import system
from calibrate_your_listeners.src.models import (
    dropout_listener,
    listener,
    speaker_clip,
)
from calibrate_your_listeners.src.objectives import (
    listener_scores,
    dropout_listener_scores,
    clip_listener_scores
)
from calibrate_your_listeners.src.systems import utils
from calibrate_your_listeners import constants
from pytorch_lightning.trainer.supporters import CombinedLoader

from transformers import CLIPTokenizer

import os
import torch
import torch.nn as nn
import pandas as pd
import wandb
import numpy as np
from PIL import Image
from pkg_resources import packaging
from torch.nn import functional as F
import clip
from calibrate_your_listeners import constants
import math

class SpeakerCLIPSystem(system.BasicSystem):

    def __init__(self, config):
        super().__init__(config=config)

        self._max_seq_len = constants.MAX_SEQ_LEN
        self.post_model_init()

    def post_model_init(self):
        # model = self.train_listeners[0]
        model = self.val_listeners[0]
        self.train_dataset.listener_tokenize_f=model.tokenize
        self.val_dataset.listener_tokenize_f=model.tokenize
        self.test_dataset.listener_tokenize_f=model.tokenize

        self.exp_name = self.config.wandb_params.exp_name
        self.lang_table_path = os.path.join("/data3/pawanw/lang_table/clip_vocab", self.exp_name)
        if not os.path.exists(self.lang_table_path):
            os.makedirs(self.lang_table_path)

    def freeze_model(self, model):
        for p in model.parameters():
            p.requires_grad = False

    def _load_listener(self, listener_type, vocab_type, listener_idx):
        # if vocab_type == "shapeworld":
        #     vocab = "small_vocab"
        # elif vocab_type == "gpt2":
        #     vocab = "big_vocab"

        vocab = 'big_clip_vocab'

        if listener_type == "normal":
            # l0 = listener.Listener(config=self.config)
            l0_dir = constants.NORMAL_LISTENER_MODEL_DIR
            model_fname = os.path.join(
                l0_dir,
                vocab,
                f"normal_listener_{listener_idx}.pt"
                )
            self.l0_scorer = listener_scores.ListenerScores
        elif listener_type == "ensemble":
            # l0 = listener.Listener(config=self.config)
            l0_dir = constants.NORMAL_LISTENER_MODEL_DIR
            model_fname = os.path.join(
                l0_dir,
                vocab,
                f"ensemble_listener_{listener_idx}.pt"
                )
            self.l0_scorer = listener_scores.ListenerScores
        elif listener_type == "dropout":
            # l0 = dropout_listener.DropoutListener(config=self.config)
            l0_dir = constants.DROPOUT_LISTENER_MODEL_DIR
            model_fname = os.path.join(
                l0_dir,
                vocab,
                f"dropout_listener_{listener_idx}.pt"
                )
            self.l0_scorer = dropout_listener_scores.DropoutListenerScores

        print(f'Loading listener from {model_fname}')
        # state_dict = self ._load_and_process_state_dict(model_fname)
        # l0.load_state_dict(state_dict)

        # Keep dropout for speaker
        # if listener_type == "normal":
        #     l0.eval()
        l0 = torch.load(model_fname)
        return l0

    def _load_and_process_state_dict(self, model_fname):
        state_dict = torch.load(model_fname)['state_dict']
        new_state_dict = dict()
        for k, v in state_dict.items():
            key_ = 'model.'
            if k.startswith(key_):
                k = k[len(key_):]
            new_state_dict[k] = v
        return new_state_dict

    def load_listeners(self):
        # CLIP as listener
        self.clip_listener, self.preprocess = clip.load("ViT-B/32")
        self.freeze_model(self.clip_listener)
        self.clip_scorer = clip_listener_scores.CLIPListenerScores
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        self.val_listeners = []

        for listener_idx in range(0, self.config.listener_params.ensemble_size):
            # Val listeners
            print('Loading validation listener')
            # val_idx = (self.config.listener_params.val_idx
            #            if self.config.listener_params.val_idx else listener_idx)
            print(f'Val idx: {listener_idx}')
            listener = self._load_listener(
                listener_type=self.config.listener_params.type,
                vocab_type='big_clip_vocab', # self.config.model_params.vocab,
                listener_idx = listener_idx + 4 # if self.config.dataset_params.data_dir == "clip/sw" else listener_idx + 1
                )
            self.freeze_model(listener)
            self.val_listeners.append(listener)
        print(f'A validation listener arch: {self.val_listeners[0]}')


    def load_speaker(self):
        self.model = speaker_clip.Speaker(config=self.config)

    def set_models(self):
        num_tokens = len(self.train_dataset.vocab['w2i'].keys())
        self.config.dataset_params.num_shapeworld_tokens = num_tokens
        self.load_listeners()
        self.load_speaker()

    def get_entropy(self, probs):
        entropy = 0
        for prob in probs:
            prob_val = prob.item()
            if prob_val != 0:
                to_add = (-(prob_val * math.log(prob_val, 3)))
                entropy += to_add
        return entropy

    def get_entropies(self, lis_scores):
        result = []
        for i in range(len(lis_scores)):
            entropy = self.get_entropy(lis_scores[i])
            result.append(entropy)
        return result

    def _process_gt(self, gt):
        if isinstance(gt, dict):
            result = []
            for seq_id in range(gt['input_ids'].shape[0]):
                # change aug 10: because we no longer use train_listeners that come from Listener class. we use CLIP as train listeners instead.
                # TODO: technically in this module, self.val_listeners[0]._tokenizer.decode should be the same as self.tokenizer.decode, so can replace the former with the latter
                result.append(self.val_listeners[0]._tokenizer.decode(gt['input_ids'][seq_id]))  # self.train_listeners[0]._tokenizer is now CLIP's, not GPT2's
                # result.append(self.train_listeners[0]._tokenizer.decode(gt['input_ids'][seq_id]))  # self.train_listeners[0]._tokenizer is now CLIP's, not GPT2's
            return result
        else:
            return self.train_dataset.to_text(gt.argmax(-1))
        
    def _process_gt_clip(self, gt):
        if isinstance(gt, dict):
            result = []
            for seq_id in range(gt['input_ids'].shape[0]):
                result.append(self.tokenizer.decode(gt['input_ids'][seq_id]))
            return result

    def construct_lang_table_original(self, lang, gt):
        data = []
        text_gts = self._process_gt(gt)
        if isinstance(gt, dict):
            lang = {'input_ids': lang.argmax(-1)}
        text_langs = self._process_gt_clip(lang) # self.train_dataset.to_text(lang.argmax(-1))
        for l, gt in zip(text_langs, text_gts):
            data.append({
                'epoch': self.trainer.current_epoch,
                'speaker_utterance': l,
                'ground_truth': gt
            })
        return pd.DataFrame(data)

    def construct_lang_table(self, lang, gt, lis_scores):
        data = []
        text_gts = self._process_gt(gt)
        if isinstance(gt, dict):
            lang = {'input_ids': lang.argmax(-1)}
        text_langs = self._process_gt_clip(lang) # self.train_dataset.to_text(lang.argmax(-1))
        entropies = self.get_entropies(lis_scores)
        for e, ls, l, gt in zip(entropies, lis_scores, text_langs, text_gts):
            data.append({
                'epoch': self.trainer.current_epoch,
                'entropy': e,
                'lis_scores': ls.tolist(),
                'speaker_utterance': l,
                'ground_truth': gt
            })
        return pd.DataFrame(data)
    
    def construct_and_save_lang_table(self, lang, gt, lis_scores, batch_idx, prefix):
        if (self.trainer.current_epoch > 90) or (batch_idx % 30 == 0):
            df=self.construct_lang_table(lang, gt, lis_scores)
            self.save_lang_table(df, batch_idx, prefix)
    
    def get_teacher_forcing_loss(self, gt, img, targets):
        loss = 0.0
        gt_tokens = gt['input_ids']
        gt_onehot = F.one_hot(gt_tokens, num_classes=self.model.vocab_size).cuda().float()
        if self.config.training_params.tf_inputnotgt:
            lang, lang_length, eos_loss, lang_4_tfloss = self.model.teacher_forcing_forward_inputnotgt(feats=img, targets=targets, seq=gt_onehot)
        else:
            lang, lang_length, eos_loss, lang_4_tfloss = self.model.teacher_forcing_forward(feats=img, targets=targets, seq=gt_onehot)
        batch_size = lang_4_tfloss.shape[0]
        lang_4_tfloss = lang_4_tfloss.view(batch_size*lang_4_tfloss.size(1), self.model.vocab_size)
        gt_onehot = gt_onehot[:, 1:11, :]
        gt = gt_onehot.long().view(batch_size*gt_onehot.size(1), self.model.vocab_size)
        loss_f = nn.CrossEntropyLoss()
        tf_loss = loss_f(lang_4_tfloss.cuda(), torch.max(gt, 1)[1].cuda())
        return lang, lang_length, tf_loss 

    # TEACHER FORCING
    def get_losses_for_batch_tf(self, batch, batch_idx, which_listener, prefix):
        imgs_speaker, labels, utterances, imgs_clip = (
            batch['imgs'].float(), batch['label'].argmax(-1).long(), batch['utterance'], batch['imgs_original'])

        lang, lang_length, tf_loss = self.get_teacher_forcing_loss(utterances, imgs_speaker, labels)

        if (which_listener == "clip") or (which_listener == "train"):
            clip_scorer = self.clip_scorer(
                listener=self.clip_listener,
                imgs=imgs_clip,
                tokenizer = self.tokenizer,
                preprocess=self.preprocess,
                vocab_type=self.config.model_params.vocab,
                lang=lang,
                lang_length=lang_length,
                config = self.config
            )
            lis_scores = clip_scorer.listener_scores
            loss = nn.CrossEntropyLoss()
            lis_pred = lis_scores.argmax(1)
            losses = loss(lis_scores, labels)
            acc = (lis_pred == labels).float().mean()
        elif (which_listener == "val") or (which_listener == "any"):
            l0_scorer = self.l0_scorer(
                listeners=self.val_listeners,
                imgs=imgs_speaker,
                lang=lang,
                lang_length=lang_length,
                config=self.config
            )
            lis_scores = l0_scorer.get_average_l0_score()
            lis_pred = lis_scores.argmax(1)
            loss = nn.CrossEntropyLoss()
            losses = loss(lis_scores, labels)
            acc = (lis_pred == labels).float().mean()

        # df=self.construct_lang_table(lang=lang, gt=utterances, lis_scores=lis_scores)
        # self.save_lang_table(df, batch_idx, prefix)

        self.construct_and_save_lang_table(lang=lang, gt=utterances, lis_scores=lis_scores, batch_idx=batch_idx, prefix=prefix)

        return {
            'pragmatic_loss': losses,
            'pragmatic_acc': acc,
            'teacher_forcing_loss': tf_loss,
            'total_loss': losses + tf_loss
        #     'lang_table': df
        #     'lang_table': wandb.Table(
        #         dataframe=self.construct_lang_table(lang=lang, gt=utterances)
        #     )
        }
   
    def get_losses_for_batch(self, batch, batch_idx, which_listener, prefix):

        imgs_speaker, labels, utterances, imgs_clip = (
            batch['imgs'].float(), batch['label'].argmax(-1).long(), batch['utterance'], batch['imgs_original'])

        lang, lang_length, eos_loss = self.model(imgs_speaker, labels)

        if which_listener == "train":

            clip_scorer = self.clip_scorer(
                listener=self.clip_listener,
                imgs=imgs_clip,
                tokenizer = self.tokenizer,
                preprocess=self.preprocess,
                vocab_type=self.config.model_params.vocab,
                lang=lang,
                lang_length=lang_length,
                config = self.config
            )
            lis_scores = clip_scorer.listener_scores
            loss = nn.CrossEntropyLoss()
            lis_pred = lis_scores.argmax(1)
            losses = loss(lis_scores, labels)
            acc = (lis_pred == labels).float().mean()
        elif which_listener == "val":
            l0_scorer = self.l0_scorer(
                listeners=self.val_listeners,
                imgs=imgs_speaker,
                lang=lang,
                lang_length=lang_length,
                config=self.config
            )
            lis_scores = l0_scorer.get_average_l0_score()
            lis_pred = lis_scores.argmax(1)
            loss = nn.CrossEntropyLoss()
            losses = loss(lis_scores, labels)
            acc = (lis_pred == labels).float().mean()
        elif which_listener == "clip":
            clip_scorer = self.clip_scorer(
                    listener=self.clip_listener,
                    imgs=imgs_clip,
                    tokenizer = self.tokenizer,
                    preprocess=self.preprocess,
                    vocab_type=self.config.model_params.vocab,
                    lang=lang,
                    lang_length=lang_length,
                    config = self.config
            )
            lis_scores = clip_scorer.listener_scores
            loss = nn.CrossEntropyLoss()
            lis_pred = lis_scores.argmax(1)
            losses = loss(lis_scores, labels)
            acc = (lis_pred == labels).float().mean()

        # df=self.construct_lang_table(lang=lang, gt=utterances, lis_scores=lis_scores)
        # self.save_lang_table(df, batch_idx, prefix)

        self.construct_and_save_lang_table(lang=lang, gt=utterances, lis_scores=lis_scores, batch_idx=batch_idx, prefix=prefix)
        
        return {
            'pragmatic_loss': losses,
            'pragmatic_acc': acc,
        }

    def save_lang_table(self, df, batch_idx, prefix):
        vocab = "clip_vocab"
        fpath = os.path.join(
            self.lang_table_path,
            f"{prefix}-{self.trainer.current_epoch}-{batch_idx}.csv"
        )
        df.to_csv(fpath, index=False, escapechar="\\")

    def _convert_results_to_floats(self, result):
        results = dict()
        for k, v in result.items():
            if k == "lang_table":
                results[k] = v
            else:
                results[k] = v.item()
        return results

    def log_results(self, result, category):
        # if category == "train":
        #     # remove lang table
        #     result.pop('lang_table', None)
        super().log_results(result, category)

    def training_step(self, batch, batch_idx):
        if self.config.training_params.tf:
            if self.config.training_params.prag_lmbd > 0:
                which_listener = "train"
            else: 
                which_listener = "any"
            result = self.get_losses_for_batch_tf(batch, batch_idx, which_listener=which_listener, prefix="train")
            prag_loss = result['pragmatic_loss']
            tf_loss = result['teacher_forcing_loss']
            self.log_results(result=result, category="train")
            loss_final = ((prag_loss * self.config.training_params.prag_lmbd)
                            + (tf_loss * self.config.training_params.tf_lmbd))
            return loss_final

        result = self.get_losses_for_batch(batch, batch_idx, which_listener="train", prefix="train")
        prag_loss = result['pragmatic_loss']
        self.log_results(result=result, category="train")
        return prag_loss

    def test_step(self, batch, batch_idx):
        result = self.get_losses_for_batch(batch, batch_idx, which_listener="test", prefix="test")
        loss = result['loss']
        return loss

    def validation_step(self, batch, batch_idx):
        if self.config.training_params.tf:
            settings = ["valL0_trainD", "valL0_valD", "CLIP_trainD", "CLIP_valD"]
            for setting in settings: 
                if "valL0" in setting:
                    which_listener = "val"
                elif "CLIP" in setting:
                    which_listener = "clip"
                result = self.get_losses_for_batch_tf(batch[setting], batch_idx, which_listener=which_listener, prefix=setting)
                prag_loss = result['pragmatic_loss']
                tf_loss = result['teacher_forcing_loss']
                self.log_results(result=result, category=setting)
                loss_final = ((prag_loss * self.config.training_params.prag_lmbd)
                            + (tf_loss * self.config.training_params.tf_lmbd))
            return loss_final

        if ((self.trainer.current_epoch % 10) == 0) or (self.trainer.current_epoch < 6) or (self.trainer.current_epoch > 90):
            settings = ["valL0_trainD", "valL0_valD", "CLIP_trainD", "CLIP_valD"]
        else:
            return None
        for setting in settings:  
            if "valL0" in setting:
                which_listener = "val"
            elif "CLIP" in setting:
                which_listener = "clip"
            result = self.get_losses_for_batch(batch[setting], batch_idx, which_listener=which_listener, prefix=setting)
            prag_loss = result['pragmatic_loss']
            self.log_results(result=result, category=setting)
        return prag_loss

    def val_dataloader(self):
        loaders = {
            "trainL0_trainD": utils.create_dataloader(self.train_dataset, self.config, shuffle=False),
            "trainL0_valD": utils.create_dataloader(self.val_dataset, self.config, shuffle=False),
            "valL0_trainD": utils.create_dataloader(self.train_dataset, self.config, shuffle=False),
            "valL0_valD": utils.create_dataloader(self.val_dataset, self.config, shuffle=False),
            "CLIP_trainD": utils.create_dataloader(self.train_dataset, self.config, shuffle=False),
            "CLIP_valD": utils.create_dataloader(self.val_dataset, self.config, shuffle=False)
        }
        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return combined_loaders
