from calibrate_your_listeners.src.systems import system
# from calibrate_your_listeners.src.models_original import ( # TODO: continue debugging 'import models' error
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

        # TODO: Check self.parameters() - check that the speaker parameters are in here
        # import pdb; pdb.set_trace()

    def post_model_init(self):
        # import pdb; pdb.set_trace()
        model = self.train_listeners[0]
        self.train_dataset.listener_tokenize_f=model.tokenize
        self.val_dataset.listener_tokenize_f=model.tokenize
        self.test_dataset.listener_tokenize_f=model.tokenize

        # import pdb; pdb.set_trace()
        self.exp_name = self.config.wandb_params.exp_name
        # path = os.path.join(constants.MAIN_REPO_DIR, "clip", "lang_table", "clip_vocab", self.exp_name)
        # self.lang_table_path = os.path.join("data3", "pawanw", "lang_table", "clip_vocab", self.exp_name)
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
        """self.listener, self.preprocess = clip.load("ViT-B/32")
        self.listener.cuda().eval()
        self.freeze_model(self.listener)"""

        """self.old_listener = self._load_listener(
                listener_type=self.config.listener_params.type,
                vocab_type=self.config.model_params.vocab,
                listener_idx = listener_idx + 1    
                )
        self.freeze_model(self.old_listener)"""

        # CLIP as listener
        self.clip_listener, self.preprocess = clip.load("ViT-B/32")
        # self.clip_listener.cuda().eval()
        self.freeze_model(self.clip_listener)
        self.clip_scorer = clip_listener_scores.CLIPListenerScores
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        self.train_listeners = []
        self.val_listeners = []
        # Training listener
        for listener_idx in range(0, self.config.listener_params.ensemble_size):
            print('Loading training listener')
            print(f'Train idx: {listener_idx}')
            listener = self._load_listener(
                listener_type=self.config.listener_params.type,
                vocab_type= 'big_clip_vocab', # self.config.model_params.vocab,
                listener_idx = listener_idx + 7 # if self.config.dataset_params.data_dir == "clip/sw" else listener_idx + 1
                )
            self.freeze_model(listener)
            self.train_listeners.append(listener)
        print(f'A training listener arch: {self.train_listeners[0]}')

        for listener_idx in range(self.config.listener_params.ensemble_size,
                                  2*self.config.listener_params.ensemble_size):
            # Val listeners
            print('Loading validation listener')
            # val_idx = (self.config.listener_params.val_idx
            #            if self.config.listener_params.val_idx else listener_idx)
            print(f'Val idx: {listener_idx}')
            listener = self._load_listener(
                listener_type=self.config.listener_params.type,
                vocab_type='big_clip_vocab', # self.config.model_params.vocab,
                listener_idx = listener_idx + 7 # if self.config.dataset_params.data_dir == "clip/sw" else listener_idx + 1
                )
            self.freeze_model(listener)
            self.val_listeners.append(listener)
        print(f'A training listener arch: {self.train_listeners[0]}')
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
                result.append(self.train_listeners[0]._tokenizer.decode(gt['input_ids'][seq_id]))  # self.train_listeners[0]._tokenizer is now CLIP's, not GPT2's
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
        # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()
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

    def construct_lang_table_with_ood_loss(self, lang, gt, lis_scores, ood_losses):
        # import pdb; pdb.set_trace()
        data = []
        text_gts = self._process_gt(gt)
        if isinstance(gt, dict):
            lang = {'input_ids': lang.argmax(-1)}
        text_langs = self._process_gt_clip(lang) # self.train_dataset.to_text(lang.argmax(-1))
        entropies = self.get_entropies(lis_scores)
        for e, ls, l, gt, ol in zip(entropies, lis_scores, text_langs, text_gts, ood_losses):
            data.append({
                'epoch': self.trainer.current_epoch,
                'entropy': e,
                'lis_scores': ls.tolist(),
                'ood_loss': ol.item(),
                'speaker_utterance': l,
                'ground_truth': gt
            })
        return pd.DataFrame(data)

    # GET LOSSES FOR BATCH WITH OOD LOSS
    def get_losses_for_batch_with_ood_loss(self, batch, batch_idx, which_listener, prefix):
        # import pdb; pdb.set_trace()

        # imgs, labels, utterances = (
        imgs_speaker, labels, utterances, imgs_clip = (
            # batch['imgs'], batch['label'].argmax(-1).long(), batch['utterance'])
            batch['imgs'].float(), batch['label'].argmax(-1).long(), batch['utterance'], batch['imgs_original'])

        lang, lang_length, loss, embedding_module = self.model(imgs_speaker, labels)

        if which_listener == "train":
            clip_scorer = self.clip_scorer(
                listener=self.clip_listener,
                imgs=imgs_clip,
                tokenizer = self.tokenizer,
                preprocess=self.preprocess,
                vocab_type=self.config.model_params.vocab,
                lang=lang,
                lang_length=lang_length,
                embedding_module=embedding_module,
                config=self.config
            )
            lis_scores = clip_scorer.listener_scores
            ood_losses = clip_scorer.ood_losses
            ood_loss_total = torch.sum(ood_losses)
            loss = nn.CrossEntropyLoss()
            lis_pred = lis_scores.argmax(1)
            losses = loss(lis_scores, labels)
            acc = (lis_pred == labels).float().mean()
        else:
            # TEMP: to get ood_losses
            clip_scorer = self.clip_scorer(
                listener=self.clip_listener,
                imgs=imgs_clip,
                tokenizer = self.tokenizer,
                preprocess=self.preprocess,
                vocab_type=self.config.model_params.vocab,
                lang=lang,
                lang_length=lang_length,
                embedding_module=embedding_module,
                config=self.config
            )
            ood_losses = clip_scorer.ood_losses
            ood_loss_total = torch.sum(ood_losses)

            # Actual val listener scorer
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

        df=self.construct_lang_table_with_ood_loss(lang=lang, gt=utterances, lis_scores=lis_scores, ood_losses=ood_losses)
        self.save_lang_table(df, batch_idx, prefix)

        # import pdb; pdb.set_trace()
        # teacher_forcing_loss, teacher_forcing_onehots = self.get_teacher_forcing_loss(utterances, imgs_speaker, labels)
        # teacher_forcing_loss = teacher_forcing_loss * 100
        
        return {
            'loss': losses,
            'acc': acc,
            'ood_loss': ood_loss_total,
        #     'teacher_forcing_loss': teacher_forcing_loss
        #     'lang_table': df
        #     'lang_table': wandb.Table(
        #         dataframe=self.construct_lang_table(lang=lang, gt=utterances)
        #     )
        }
    
    def get_teacher_forcing_loss_original(self, gt, img, targets):
        # import pdb; pdb.set_trace()
                             # listener, vocab, targets, sw):
        loss = 0.0
        # if args.teacher_forcing_loss:
        # lang = process_utterances(listener=listener, lang=lang,
        #                         vocab=vocab, sw=sw, args=args)
        # length = listener.get_length(lang)
        # lang_tokens = lang['input_ids']
        input_ids = gt['input_ids']
        gt_tokens = torch.cat((input_ids[:, :0], input_ids[:, 1:]), axis=1)
        gt_onehot = F.one_hot(gt_tokens,
                num_classes=self.model.vocab_size).cuda().float()
        # length = listener.get_length(lang)
        
        # predicted_onehots below can be fed into CLIP listener scores
        # lang_out = self.model.teacher_forcing_forward(feats=img, seq=gt_onehot,
        lang_out, tf_onehots = self.model.teacher_forcing_forward(feats=img, seq=gt_onehot,
                                        # length=length, 
                                        targets=targets)
        batch_size = lang_out.shape[0]
        # max_len = constants.MAX_SEQ_LEN + 2
        # lang_out = lang_out[:, :max_len, :].contiguous()
        lang_out = lang_out.view(batch_size*lang_out.size(1), self.model.vocab_size)
        # lang_onehot = lang_onehot[:, :max_len, :]
        gt = gt_onehot.long().view(batch_size*gt_onehot.size(1), self.model.vocab_size)
        loss_f = nn.CrossEntropyLoss()
        tf_loss = loss_f(lang_out.cuda(), torch.max(gt, 1)[1].cuda())
        return tf_loss, tf_onehots

    def get_teacher_forcing_loss(self, gt, img, targets):
        # import pdb; pdb.set_trace()
                            # listener, vocab, targets, sw):
        loss = 0.0
        # if args.teacher_forcing_loss:
        # lang = process_utterances(listener=listener, lang=lang,
        #                         vocab=vocab, sw=sw, args=args)
        # length = listener.get_length(lang)
        # lang_tokens = lang['input_ids']
        gt_tokens = gt['input_ids']
        # input_ids = gt['input_ids']
        # gt_tokens = torch.cat((input_ids[:, :0], input_ids[:, 1:]), axis=1)
        gt_onehot = F.one_hot(gt_tokens, num_classes=self.model.vocab_size).cuda().float()
        # length = listener.get_length(lang)
        
        # predicted_onehots below can be fed into CLIP listener scores
        # lang_out = self.model.teacher_forcing_forward(feats=img, seq=gt_onehot,
        # lang_out, tf_onehots = self.model.teacher_forcing_forward(feats=img, targets=targets, seq=gt_onehot)
        if self.config.training_params.tf_inputnotgt:
            lang, lang_length, eos_loss, lang_4_tfloss = self.model.teacher_forcing_forward_inputnotgt(feats=img, targets=targets, seq=gt_onehot)
        else:
            lang, lang_length, eos_loss, lang_4_tfloss = self.model.teacher_forcing_forward(feats=img, targets=targets, seq=gt_onehot)
        # batch_size = lang_out.shape[0]
        batch_size = lang_4_tfloss.shape[0]
        """if lang_4_tfloss.size(1) != 10:
            import pdb; pdb.set_trace()"""
        # max_len = constants.MAX_SEQ_LEN + 2
        # lang_out = lang_out[:, :max_len, :].contiguous()
        lang_4_tfloss = lang_4_tfloss.view(batch_size*lang_4_tfloss.size(1), self.model.vocab_size)
        # lang_onehot = lang_onehot[:, :max_len, :]
        gt_onehot = gt_onehot[:, 1:11, :]
        gt = gt_onehot.long().view(batch_size*gt_onehot.size(1), self.model.vocab_size)
        loss_f = nn.CrossEntropyLoss()
        if lang_4_tfloss.size(0) != gt.size(0):
            import pdb; pdb.set_trace()
        tf_loss = loss_f(lang_4_tfloss.cuda(), torch.max(gt, 1)[1].cuda())
        return lang, lang_length, tf_loss # TODO: return lang, lang_length, tf_loss

    def get_losses_for_batch_tf_with_ood_loss(self, batch, batch_idx, which_listener, prefix):
        # import pdb; pdb.set_trace()

        # imgs, labels, utterances = (
        imgs_speaker, labels, utterances, imgs_clip = (
            # batch['imgs'], batch['label'].argmax(-1).long(), batch['utterance'])
            batch['imgs'].float(), batch['label'].argmax(-1).long(), batch['utterance'], batch['imgs_original'])

        teacher_forcing_loss, teacher_forcing_onehots = self.get_teacher_forcing_loss(utterances, imgs_speaker, labels)
        lang = teacher_forcing_onehots

        lang_nontf, lang_length, loss, embedding_module = self.model(imgs_speaker, labels) # TEMP EDIT: jul 29

        # import pdb; pdb.set_trace()
        if which_listener == "train":
            clip_scorer = self.clip_scorer(
                listener=self.clip_listener,
                imgs=imgs_clip,
                tokenizer = self.tokenizer,
                preprocess=self.preprocess,
                vocab_type=self.config.model_params.vocab,
                lang=lang,
                lang_length=lang_length,
                embedding_module=embedding_module,
                config=self.config
            )
            ood_losses = clip_scorer.ood_losses
            ood_loss_total = torch.sum(ood_losses)

            lis_scores = clip_scorer.listener_scores
            loss = nn.CrossEntropyLoss()
            lis_pred = lis_scores.argmax(1)
            losses = loss(lis_scores, labels)
            acc = (lis_pred == labels).float().mean()
        else:
            # to get ood_losses
            clip_scorer = self.clip_scorer(
                listener=self.clip_listener,
                imgs=imgs_clip,
                tokenizer = self.tokenizer,
                preprocess=self.preprocess,
                vocab_type=self.config.model_params.vocab,
                lang=lang,
                lang_length=lang_length,
                embedding_module=embedding_module,
                config=self.config
            )
            ood_losses = clip_scorer.ood_losses
            ood_loss_total = torch.sum(ood_losses)

            # Actual val listener score
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

        df=self.construct_lang_table_with_ood_loss(lang=lang, gt=utterances, lis_scores=lis_scores, ood_losses=ood_losses)
        self.save_lang_table(df, batch_idx, prefix)

        return {
            'loss': losses,
            'acc': acc,
            'teacher_forcing_loss': teacher_forcing_loss,
            'ood_loss': ood_loss_total
        #     'lang_table': df
        #     'lang_table': wandb.Table(
        #         dataframe=self.construct_lang_table(lang=lang, gt=utterances)
        #     )
        }
    
    # TEACHER FORCING
    def get_losses_for_batch_tf(self, batch, batch_idx, which_listener, prefix):
        # import pdb; pdb.set_trace()
        # imgs, labels, utterances = (
        imgs_speaker, labels, utterances, imgs_clip = (
            # batch['imgs'], batch['label'].argmax(-1).long(), batch['utterance'])
            batch['imgs'].float(), batch['label'].argmax(-1).long(), batch['utterance'], batch['imgs_original'])

        lang, lang_length, tf_loss = self.get_teacher_forcing_loss(utterances, imgs_speaker, labels)

        # lang_nontf, lang_length_nontf, loss_nontf, embedding_module_nontf = self.model(imgs_speaker, labels) 

        # import pdb; pdb.set_trace()
        if which_listener == "train":
            clip_scorer = self.clip_scorer(
                listener=self.clip_listener,
                imgs=imgs_clip,
                tokenizer = self.tokenizer,
                preprocess=self.preprocess,
                vocab_type=self.config.model_params.vocab,
                lang=lang,
                lang_length=lang_length,
                # embedding_module=embedding_module
                config = self.config
            )
            lis_scores = clip_scorer.listener_scores
            loss = nn.CrossEntropyLoss()
            lis_pred = lis_scores.argmax(1)
            losses = loss(lis_scores, labels)
            acc = (lis_pred == labels).float().mean()
        else:
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

        df=self.construct_lang_table(lang=lang, gt=utterances, lis_scores=lis_scores)
        self.save_lang_table(df, batch_idx, prefix)

        # import pdb; pdb.set_trace()
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
        # import pdb; pdb.set_trace()

        # imgs, labels, utterances = (
        imgs_speaker, labels, utterances, imgs_clip = (
            # batch['imgs'], batch['label'].argmax(-1).long(), batch['utterance'])
            batch['imgs'].float(), batch['label'].argmax(-1).long(), batch['utterance'], batch['imgs_original'])

        # lang, lang_length, loss, embedding_module = self.model(imgs_speaker, labels)
        lang, lang_length, loss = self.model(imgs_speaker, labels)

        if which_listener == "train":
            clip_scorer = self.clip_scorer(
                listener=self.clip_listener,
                imgs=imgs_clip,
                tokenizer = self.tokenizer,
                preprocess=self.preprocess,
                vocab_type=self.config.model_params.vocab,
                lang=lang,
                lang_length=lang_length,
                # embedding_module=embedding_module,
                config = self.config
            )
            lis_scores = clip_scorer.listener_scores
            loss = nn.CrossEntropyLoss()
            lis_pred = lis_scores.argmax(1)
            losses = loss(lis_scores, labels)
            acc = (lis_pred == labels).float().mean()
        else:
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

        df=self.construct_lang_table(lang=lang, gt=utterances, lis_scores=lis_scores)
        self.save_lang_table(df, batch_idx, prefix)

        # import pdb; pdb.set_trace()
        # teacher_forcing_loss, teacher_forcing_onehots = self.get_teacher_forcing_loss(utterances, imgs_speaker, labels)
        # teacher_forcing_loss = teacher_forcing_loss * 100
        
        return {
            'pragmatic_loss': losses,
            'pragmatic_acc': acc,
        #     'teacher_forcing_loss': teacher_forcing_loss
        #     'lang_table': df
        #     'lang_table': wandb.Table(
        #         dataframe=self.construct_lang_table(lang=lang, gt=utterances)
        #     )
        }

    def save_lang_table(self, df, batch_idx, prefix):
        # import pdb; pdb.set_trace()
        # vocab_type = self.config.model_params.vocab
        # listener_type=self.config.listener_params.type
        # if vocab_type == "shapeworld":
        #     vocab = "small_vocab"
        # elif vocab_type == "gpt2":
        #     vocab = "big_vocab"
        # import pdb; pdb.set_trace()
        vocab = "clip_vocab"
        """fpath = os.path.join(
                constants.MAIN_REPO_DIR,
                "clip",
                "lang_table",
                vocab,
                # listener_type,
                self.exp_name,
                f"{prefix}-{self.trainer.current_epoch}-{batch_idx}.csv"
                )"""
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
        # import pdb; pdb.set_trace()
        if self.config.training_params.tf:
            # result = self.get_losses_for_batch_tf_with_ood_loss(batch, batch_idx, which_listener="train", prefix="train")
            result = self.get_losses_for_batch_tf(batch, batch_idx, which_listener="train", prefix="train")
            prag_loss = result['pragmatic_loss']
            tf_loss = result['teacher_forcing_loss']
            self.log_results(result=result, category="train")
            loss_final = ((prag_loss * self.config.training_params.prag_lmbd)
                            + (tf_loss * self.config.training_params.tf_lmbd))
            return loss_final

            # # L_p + L_tf:
            # total_loss = result['total_loss']
            # return total_loss
            
            # # L_tf only:
            # """tf_loss = result['teacher_forcing_loss']
            # return tf_loss"""

        if self.config.training_params.ood_loss:
            result = self.get_losses_for_batch_with_ood_loss(batch, batch_idx, which_listener="train", prefix="train")
            loss = result['loss']
            ood_loss = result['ood_loss']
            # tf_loss = result['teacher_forcing_loss']
            self.log_results(result=result, category="train")
            return ood_loss
            # return loss + ood_loss + tf_loss

        result = self.get_losses_for_batch(batch, batch_idx, which_listener="train", prefix="train")
        prag_loss = result['pragmatic_loss']
        self.log_results(result=result, category="train")
        return prag_loss

    def test_step(self, batch, batch_idx):
        result = self.get_losses_for_batch(batch, batch_idx, which_listener="test", prefix="test")
        loss = result['loss']
        # teacher_forcing_loss = result['teacher_forcing_loss']
        # self.log_results(result=result, category="test")
        # return loss + teacher_forcing_loss
        # return teacher_forcing_loss
        return loss

    def validation_step(self, batch, batch_idx):
        # import pdb; pdb.set_trace()
        
        if self.config.training_params.tf:
            for setting in ["trainL0_trainD", "trainL0_valD", "valL0_trainD", "valL0_valD"]: 
                # import pdb; pdb.set_trace()
                which_listener = "train" if "trainL0" in setting else "val"
                # result = self.get_losses_for_batch_tf_with_ood_loss(batch[setting], batch_idx, which_listener="train", prefix="train")
                result = self.get_losses_for_batch_tf(batch[setting], batch_idx, which_listener=which_listener, prefix=setting)
                prag_loss = result['pragmatic_loss']
                tf_loss = result['teacher_forcing_loss']
                # total_loss = result['total_loss']
                # ood_loss = result['ood_loss']
                self.log_results(result=result, category=setting)
                loss_final = ((prag_loss * self.config.training_params.prag_lmbd)
                            + (tf_loss * self.config.training_params.tf_lmbd))
            return loss_final
            
            # # L_p + L_tf:
            # for setting in ["trainL0_trainD", "trainL0_valD", "valL0_trainD", "valL0_valD"]: 
            #     # import pdb; pdb.set_trace()
            #     which_listener = "train" if "trainL0" in setting else "val"
            #     # result = self.get_losses_for_batch_tf_with_ood_loss(batch[setting], batch_idx, which_listener="train", prefix="train")
            #     result = self.get_losses_for_batch_tf(batch[setting], batch_idx, which_listener="train", prefix=setting)
            #     # pragmatic_loss = result['pragmatic_loss']
            #     # tf_loss = result['teacher_forcing_loss']
            #     total_loss = result['total_loss']
            #     # ood_loss = result['ood_loss']
            #     self.log_results(result=result, category=setting)
            # return total_loss

            # # L_tf only:
            # """for setting in ["trainL0_trainD", "trainL0_valD", "valL0_trainD", "valL0_valD"]: 
            #     # import pdb; pdb.set_trace()
            #     which_listener = "train" if "trainL0" in setting else "val"
            #     # result = self.get_losses_for_batch_tf_with_ood_loss(batch[setting], batch_idx, which_listener="train", prefix="train")
            #     result = self.get_losses_for_batch_tf(batch[setting], batch_idx, which_listener="train", prefix=setting)
            #     # pragmatic_loss = result['pragmatic_loss']
            #     tf_loss = result['teacher_forcing_loss']
            #     # total_loss = result['total_loss']
            #     # ood_loss = result['ood_loss']
            #     self.log_results(result=result, category=setting)
            # return tf_loss"""

        if self.config.training_params.ood_loss:
            for setting in ["trainL0_trainD", "trainL0_valD", "valL0_trainD", "valL0_valD"]:  
                # import pdb; pdb.set_trace()
                which_listener = "train" if "trainL0" in setting else "val"
                result = self.get_losses_for_batch_with_ood_loss(
                    batch[setting], batch_idx, which_listener=which_listener, prefix=setting)
                loss = result['loss']
                # tf_loss = result['teacher_forcing_loss']
                ood_loss = result['ood_loss']
                self.log_results(result=result, category=setting)
            # return loss + teacher_forcing_loss
            # return teacher_forcing_loss
            # return loss + ood_loss + tf_loss
            return ood_loss

        for setting in ["trainL0_trainD", "trainL0_valD", "valL0_trainD", "valL0_valD"]:  
            import pdb; pdb.set_trace()
            which_listener = "train" if "trainL0" in setting else "val"
            result = self.get_losses_for_batch(
                batch[setting], batch_idx, which_listener=which_listener, prefix=setting)
            prag_loss = result['pragmatic_loss']
            self.log_results(result=result, category=setting)
        return prag_loss

    def val_dataloader(self):
        # train L0 - train D
        # train L0 - val D
        # val L0 - train D
        # val L0 - val D
        loaders = {
            # CLIP (which speaker was trained with) and train dataset
            "trainL0_trainD": utils.create_dataloader(self.train_dataset, self.config, shuffle=False),
            # CLIP (which speaker was trained with) and val dataset
            "trainL0_valD": utils.create_dataloader(self.val_dataset, self.config, shuffle=False),
            "valL0_trainD": utils.create_dataloader(self.train_dataset, self.config, shuffle=False),
            "valL0_valD": utils.create_dataloader(self.val_dataset, self.config, shuffle=False),
        }
        combined_loaders = CombinedLoader(loaders, "max_size_cycle")
        return combined_loaders
