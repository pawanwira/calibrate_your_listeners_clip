import torch
import numpy as np
from PIL import Image
from torch import nn
import clip
from transformers import CLIPTextConfig

class CLIPListenerScores(object):

    def __init__(self, listener, imgs, preprocess, vocab_type, lang, lang_length):
        self.listener = listener
        self.imgs = imgs
        # self.df = df
        self.preprocess = preprocess
        self.vocab_type = vocab_type
        self.lang = lang
        self.lang_length = lang_length

        self.listener_scores = self._calculate_listener_scores()

        # create a CLIPListener class to store all these
        self.clip_text_config = CLIPTextConfig()
        """
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, self.clip_text_config.hidden_size))
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )"""

    def _preprocess(self, image):
        return self.preprocess(Image.fromarray(np.uint8(image.cpu())).convert('RGB'))

    def _get_utterance_tokens(self, i):
        if self.vocab_type == "gpt2":
            lang = self.df['speaker_utterance'][i][:-13]
        elif self.vocab_type == "shapeworld":
            lang = self.df['speaker_utterance'][i].replace("<sos>", "").replace("<eos>", "").replace("<UNK>", "").replace("<PAD>", "")
            lang = " ".join(lang.split())

        if len(lang):
            if lang[0] in ['a', 'e', 'i', 'o', 'u']:
                utterance = "This is an " + lang 
            else:
                utterance = "This is a " + lang
        else:
            utterance = " "

        utterance_tokens = clip.tokenize(utterance).cuda()
        return utterance_tokens

    """def encode_text(self, x):
        # x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x"""

    def _calculate_listener_scores(self):
        import pdb; pdb.set_trace()
        lis_scores = []
        for i in range(len(self.imgs)):
            states = self.imgs[i]
            images = torch.tensor(np.stack([self._preprocess(state) for state in states])).cuda()

            """for j in range(3):
                image = Image.fromarray(np.uint8(states[j].cpu())).convert('RGB')
                images.append(self.preprocess(image))
            images_pre = torch.tensor(np.stack(images)).cuda()""" 

            # utterance_tokens = self._get_utterance_tokens(i)
            utterance_tokens = self.lang[i]

            """with torch.no_grad():
                image_features = self.listener.encode_image(images).float()
                utterance_features = self.listener.encode_text(utterance_tokens).float()"""
            
            image_features = self.listener.encode_image(images).float()
            utterance_features = self.listener.encode_text(utterance_tokens).float()
            # utterance_features = self.encode_text(self.lang[i]).float() # clip update
            # image_features /= image_features.clone().norm(dim=-1, keepdim=True)
            image_features = image_features.clone() / image_features.clone().norm(dim=-1, keepdim=True)
            # utterance_features /= utterance_features.clone().norm(dim=-1, keepdim=True)
            utterance_features = utterance_features.clone() / utterance_features.clone().norm(dim=-1, keepdim=True)
            image_probs = (100.0 * utterance_features @ image_features.T).softmax(dim=-1)
            lis_scores.append(image_probs[0])

        import pdb; pdb.set_trace()
        lis_scores_final = torch.stack(lis_scores)
        return lis_scores_final

    # def get_average_l0_score(self):
    #     return torch.mean(self.listener_scores, axis=0) # average across listeners
