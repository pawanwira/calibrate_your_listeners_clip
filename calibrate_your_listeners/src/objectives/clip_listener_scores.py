import torch
import numpy as np
from PIL import Image
import clip

class CLIPListenerScores(object):

    def __init__(self, listener, imgs, df, preprocess, vocab_type):
        self.listener = listener
        self.imgs = imgs
        self.df = df
        self.preprocess = preprocess
        self.vocab_type = vocab_type

        self.listener_scores = self._calculate_listener_scores()

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

    def _calculate_listener_scores(self):
        # import pdb; pdb.set_trace()
        lis_scores = []
        for i in range(len(self.imgs)):
            states = self.imgs[i]
            images = torch.tensor(np.stack([self._preprocess(state) for state in states])).cuda()

            """for j in range(3):
                image = Image.fromarray(np.uint8(states[j].cpu())).convert('RGB')
                images.append(self.preprocess(image))
            images_pre = torch.tensor(np.stack(images)).cuda()""" 

            utterance_tokens = self._get_utterance_tokens(i)
 
            with torch.no_grad():
                image_features = self.listener.encode_image(images).float()
                utterance_features = self.listener.encode_text(utterance_tokens).float()
            
            image_features = self.listener.encode_image(images).float()
            utterance_features = self.listener.encode_text(utterance_tokens).float()
            # image_features /= image_features.clone().norm(dim=-1, keepdim=True)
            image_features = image_features.clone() / image_features.clone().norm(dim=-1, keepdim=True)
            # utterance_features /= utterance_features.clone().norm(dim=-1, keepdim=True)
            utterance_features = utterance_features.clone() / utterance_features.clone().norm(dim=-1, keepdim=True)
            image_probs = (100.0 * utterance_features @ image_features.T).softmax(dim=-1)
            lis_scores.append(image_probs[0])

        # import pdb; pdb.set_trace()
        lis_scores_final = torch.stack(lis_scores)
        return lis_scores_final

    def get_average_l0_score(self):
        return torch.mean(self.listener_scores, axis=0) # average across listeners
