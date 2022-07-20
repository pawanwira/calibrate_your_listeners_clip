import os
import torch
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
from calibrate_your_listeners.src.datasets import generate_shapeworld_data
from calibrate_your_listeners import constants
from transformers import CLIPTokenizer, CLIPTextConfig

PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<UNK>'

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

def _init_vocab(langs):

    i2w = {
        PAD_IDX: PAD_TOKEN,
        SOS_IDX: SOS_TOKEN,
        EOS_IDX: EOS_TOKEN,
        UNK_IDX: UNK_TOKEN,
    }
    w2i = {
        PAD_TOKEN: PAD_IDX,
        SOS_TOKEN: SOS_IDX,
        EOS_TOKEN: EOS_IDX,
        UNK_TOKEN: UNK_IDX,
    }

    for lang in langs:
        for tok in lang:
            if tok not in w2i:
                i = len(w2i)
                w2i[tok] = i
                i2w[i] = tok
    return {'w2i': w2i, 'i2w': i2w}


def load_raw_data(data_file, dataset):
    data = np.load(data_file)
    # Preprocessing/tokenization
    return {
        'imgs': data['imgs'].transpose(0, 1, 4, 2, 3),  
        # 'imgs': data['imgs'],  # edit jul 9
        'labels': data['labels'],
        'langs': np.array([t.lower().split() for t in data['langs']]),
        'imgs_original': data['imgs']
    }
    # try:
    # except:
    #     return {
    #         'imgs': data['imgs'],
    #         'labels': data['labels'],
    #         'langs': data['langs']
    #     }

class Shapeworld(data.Dataset):
    """
    Adapted from: https://github.com/juliaiwhite/amortized-rsa/blob/master/data.py#L97
    """

    def __init__(self, train, config):
        """
        """
        super().__init__()
        # import pdb; pdb.set_trace()
        self.train = train
        self.config = config
        # TODO update cofig dir
        self.directory = os.path.join(
            constants.MAIN_REPO_DIR,
            self.config.dataset_params.data_dir
        )
        self.config.dataset_params.data_dir = self.directory

        # TEMP START
        self._tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self._max_seq_len = constants.MAX_SEQ_LEN
        self.clip_text_config = CLIPTextConfig()
        self._end_token = 49407
        # TEMP END

        # Data directory information
        if self.config.model_params.name == "l0":
            self.idx = self.config.model_params.listener_idx
            self.l0_filepaths = [
                os.path.join(self.directory, f'reference-1000-{f_index}.npz')
                for f_index in range(5*self.idx, 5*(self.idx+1))
            ]
        self.name = self.config.model_params.name
        self.s1_filepaths = [
            os.path.join(self.directory, f'reference-1000-{f_index}.npz') for f_index in range(60, 70) # range(135, 145)
        ]

        # Datasetloading
        self.generate_data()

        # Vocab
        self.load_vocab()
        self.w2i = self.vocab['w2i']
        self.i2w = self.vocab['i2w']

        self.load_data()
        self.imgs = self.raw_data['imgs']
        self.labels = self.raw_data['labels']
        self.imgs_original = self.raw_data['imgs_original']

    def load_vocab(self):
        vocab_fpath = os.path.join(self.directory, 'vocab.pt')
        print(f'[ config ] vocab fpath at {vocab_fpath}')
        if os.path.exists(vocab_fpath):
            vocab = torch.load(vocab_fpath)
        else:
            langs = np.array([])
            # All Shapeworld files
            f_indices = np.arange(1, 15+1) * 5 # array: 5, 10, ..., 75.
            pretrain_data = [
                [os.path.join(self.directory, "reference-1000-{}.npz".format(f_name))
                    for f_name in range(indx-5, indx)] for indx in f_indices]
            for files in pretrain_data:
                for fpath in files:
                    d = load_raw_data(fpath, dataset='shapeworld')
                    langs = np.append(langs, d['langs'])
            vocab = _init_vocab(langs)
            torch.save(vocab, vocab_fpath)
        self.vocab = vocab

    def generate_data(self):
        # Generate data if needed
        # TODO check fpaths and checks down here
        if not (os.path.isdir(self.directory) and len(os.listdir(self.directory))):
            generate_shapeworld_data.run(self.config.dataset_params)

    def load_data(self):
        if self.train and self.name == "l0":
            self.filepaths = self.l0_filepaths[:-1]
        elif not self.train and self.name == "l0":
            self.filepaths = self.l0_filepaths[-1:]
        elif self.train and self.name == "s1":
            self.filepaths = self.s1_filepaths[:5]
        elif not self.train and self.name == "s1":
            self.filepaths = self.s1_filepaths[5:]
        else:
            raise ValueError()

        print(f"Filepaths: {self.filepaths}")

        # import pdb; pdb.set_trace()
        raw_data = {"imgs": np.array([]), "labels": np.array([]), "langs": np.array([]), "imgs_original": np.array([])}
        for fpath in self.filepaths:
            d = load_raw_data(fpath, dataset=self.config.dataset_params.name)
            raw_data["imgs"] = d['imgs'] if not raw_data['imgs'].size else np.concatenate((
                raw_data['imgs'], d['imgs']), axis=0)
            raw_data["labels"] = d['labels'] if not raw_data['labels'].size else np.concatenate((
                raw_data['labels'], d['labels']), axis=0)
            raw_data["langs"] = d['langs'] if not raw_data['langs'].size else np.concatenate((
                raw_data['langs'], d['langs']), axis=0)
            raw_data["imgs_original"] = d['imgs_original'] if not raw_data['imgs_original'].size else np.concatenate((
                raw_data['imgs_original'], d['imgs_original']), axis=0)
        self.raw_data = raw_data
        self.lang_raw = self.raw_data['langs']
        # import pdb; pdb.set_trace()
        self.lang_idx, self.lang_len = self.to_idx(self.lang_raw)

    def __len__(self):
        return len(self.lang_raw)

    def __getitem__(self, i):
        # Reference game format.
        imgs = self.imgs[i]
        label = self.labels[i]
        lang = torch.Tensor(self.lang_idx[i]).type(torch.int64)
        lang = self.process_utterance(lang)
        imgs_original = self.imgs_original[i] 

        result = {"imgs": imgs, "label": label, "utterance": lang, "imgs_original": imgs_original}
        return result

    def process_utterance(self, lang):
        if self.config.model_params.vocab == "shapeworld":
            max_len = 40
            lang[lang>=len(self.vocab['w2i'].keys())] = 3
            lang = F.one_hot(lang, num_classes = len(self.vocab['w2i'].keys()))
            lang = F.pad(lang,(0,0,0,max_len-lang.shape[1])).float()
            #   for B in range(lang.shape[0]):
            #       if lang[B].sum() == 0.:
            #           lang[B][SOS_IDX] = 1.0
            #       # for L in range(lang.shape[1]):
            #       #     if lang[B][L].sum() == 0:
            #       #         # lang[B][L][0] = SOS_IDX
            #       #         lang[B][L] = SOS_IDX
        else:
            str_lang = self.to_str_text([lang])
            # import pdb; pdb.set_trace()
            # lang = self.listener_tokenize_f(str_lang) 
            lang = self.tokenize(str_lang) # temp replacement
        return lang

    # TEMP
    def tokenize(self, utterances):
        # import pdb; pdb.set_trace() # does not work
        # self._max_seq_len = max_seq_len
        """self._max_seq_len = constants.MAX_SEQ_LEN
        self._tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_text_config = CLIPTextConfig()
        self._end_token = 49407"""
        encoded_input = self._tokenizer(
            utterances,
            padding=True,
            truncation=True,
            # max_length=self._max_seq_len-1,
            max_length=self._max_seq_len + 2,
            return_tensors="pt")
        # pad
        seq_length = encoded_input['input_ids'].shape[1]
        """eos_input_ids = torch.tensor([
            self._end_token for _ in range(self._max_seq_len-seq_length)]).unsqueeze(0)
        eos_attention = torch.tensor([0 for _ in range(self._max_seq_len-seq_length)]).unsqueeze(0)"""
        eos_input_ids = torch.tensor([
            self._end_token for _ in range(self.clip_text_config.max_position_embeddings-seq_length)]).unsqueeze(0)
        eos_attention = torch.tensor([0 for _ in range(self.clip_text_config.max_position_embeddings-seq_length)]).unsqueeze(0)
        # Add an EOS token at the very end if it doesn't already exist
        # and add attention to ignore the EOS tokens
        # batch_size x 1
        # eos_input_ids = torch.tensor([self._end_token for _ in range(batch_size)]).unsqueeze(1)
        encoded_input['input_ids'] = torch.cat((encoded_input['input_ids'],
                                                eos_input_ids), dim=1)
        encoded_input['attention_mask'] = torch.cat((encoded_input['attention_mask'],
                                                eos_attention), dim=1)
        encoded_input = {k : v.squeeze(0) for k, v in encoded_input.items()}
        return encoded_input# .to(self.device)

    def to_str_text(self, idxs):
        """Omits the <sos> and <eos> tokens"""
        texts = []
        for lang in idxs:
            toks = []
            for i in lang:
                i = i.item()
                # Skip over <sos> and <eos>
                if i == self.w2i[SOS_TOKEN] or i == self.w2i[EOS_TOKEN]:
                    continue
                if i == self.w2i[PAD_TOKEN]:
                    break
                toks.append(self.i2w.get(i, UNK_TOKEN))
            texts.append(' '.join(toks))
        return texts

    def to_text(self, idxs):
        texts = []
        for lang in idxs:
            toks = []
            for i in lang:
                i = i.item()
                if i == self.w2i[PAD_TOKEN]:
                    break
                toks.append(self.i2w.get(i, UNK_TOKEN))
            texts.append(' '.join(toks))
        return texts

    def to_idx(self, langs):
        # Add SOS, EOS
        lang_len = np.array([len(t) for t in langs], dtype=np.int) + 2
        lang_idx = np.full((len(self), max(lang_len)), self.w2i[PAD_TOKEN], dtype=np.int)
        for i, toks in enumerate(langs):
            lang_idx[i, 0] = self.w2i[SOS_TOKEN]
            for j, tok in enumerate(toks, start=1):
                lang_idx[i, j] = self.w2i.get(tok, self.w2i[UNK_TOKEN])
            lang_idx[i, j + 1] = self.w2i[EOS_TOKEN]
        return lang_idx, lang_len
