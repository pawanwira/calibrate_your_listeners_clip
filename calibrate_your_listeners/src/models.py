import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn_utils
from transformers import AlbertConfig, AlbertTokenizer, AlbertModel
from transformers import AutoTokenizer, AutoModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model

from calibrate_your_listeners import constants

# import data
# import vision

from calibrate_your_listeners.src.models_temp import vision

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2

MAX_SEQ_LEN=10
EPS=1e-5

# L0_seen_vocab = torch.load("data/L0_seen_tokens.pt")
# L0_not_seen_vocab = torch.load("data/L0_not_seen_tokens.pt")
# sorted_L0_seen_vocab = torch.load("data/sorted_L0_seen_tokens.pt")
# sorted_L0_not_seen_vocab = torch.load("data/sorted_L0_not_seen_tokens.pt")
# L0_seen_vocab = torch.load("data/L0_10x_seen.pt")
# sorted_L0_seen_vocab = torch.load("data/sorted_L0_10x_seen.pt")

# class FeatureMLP(nn.Module):
#     def __init__(self, input_size=16, output_size=16):
#         super(FeatureMLP, self).__init__()
#         self.trunk = nn.Sequential(
#             nn.Linear(input_size, output_size),
#             nn.ReLU(),
#             nn.Linear(output_size, output_size)
#         )
#         self.input_size = input_size
#         self.output_size = output_size

#     def forward(self, x):
#         return self.trunk(x)


# def to_onehot(y, n=3):
#     y_onehot = torch.zeros(y.shape[0], n).to(y.device)
#     y_onehot.scatter_(1, y.view(-1, 1), 1)
#     return y_onehot


# class Speaker(nn.Module):
#     def __init__(self, feat_model, embedding_module, with_embed, use_seen_vocab,
#                  is_old, tokenizer=None, hidden_size=100, max_seq_len=MAX_SEQ_LEN):
#         """Initializing the rational speaker.

#         :param feat_model: f_S, the CNN encoder to map images (3, img_width, img_height)
#             to an encoded representation.
#         :param embedding_module:

#         """
#         super(Speaker, self).__init__()
#         self._is_old = is_old
#         self.feat_model = feat_model
#         self.embedding = embedding_module

#         self.feat_size = feat_model.final_feat_dim
#         self.embedding_dim = embedding_module.embedding_dim
#         self.vocab_size = embedding_module.num_embeddings
#         self.hidden_size = hidden_size

#         self.gru = nn.GRU(self.embedding_dim, self.hidden_size)
#         self._use_seen_vocab = use_seen_vocab

#         self._tokenizer = tokenizer
#         if self._use_seen_vocab:
#             self.outputs2seenVocab = nn.Linear(self.hidden_size, len(L0_seen_vocab))
#             # translation matrix from seen vocabulary to all vocaulbary
#             self.seenVocab2allVocab = torch.zeros(len(L0_seen_vocab), self.vocab_size)
#             self._set_seen2allVocab()
#         else:
#             self.outputs2vocab = nn.Linear(self.hidden_size, self.vocab_size)
#         self._set_tokens()
#         # n_obj of feature size + 1/0 indicating target index
#         self.init_h = nn.Linear(3 * (self.feat_size + 1), self.hidden_size)
#         self._max_seq_len = max_seq_len
#         self._with_embed = with_embed

#     def _set_seen2allVocab(self):
#         # map the index location of seenVocab to it's value index in matrix
#         for seenVocabIndx, allVocabIndx in enumerate(sorted_L0_seen_vocab):
#             self.seenVocab2allVocab[seenVocabIndx, allVocabIndx] = 1.0
#         self.seenVocab2allVocab = self.seenVocab2allVocab.cuda()

#     def _set_tokens(self):
#         if self._tokenizer is None:
#             self._start_token = data.SOS_IDX
#             self._end_token = data.EOS_IDX
#         elif self._tokenizer.name_or_path == "gpt2":
#             # Adding padding token to GPT tokenizer
#             self._tokenizer.pad_token = self._tokenizer.eos_token
#             self._start_token = None
#             self._end_token = self._tokenizer.eos_token_id
#         else:
#             raise ValueError("don't recognize tokenizer {}".format(self._tokenizer))

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self._torch_end_token = torch.tensor(self._end_token).to(device)

#     def embed_features(self, feats, targets):
#         batch_size = feats.shape[0]
#         if self._with_embed:
#             feats_emb = feats
#         else:
#             n_obj = feats.shape[1]
#             rest = feats.shape[2:]
#             feats_flat = feats.view(batch_size * n_obj, *rest)
#             feats_emb_flat = self.feat_model(feats_flat)
#             feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)
#         # Add targets
#         targets_onehot = to_onehot(targets)
#         feats_and_targets = torch.cat((feats_emb, targets_onehot.unsqueeze(2)), 2)
#         ft_concat = feats_and_targets.view(batch_size, -1)
#         return ft_concat

#     def forward(self, feats, targets,
#             greedy=False, activation='gumbel',
#             tau=1, length_penalty=False):
#         batch_size = feats.size(0)
#         feats_emb = self.embed_features(feats, targets)

#         # initialize hidden states using image features
#         states = self.init_h(feats_emb)
#         states = states.unsqueeze(0)

#         # This contains are series of sampled onehot vectors
#         lang = []
#         lang_prob = None
#         if length_penalty:
#             eos_prob = []

#         # And vector lengths
#         lang_length = np.ones(batch_size, dtype=np.int64)
#         done_sampling = np.array([False for _ in range(batch_size)])

#         # first input is SOS token
#         # (batch_size, n_vocab)
#         inputs_onehot = torch.zeros(batch_size, self.vocab_size, device=feats.device)
#         if self._is_old:
#             inputs_onehot[:, data.SOS_IDX] = 1.0

#         # No start token for GPT - leave inputs as onehot

#         # (batch_size, len, n_vocab)
#         inputs_onehot = inputs_onehot.unsqueeze(1)

#         # Add SOS to lang
#         lang.append(inputs_onehot)

#         # (B,L,D) to (L,B,D)
#         inputs_onehot = inputs_onehot.transpose(0, 1)

#         # compute embeddings
#         # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
#         inputs = inputs_onehot @ self.embedding.weight
#         if self._is_old:
#             max_len = self._max_seq_len - 2
#         else:
#             max_len = self._max_seq_len - 1

#         for i in range(max_len):  # Have room for EOS if never sampled
#             # FIXME: This is inefficient since I do sampling even if we've
#             # finished generating language.
#             if all(done_sampling):
#                 break

#             outputs, states = self.gru(inputs, states)  # outputs: (L=1,B,H)
#             outputs = outputs.squeeze(0)                # outputs: (B,H)

#             if self._use_seen_vocab:
#                 outputs = self.outputs2seenVocab(outputs)       # outputs: (B,V')
#             else:
#                 outputs = self.outputs2vocab(outputs)       # outputs: (B,V)

#             if activation=='gumbel'or activation==None:
#                 predicted_onehot = F.gumbel_softmax(outputs, tau=tau, hard=True)
#             elif activation=='softmax':
#                 predicted_onehot = F.softmax(outputs/tau)
#             elif activation=='softmax_noise':
#                 predicted_onehot = F.gumbel_softmax(outputs, tau=tau, hard=False)
#             else:
#                 raise NotImplementedError(activation)

#             if self._use_seen_vocab:
#                 # translate from V' to V
#                 predicted_onehot = torch.matmul(predicted_onehot, self.seenVocab2allVocab)

#             # Add to lang
#             lang.append(predicted_onehot.unsqueeze(1))
#             if length_penalty:
#                 idx_prob = F.log_softmax(outputs, dim = 1)
#                 eos_prob.append(idx_prob[:, self._end_token])

#             # Update language lengths
#             lang_length += ~done_sampling
#             done_sampling = np.logical_or(
#                 done_sampling,
#                 (predicted_onehot[:, self._end_token] == 1.0).cpu().numpy())
#             # assert activation in {'gumbel', 'multinomial'}, "check activation either gumbel or multinom"

#             # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
#             inputs = (predicted_onehot.unsqueeze(0)) @ self.embedding.weight

#         # Add EOS if we've never sampled it
#         eos_onehot = torch.zeros(batch_size, 1, self.vocab_size, device=feats.device)
#         eos_onehot[:, 0, self._end_token] = 1.0
#         lang.append(eos_onehot)

#         # Cut off the rest of the sentences
#         lang_length += (~done_sampling)

#         # Cat language tensors (batch_size, max_seq_length, vocab_size)
#         # skip first element b/c it's just 0s
#         # no SOS token for GPT
#         if self._is_old:
#             lang_tensor = torch.cat(lang, 1)
#         else:
#             lang_tensor = torch.cat(lang[1:], 1)
#             lang_length -= 1

#         for i in range(lang_tensor.shape[0]):
#             lang_tensor[i, lang_length[i]:] = 0

#         # Trim max length
#         max_lang_len = lang_length.max()
#         lang_tensor = lang_tensor[:, :max_lang_len, :]

#         if length_penalty:
#             # eos prob -> eos loss
#             eos_prob = torch.stack(eos_prob, dim = 1)
#             for i in range(eos_prob.shape[0]):
#                 r_len = torch.arange(1,eos_prob.shape[1]+1,dtype=torch.float32)
#                 eos_prob[i] = eos_prob[i]*r_len.to(eos_prob.device)
#                 eos_prob[i, lang_length[i]:] = 0
#             eos_loss = -eos_prob
#             eos_loss = eos_loss.sum(1)/torch.tensor(
#                 lang_length,dtype=torch.float32, device=eos_loss.device)
#             eos_loss = eos_loss.mean()
#         else:
#             eos_loss = 0

#         # Sum up log probabilities of samples
#         return lang_tensor, lang_length, eos_loss, lang_prob

#     def teacher_forcing_forward(self, feats, seq, length, y):
#         batch_size = seq.shape[0]
#         feats_emb = self.embed_features(feats=feats, targets=y.long())

#         # reorder from (B,L,D) to (L,B,D)
#         seq = seq.transpose(0, 1).to(feats.device)

#         # embed your sequences
#         embed_seq = seq @ self.embedding.weight

#         feats_emb = self.init_h(feats_emb)
#         feats_emb = feats_emb.unsqueeze(0)

#         # shape = (seq_len, batch, hidden_dim)
#         output, _ = self.gru(embed_seq, feats_emb)

#         # reorder from (L,B,D) to (B,L,D)
#         # (batch_size, max_sequence_length, hidden unit size)
#         output = output.transpose(0, 1)

#         max_length = output.size(1)
#         output_2d = output.reshape(batch_size * max_length, -1)
#         outputs_2d = self.outputs2vocab(output_2d)
#         # Distribution over vocab for each batch, (batch_size, max_seq_length, vocab_size)
#         lang_tensor = outputs_2d.reshape(batch_size, max_length, self.vocab_size)
#         return lang_tensor

#     # def forward(self, feats, targets, greedy=False, activation='gumbel', tau = 1, length_penalty=False):
#     #     """Generates language given images and target image.

#     #     :param feats: tensor of shape (batch_size, num_imgs in reference game,
#     #         num_img_channels, img_width, img_height).
#     #         Represents the images in each reference game, I_t where
#     #         0 <= t < num_imgs in reference game.
#     #     :param targets: Tensor of length batch_size, represents target reference image.

#     #     """
#     #     batch_size = feats.size(0)

#     #     feats_emb = self.embed_features(feats, targets)

#     #     # initialize hidden states using image features
#     #     states = self.init_h(feats_emb)
#     #     states = states.unsqueeze(0)

#     #     # This contains are series of sampled onehot vectors
#     #     lang = []
#     #     if length_penalty:
#     #         eos_prob = []

#     #     if activation == 'multinomial':
#     #         lang_prob = []
#     #     else:
#     #         lang_prob = None

#     #     # And vector lengths
#     #     lang_length = np.ones(batch_size, dtype=np.int64)
#     #     done_sampling = np.array([False for _ in range(batch_size)])

#     #     # first input is SOS token
#     #     # (batch_size, n_vocab)
#     #     inputs_onehot = torch.zeros(batch_size, self.vocab_size, device=feats.device)

#     #     if self._start_token is not None: # GPT2 doesn't have a start token
#     #         inputs_onehot[:, self._start_token] = 1.0

#     #     # (batch_size, len, n_vocab)
#     #     inputs_onehot = inputs_onehot.unsqueeze(1)

#     #     # Add SOS to lang
#     #     lang.append(inputs_onehot)

#     #     # (B,L,D) to (L,B,D)
#     #     inputs_onehot = inputs_onehot.transpose(0, 1)

#     #     # compute embeddings
#     #     # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
#     #     inputs = inputs_onehot @ self.embedding.weight

#     #     for i in range(self._max_seq_len - 2):  # Have room for SOS, EOS if never sampled
#     #         # FIXME: This is inefficient since I do sampling even if we've
#     #         # finished generating language.
#     #         if all(done_sampling):
#     #             break
#     #         # self.gru.flatten_parameters()
#     #         outputs, states = self.gru(inputs, states)  # outputs: (L=1,B,H)
#     #         outputs = outputs.squeeze(0)                # outputs: (B,H)
#     #         outputs = self.outputs2vocab(outputs)       # outputs: (B,V)

#     #         if greedy:
#     #             predicted = outputs.max(1)[1]
#     #             predicted = predicted.unsqueeze(1)
#     #         else:
#     #             #  outputs = F.softmax(outputs, dim=1)
#     #             #  predicted = torch.multinomial(outputs, 1)
#     #             # TODO: Need to let language model accept one-hot vectors.
#     #             if activation=='gumbel'or activation==None:
#     #                 predicted_onehot = F.gumbel_softmax(outputs, tau=tau, hard=True)
#     #             elif activation=='softmax':
#     #                 predicted_onehot = F.softmax(outputs/tau)
#     #             elif activation=='softmax_noise':
#     #                 predicted_onehot = F.gumbel_softmax(outputs, tau=tau, hard=False)
#     #             elif activation == 'multinomial':
#     #                 # Normal non-differentiable sampling from the RNN, trained with REINFORCE
#     #                 TEMP = 5.0
#     #                 idx_prob = F.log_softmax(outputs / TEMP, dim=1)
#     #                 predicted = torch.multinomial(idx_prob.exp(), 1)
#     #                 predicted_onehot = to_onehot(predicted, n=self.vocab_size)
#     #                 predicted_logprob = torch.gather(idx_prob, 1, predicted)
#     #                 lang_prob.append(predicted_logprob)
#     #             else:
#     #                 raise NotImplementedError(activation)

#     #             # Add to lang
#     #             lang.append(predicted_onehot.unsqueeze(1))
#     #             if length_penalty:
#     #                 idx_prob = F.log_softmax(outputs, dim = 1)
#     #                 eos_prob.append(idx_prob[:, self._end_token])

#     #         # Update language lengths
#     #         lang_length += ~done_sampling
#     #         done_sampling = np.logical_or(
#     #             done_sampling,
#     #             (predicted_onehot[:, self._end_token] == 1.0).cpu().numpy())
#     #         # assert activation in {'gumbel', 'multinomial'}, "check activation either gumbel or multinom"

#     #         # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
#     #         inputs = (predicted_onehot.unsqueeze(0)) @ self.embedding.weight

#     #     # If multinomial, we need to run inputs once more to get the logprob of
#     #     # EOS (in case we've sampled that far)
#     #     if activation == 'multinomial':
#     #         self.gru.flatten_parameters()
#     #         outputs, states = self.gru(inputs, states)  # outputs: (L=1,B,H)
#     #         outputs = outputs.squeeze(0)                # outputs: (B,H)
#     #         outputs = self.outputs2vocab(outputs)       # outputs: (B,V)
#     #         idx_prob = F.log_softmax(outputs, dim=1)
#     #         lang_prob.append(idx_prob[:, self._end_token].unsqueeze(1))

#     #     # Add EOS if we've never sampled it
#     #     eos_onehot = torch.zeros(batch_size, 1, self.vocab_size, device=feats.device)
#     #     eos_onehot[:, 0, self._end_token] = 1.0
#     #     lang.append(eos_onehot)

#     #     # Cut off the rest of the sentences
#     #     lang_length += (~done_sampling)
#     #     # lang_length[done_sampling != 1] += 1

#     #     # Cat language tensors (batch_size, max_seq_length, vocab_size)
#     #     lang_tensor = torch.cat(lang, 1)

#     #     for i in range(lang_tensor.shape[0]):
#     #         lang_tensor[i, lang_length[i]:] = 0

#     #     # Trim max length
#     #     max_lang_len = lang_length.max()
#     #     lang_tensor = lang_tensor[:, :max_lang_len, :]

#     #     if activation == 'multinomial':
#     #         lang_prob_tensor = torch.cat(lang_prob, 1)
#     #         for i in range(lang_prob_tensor.shape[0]):
#     #             lang_prob_tensor[i, lang_length[i]:] = 0
#     #         lang_prob_tensor = lang_prob_tensor[:, :max_lang_len]
#     #         lang_prob = lang_prob_tensor.sum(1)

#     #     if length_penalty:
#     #         # eos prob -> eos loss
#     #         eos_prob = torch.stack(eos_prob, dim = 1)
#     #         for i in range(eos_prob.shape[0]):
#     #             r_len = torch.arange(1,eos_prob.shape[1]+1,dtype=torch.float32)
#     #             eos_prob[i] = eos_prob[i]*r_len.to(eos_prob.device)
#     #             eos_prob[i, lang_length[i]:] = 0
#     #         eos_loss = -eos_prob
#     #         eos_loss = eos_loss.sum(1)/torch.tensor(
#     #             lang_length,dtype=torch.float32, device=eos_loss.device)
#     #         eos_loss = eos_loss.mean()
#     #     else:
#     #         eos_loss = 0

#     #     # Sum up log probabilities of samples
#     #     return lang_tensor, lang_length, eos_loss, lang_prob

#     def to_text(self, lang_onehot):
#         texts = []
#         lang = lang_onehot.argmax(2)
#         for sample in lang.cpu().numpy():
#             text = []
#             for item in sample:
#                 text.append(data.ITOS[item])
#                 # if item == data.EOS_IDX:
#                 if item == self._end_token:
#                     break
#             texts.append(' '.join(text))
#         return np.array(texts, dtype=np.unicode_)


# class LiteralSpeaker(nn.Module):
#     def __init__(self, feat_model, embedding_module, with_embed,
#             hidden_size=100, max_seq_len=MAX_SEQ_LEN, contextual=True,
#                  use_seen_vocab=False):
#         super(LiteralSpeaker, self).__init__()
#         self.contextual = contextual

#         self.embedding = embedding_module
#         self.embedding_dim = embedding_module.embedding_dim
#         self.vocab_size = embedding_module.num_embeddings
#         self.hidden_size = hidden_size
#         self.gru = nn.GRU(self.embedding_dim, self.hidden_size)

#         self.feat_model = feat_model # f_S(I_t)
#         self.feat_size = feat_model.final_feat_dim
#         self.outputs2vocab = nn.Linear(self.hidden_size, self.vocab_size)
#         if self.contextual:
#             # feat_size + 1 because we have a one-hot encoding for whether the
#             # image is the target reference.
#             self.init_h = nn.Linear(3 * (self.feat_size + 1), self.hidden_size)
#         else:
#             self.init_h = nn.Linear(self.feat_size, self.hidden_size)

#         self._max_seq_len = max_seq_len
#         self._with_embed = with_embed

#     def _get_batch_size(self, seq):
#         return seq.shape[0]

#     def embed_feats(self, feats, y):
#         if self.contextual:
#             if self._with_embed:
#                 feats_emb = feats
#             else:
#                 n_obj = feats.shape[1]
#                 rest = feats.shape[2:]
#                 feats = feats.view(batch_size * n_obj, *rest)
#                 feats_emb_flat = self.feat_model(feats)
#                 feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)
#             # Add targets
#             targets_onehot = to_onehot(y)
#             feats_and_targets = torch.cat((feats_emb, targets_onehot.unsqueeze(2)), 2)
#             feats_emb = feats_and_targets.view(batch_size, -1)
#         else:
#             if self._with_embed:
#                 feats_emb = feats[:, 0, :]
#             else:
#                 feats = torch.from_numpy(
#                     np.array([np.array(feat[y[idx],:,:,:].cpu())
#                               for idx, feat in enumerate(feats)])).cuda()
#                 feats_emb = self.feat_model(feats.cuda())
#         return feats_emb

#     def forward(self, feats, seq, length, y):
#         """Calculates S_0(u|I, t) = p_{RNN-IC}(u |f_s(:)

#         :param feats: tensor of shape (batch_size, num_imgs in reference game,
#             num_img_channels, img_width, img_height).
#             Represents the images in each reference game, I_t where
#             0 <= t < num_imgs in reference game.
#         :param seq: Tensor of size (max_seq_length, batch_size, vocab_size).
#             max_sequence_lenth is typically 40.
#             batch_size is typically 32.
#             vocab_size is typically 1201 for colors.
#             Represents the observed utterances. The literal speaker is trained
#             as an IC model.
#         :param length: Tensor of length batch_size. Specifies the length
#             of each sequence in the batch, ie. elements are <= max_seq_length.
#         :param y: Tensor of length batch_size, represents target reference image.

#         :returns:
#         """
#         batch_size = self._get_batch_size(seq)
#         feats_emb = self.embed_feats(feats=feats, y=y)

#         # reorder from (B,L,D) to (L,B,D)
#         seq = seq.transpose(0, 1).to(feats.device)

#         # embed your sequences
#         embed_seq = seq @ self.embedding.weight

#         feats_emb = self.init_h(feats_emb)
#         feats_emb = feats_emb.unsqueeze(0)

#         # shape = (seq_len, batch, hidden_dim)
#         output, _ = self.gru(embed_seq, feats_emb)

#         # reorder from (L,B,D) to (B,L,D)
#         # (batch_size, max_sequence_length, hidden unit size)
#         output = output.transpose(0, 1)

#         max_length = output.size(1)
#         output_2d = output.reshape(batch_size * max_length, -1)
#         outputs_2d = self.outputs2vocab(output_2d)
#         # Distribution over vocab for each batch, (batch_size, max_seq_length, vocab_size)
#         lang_tensor = outputs_2d.reshape(batch_size, max_length, self.vocab_size)
#         return lang_tensor

#     def sample_with_semantics(self, feats, y, listeners, tau=1.0,
#                               return_all_lis_scores=False):
#         """
#         return lang output and loss by listener semantics.
#         """
#         batch_size = feats.size(0)
#         feats_emb = self.embed_feats(feats=feats, y=y) # noncontextual
#         # initialize hidden states using image features
#         states = self.init_h(feats_emb)
#         states = states.unsqueeze(0)
#         # This contains are series of sampled onehot vectors
#         lang = []
#         # And vector lengths
#         lang_length = np.zeros(batch_size, dtype=np.int64)
#         done_sampling = np.array([False for _ in range(batch_size)])
#         # first input is SOS token
#         # (batch_size, n_vocab)
#         inputs_onehot = torch.zeros(batch_size, self.vocab_size, device=feats.device)

#         if listeners[0]._start_token is not None: # GPT2 doesn't have a start token
#             inputs_onehot[:, listeners[0]._start_token] = 1.0
#             # (batch_size, len, n_vocab)
#             inputs_onehot = inputs_onehot.unsqueeze(1)
#             # Add SOS to lang
#             lang.append(inputs_onehot)
#             max_seq_len = self._max_seq_len - 2
#         else: # if GPT, don't add to lang.
#             # (batch_size, len, n_vocab)
#             inputs_onehot = inputs_onehot.unsqueeze(1)
#             max_seq_len = self._max_seq_len - 1

#         # (B,L,D) to (L,B,D)
#         inputs_onehot = inputs_onehot.transpose(0, 1)
#         # compute embeddings
#         # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
#         inputs = inputs_onehot @ self.embedding.weight

#         for i in range(max_seq_len):  # Have room for SOS, EOS if never sampled
#             if all(done_sampling):
#                 break
#             outputs, states = self.gru(inputs, states)  # outputs: (L=1,B,H)
#             outputs = outputs.squeeze(0)                # outputs: (B,H)
#             outputs = self.outputs2vocab(outputs)       # outputs: (B,V)

#             predicted_onehot = F.gumbel_softmax(outputs, tau=tau, hard=True)
#             lang.append(predicted_onehot.unsqueeze(1))

#             # Update language lengths
#             lang_length += ~done_sampling
#             done_sampling = np.logical_or(
#                 done_sampling,
#                 (predicted_onehot[:, listeners[0]._end_token] == 1.0).cpu().numpy())
#             # assert activation in {'gumbel', 'multinomial'}, "check activation either gumbel or multinom"

#             # (1, batch_size, n_vocab) X (n_vocab, h) -> (1, batch_size, h)
#             inputs = (predicted_onehot.unsqueeze(0)) @ self.embedding.weight

#         # Add EOS if we've never sampled it
#         eos_onehot = torch.zeros(batch_size, 1, self.vocab_size, device=feats.device)
#         eos_onehot[:, 0, listeners[0]._end_token] = 1.0
#         lang.append(eos_onehot)

#         # # Cut off the rest of the sentences
#         lang_length += (~done_sampling)
#         # Cat language tensors (batch_size, max_seq_length, vocab_size)
#         lang_tensor = torch.cat(lang, 1)

#         for i in range(lang_tensor.shape[0]):
#             lang_tensor[i, lang_length[i]:] = 0

#         # Trim max length
#         max_lang_len = lang_length.max()

#         lang_tensor = lang_tensor[:, :max_lang_len, :]

#         lang_length -= 1 # to be used to index the last hidden state
#         lang_length = torch.tensor(lang_length, device=lang_tensor.device)

#         lis_scores = 0.0
#         for listener in listeners:
#             lis_scores += listener(feats, lang_tensor, lang_length,
#                                   used_as_internal_listener=True) * (1./len(listeners))

#         if not return_all_lis_scores:
#             lis_scores = lis_scores[:, 0]

#         # Sum up log probabilities of samples
#         return lang_tensor, lang_length, lis_scores

#     def sample(self, feats, y, greedy=False):
#         """Generate from image features using greedy search."""

#         with torch.no_grad():
#             batch_size = feats.shape[0]
#             feats_emb = self.embed_features(feats=feats, y=y) # noncontextual

#             # initialize hidden states using image features
#             feats_emb = self.init_h(feats_emb)
#             states = feats_emb.unsqueeze(0)

#             # first input is SOS token
#             inputs = np.array([SOS_IDX for _ in range(batch_size)])
#             inputs = torch.from_numpy(inputs, device=feats.device)
#             inputs = inputs.unsqueeze(1)
#             # inputs = inputs.to(feats.device)
#             inputs = F.one_hot(inputs, num_classes=self.vocab_size).float()

#             # save SOS as first generated token
#             inputs_npy = inputs.squeeze(1).cpu().numpy()
#             sampled = np.array([[w] for w in inputs_npy])
#             sampled = np.transpose(sampled, (1, 0, 2))

#             # (B,L,D) to (L,B,D)
#             inputs = inputs.transpose(0,1).to(feats.device)

#             # compute embeddings
#             inputs = inputs @ self.embedding.weight

#             for i in range(self._max_seq_len-1):
#                 self.gru.flatten_parameters()
#                 outputs, states = self.gru(inputs, states)  # outputs: (L=1,B,H)
#                 outputs = outputs.squeeze(0)                # outputs: (B,H)
#                 outputs = self.outputs2vocab(outputs)       # outputs: (B,V)

#                 if greedy:
#                     predicted = outputs.max(1)[1].cpu()
#                     predicted = predicted.unsqueeze(1)
#                 else:
#                     outputs = F.softmax(outputs, dim=1)
#                     predicted = torch.multinomial(outputs.cpu(), 1)

#                 predicted = predicted.transpose(0, 1)        # inputs: (L=1,B)
#                 predicted = F.one_hot(predicted, num_classes=self.vocab_size).float()
#                 inputs = predicted.to(feats.device) @ self.embedding.weight             # inputs: (L=1,B,E)

#                 sampled = np.concatenate((sampled,predicted),axis = 0)

#             sampled = torch.tensor(sampled).permute(1,0,2)

#             sampled_id = sampled.reshape(sampled.shape[0]*sampled.shape[1],-1).argmax(1).reshape(sampled.shape[0],sampled.shape[1])
#             sampled_lengths = torch.tensor([np.count_nonzero(t) for t in sampled_id.cpu()], dtype=np.int)
#         return sampled, sampled_lengths

# class DeepLanguageModel(nn.Module):
#     def __init__(self):
#         super(DeepLanguageModel, self).__init__()
#         from transformers import BertTokenizer, AlbertForMaskedLM
#         self._tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
#         self._model = AlbertForMaskedLM.from_pretrained('albert-base-v2') # hidden state = 768
#         # from transformers import BertTokenizer, BertForMaskedLM

#         # self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#         # self._model = BertForMaskedLM.from_pretrained('bert-base-uncased')
#         self._model.eval()
#         self._probs = torch.nn.Softmax(dim=1)

#     def probability(self, lang, lang_length, with_lm_penalty):
#         """Iteratively mask..

#         lang has shape: (batch_size, max_sequence_length, vocab_size)
#             lang is in a one-hot float representation
#         """
#         if not with_lm_penalty:
#             return 0.0
#         # sum of log probs
#         seq_probs = torch.zeros(lang.shape[0]).cuda()
#         seq_length = lang.shape[1]
#         start = time.time()
#         for mask_indx in range(seq_length):
#             start = time.time()
#             # Mask all indices from mask_indx -> to end of sequence.
#             masked = torch.zeros(lang.shape)
#             masked[:, :mask_indx, :] = lang[:, :mask_indx, :] # Keep all relevant utterances
#             masked[:, mask_indx:, self._tokenizer.mask_token_id] = 1. # Set everything else to [MASK]
#             # Get word embeddings for the sequence; (batch_size, max_seq_len, 128)
#             embedding = masked @ self._model.albert.embeddings.word_embeddings.weight
#             # Encode embeddings; (batch_size, max_seq_len, 768)
#             encoded = self._model.albert(inputs_embeds=embedding)[0]
#             # Now get token predictions! (batch_size, max_seq_len, vocab_size)
#             logit_preds = self._model.predictions(encoded)
#             # Get predictions for this specific mask_indx; (batch_size, vocab_size)
#             mask_logit_preds = logit_preds[:, mask_indx, :]
#             # Get probabilities; same shape as above.
#             probs = self._probs(mask_logit_preds)
#             # Get the actual utterance word probs
#             utterance_idxs = lang[:, mask_indx:mask_indx+1, :].argmax(2)
#             # Apply (0) if mask_indx > lang_length
#             discard_prob_mask = mask_indx > lang_length
#             log_probs = torch.log(probs.cuda().gather(1, utterance_idxs.cuda())).squeeze(1)
#             log_probs[discard_prob_mask] = 0.0
#             seq_probs += log_probs
#         return -seq_probs.mean()

# class BERTLM(nn.Module):

#     def __init__(self):
#         super(BERTLM, self).__init__()
#         self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#         self._model = GPT2LMHeadModel.from_pretrained('gpt2')

#     def probability(self, lang, lang_length, with_lm_penalty):
#         if not with_lm_penalty:
#             return 0.0
#         import pdb; pdb.set_trace()

#         # encoded = self._model.albert(inputs_embeds=embedding)[0]

# class DummyLanguageModel(nn.Module):
#     def __init__(self):
#         super(DummyLanguageModel, self).__init__()
#     def probability(self, seq, length):
#         return 0.0

# class LanguageModel(nn.Module):
#     def __init__(self, embedding_module, hidden_size=100,
#             max_seq_len=MAX_SEQ_LEN):
#         super(LanguageModel, self).__init__()
#         self.embedding = embedding_module
#         self.embedding_dim = embedding_module.embedding_dim
#         self.vocab_size = embedding_module.num_embeddings
#         self.hidden_size = hidden_size
#         self.gru = nn.GRU(self.embedding_dim, self.hidden_size)
#         self.outputs2vocab = nn.Linear(self.hidden_size, self.vocab_size)
#         self._max_seq_len = max_seq_len

#     def forward(self, seq, length):
#         batch_size = seq.shape[0]
#         if batch_size > 1:
#             sorted_lengths, sorted_idx = torch.sort(length, descending=True)
#             seq = seq[sorted_idx]

#         # reorder from (B,L,D) to (L,B,D)
#         seq = seq.transpose(0, 1)

#         # embed your sequences
#         embed_seq = seq.cuda() @ self.embedding.weight

#         # shape = (seq_len, batch, hidden_dim)
#         feats_emb = torch.zeros(1, batch_size, self.hidden_size).to(embed_seq.device)
#         self.gru.flatten_parameters()
#         output, _ = self.gru(embed_seq, feats_emb)

#         # reorder from (L,B,D) to (B,L,D)
#         output = output.transpose(0, 1)

#         if batch_size > 1:
#             _, reversed_idx = torch.sort(sorted_idx)
#             output = output[reversed_idx]

#         max_length = output.size(1)
#         output_2d = output.view(batch_size * max_length, -1)
#         outputs_2d = self.outputs2vocab(output_2d)
#         lang_tensor = outputs_2d.view(batch_size, max_length, self.vocab_size)
#         return lang_tensor

#     def probability(self, seq, length):
#         with torch.no_grad():
#             batch_size = seq.shape[0]
#             seq = F.pad(seq,(0,0,0,(self._max_seq_len-seq.shape[1]))).float()
#             # reorder from (B,L,D) to (L,B,D)
#             seq = seq.transpose(0, 1)
#             # embed your sequences
#             embed_seq = seq.cuda() @ self.embedding.weight

#             # shape = (seq_len, batch, hidden_dim)
#             feats_emb = torch.zeros(1, batch_size, self.hidden_size).to(embed_seq.device)

#             inputs = embed_seq
#             states = feats_emb
#             prob = torch.zeros(batch_size)

#             self.gru.flatten_parameters()
#             outputs, _ = self.gru(inputs, states)
#             outputs = outputs.squeeze(0)
#             outputs = self.outputs2vocab(outputs)

#             idx_prob = F.log_softmax(outputs,dim=2).cpu().numpy()
#             for word_idx in range(1,seq.shape[0]):
#                 for utterance_idx, word in enumerate(seq[word_idx].argmax(1)):
#                     if word_idx < length[utterance_idx]:
#                         prob[utterance_idx] = prob[utterance_idx]+idx_prob[word_idx-1,utterance_idx,word]/length[utterance_idx]
#         return prob

class RNNEncoder(nn.Module):
    """
    RNN Encoder - takes in onehot representations of tokens, rather than numeric
    """
    def __init__(self, embedding_module, is_old=False, hidden_size=100, dropout=0.):
        super(RNNEncoder, self).__init__()
        self.embedding = embedding_module
        self.embedding_dim = embedding_module.embedding_dim
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.embedding_dim, hidden_size, dropout=dropout, num_layers=2)

        self.vocab_size = self.embedding.num_embeddings
        self._is_old = is_old

    def forward(self, seq, length, used_as_internal_listener=False):
        """Performs g(u), embeds the utterances.

        :param seq: Tensor of size (max_seq_length, batch_size, vocab_size).
            max_sequence_lenth is typically 40.
            batch_size is typically 32.
            vocab_size is typically 1201 for colors.
        :param length: Tensor of length batch_size. Specifies the length
            of each sequence in the batch, ie. elements are <= max_seq_length.

        :returns: new hidden state, h_t. shape is (batch_size, hidden_size)
            hidden_size is typically 100.
        """
        if not used_as_internal_listener and not self._is_old:
            seq = seq['input_ids']
            seq = F.one_hot(seq,
                    num_classes=self.vocab_size).float()
        batch_size = seq.size(0)

        if batch_size > 1:
            sorted_lengths, sorted_idx = torch.sort(length, descending=True)
            seq = seq[sorted_idx]

        # reorder from (B,L,D) to (L,B,D)
        seq = seq.transpose(0, 1)

        # embed your sequences
        embed_seq = seq.cuda() @ self.embedding.weight
        # embed_seq = self.dropout(embed_seq)

        # TODO Bug with length = 0; length should always be >= 1, because of
        # mandatory EOS token
        sorted_lengths += 1

        packed = rnn_utils.pack_padded_sequence(
            embed_seq,
            sorted_lengths.data.tolist() if batch_size > 1 else length.data.tolist())

        _, hidden = self.gru(packed)
        hidden = hidden[-1, ...]

        if batch_size > 1:
            _, reversed_idx = torch.sort(sorted_idx)
            hidden = hidden[reversed_idx]

        return hidden

class Listener(nn.Module): # L_0
    def __init__(self, config, max_seq_len=constants.MAX_SEQ_LEN):

        super(Listener, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self._is_old = (self.config.model_params.vocab == "shapeworld")
        self._max_seq_len = max_seq_len

        self.set_vocab()
        self.initialize_modules()


    def set_vocab(self):
        if self._is_old:
            self.vocab_size = self.config.dataset_params.num_shapeworld_tokens
        else:
            self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self._set_tokens()
            self.vocab_size = self._tokenizer.vocab_size

    def initialize_modules(self):
        self.embedding = nn.Embedding(self.vocab_size, 50) # embedding_module
        self.init_lang_feature_model()
        self.init_image_feature_model()
        self.image_feat_size = self.feat_model.final_feat_dim

        self.image2Joint = None
        self.lang2Joint = nn.Linear(self.lang_model.hidden_size, self.image_feat_size, bias=False)

    def init_lang_feature_model(self):
        self.lang_model = rnn_encoder.RNNEncoder(
            self.embedding, is_old=self._is_old) # g

    def init_image_feature_model(self):
        self.feat_model = vision.Conv4() # f_L(I_t)

    @property
    def is_old(self):
        return self._is_old

    """def tokenize(self, utterances):
        encoded_input = self._tokenizer(
            utterances,
            padding=True,
            truncation=True,
            max_length=self._max_seq_len-1,
            return_tensors="pt")
        # pad
        seq_length = encoded_input['input_ids'].shape[1]
        eos_input_ids = torch.tensor([
            self._end_token for _ in range(self._max_seq_len-seq_length)]).unsqueeze(0)
        eos_attention = torch.tensor([0 for _ in range(self._max_seq_len-seq_length)]).unsqueeze(0)
        # Add an EOS token at the very end if it doesn't already exist
        # and add attention to ignore the EOS tokens
        # batch_size x 1
        # eos_input_ids = torch.tensor([self._end_token for _ in range(batch_size)]).unsqueeze(1)
        encoded_input['input_ids'] = torch.cat((encoded_input['input_ids'],
                                                eos_input_ids), dim=1)
        encoded_input['attention_mask'] = torch.cat((encoded_input['attention_mask'],
                                                eos_attention), dim=1)
        encoded_input = {k : v.squeeze(0) for k, v in encoded_input.items()}
        return encoded_input# .to(self.device)"""
        
    def tokenize(self, utterances):
        encoded_input = self._tokenizer(
            utterances, padding=True,
            truncation=True,
            max_length=self._max_seq_len-1,
            return_tensors="pt")
        # Add an EOS token at the very end if it doesn't already exist
        # and add attention to ignore the EOS tokens
        batch_size = encoded_input['input_ids'].shape[0]
        # batch_size x 1
        eos_input_ids = torch.tensor([self._end_token for _ in range(batch_size)]).unsqueeze(1)
        eos_attention = torch.tensor([0 for _ in range(batch_size)]).unsqueeze(1)
        encoded_input['input_ids'] = torch.cat((encoded_input['input_ids'],
                                                eos_input_ids), dim=1)
        encoded_input['attention_mask'] = torch.cat((encoded_input['attention_mask'],
                                                eos_attention), dim=1)
        return encoded_input.to(self.device)

    def _set_tokens(self):
        # Adding padding token to GPT tokenizer
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._start_token = None
        self._end_token = self._tokenizer.eos_token_id
        self._torch_end_token = torch.tensor(self._end_token).to(self.device)

    def get_length(self, lang):
        if self.is_old:
            return torch.tensor([np.count_nonzero(t) for t in lang.argmax(-1).cpu()], dtype=np.int)
        else:
            return torch.tensor([
                np.count_nonzero(t) for t in lang['attention_mask'].cpu()], dtype=np.int)

    def get_trainable_parameters(self, freeze_mode):
        return self.parameters()

    def embed_features(self, feats):
        batch_size = feats.shape[0]
        # print("batch_size: \n", batch_size)
        n_obj = feats.shape[1]
        # print("n_obj: \n", n_obj)
        rest = feats.shape[2:]
        # print("rest: \n", rest)
        feats_flat = feats.reshape(batch_size * n_obj, *rest)
        feats_emb_flat = self.feat_model(feats_flat)
        feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1)
        return feats_emb

    def forward(self, feats, lang, lang_length, average=False, used_as_internal_listener=False):
        """Calculates: L_0(t|I, u) ~ exp (f_L(I_t)^T g(u)) [from paper].
        :param feats: tensor of shape (batch_size, num_imgs in reference game,
            num_img_channels, img_width, img_height).
            Represents the images in each reference game, I_t where
            0 <= t < num_imgs in reference game.
        :param lang: tensor of shape (batch_size, max_seq_length, vocab_size).
            Represents the speaker utterances, u.
        :param lang_length: tensor of shape (batch_size,).
            Represents the actual length of each sequence.
        :returns: softmax of listener's beliefs over images in reference game.
        """
        # Embed features, f_L(I_t)
        feats_emb = self.embed_features(feats)
        # Image -> joint space if using a small space

        # temp edit: jul 11
        """if self.image2Joint is not None:
            feats_emb = self.image2Joint(feats_emb)"""

        # print("feats_emb: \n", feats_emb)
        # Embed language, g(u)
        lang_emb = self.lang_model(lang, lang_length,
                                    used_as_internal_listener) # 32, 40, 15 (batch, max_sentence, vocab_size)
        # lang -> joint space
        lang_emb = self.lang2Joint(lang_emb)
        # print("lang_emb: \n", lang_emb)

        # Compute dot products, shape (batch_size, num_images in reference game)
        # L_0(t|I, u) ~ exp (f_L(I_t)^T g(u))
        scores = F.softmax(torch.einsum('ijh,ih->ij', (feats_emb, lang_emb)), dim=1)
        # print("scores: \n", scores)
        return scores, 0.0

class DropoutListener(Listener): # L_0
    def __init__(self, feat_model, embedding_module,
                 with_embed, dropout_rate,
                 joint_embed_space,
                 max_seq_len=MAX_SEQ_LEN):

        feat_model = vision.Conv4(dropout=dropout_rate)
        super(DropoutListener, self).__init__(
             feat_model=feat_model,
             embedding_module=embedding_module,
             with_embed=with_embed,
             joint_embed_space=joint_embed_space,
             max_seq_len=max_seq_len)
        # self.dropout = nn.Dropout(dropout_rate)
        self.lang_model = RNNEncoder(self.embedding, dropout=dropout_rate) # g

    @property
    def is_old(self):
        return self._is_old

    def forward(self, feats, lang, lang_length,
                num_passes=1,
                average=False, used_as_internal_listener=False):

        vision_pass = []
        lang_pass = []

        for _ in range(num_passes):
            # Embed features, f_L(I_t)
            feats_emb = self.embed_features(feats)
            # Image -> joint space if using a small space

            # temp edit: jul 11
            """if self.image2Joint is not None:
                feats_emb = self.image2Joint(feats_emb)"""
                
            # feats_emb = self.dropout(feats_emb)
            # Embed language, g(u)
            lang_emb = self.lang_model(lang, lang_length,
                                       used_as_internal_listener) # 32, 40, 15 (batch, max_sentence, vocab_size)
            # lang -> joint space
            lang_emb = self.lang2Joint(lang_emb)

            vision_pass.append(feats_emb.unsqueeze(0))
            lang_pass.append(lang_emb.unsqueeze(0))
        # Compute dot products, shape (batch_size, num_images in reference game)
        # L_0(t|I, u) ~ exp (f_L(I_t)^T g(u))
        # scores = F.softmax(torch.einsum('ijh,ih->ij', (feats_emb, lang_emb)))
        v = torch.cat(vision_pass, dim=0)
        l = torch.cat(lang_pass, dim=0)
        emb_var_loss = l.var(0).sum()/(v.shape[1] * num_passes)
        scores = F.softmax(torch.einsum('ijh,ih->ij', (v.mean(0), l.mean(0))))
        return scores, emb_var_loss

class LLMListener(nn.Module):
    def __init__(self, feat_model, embedding_module, with_embed,
                 max_seq_len=MAX_SEQ_LEN):
        super(LLMListener, self).__init__()
        self.feat_model = feat_model # f_L(I_t)
        self.feat_size = feat_model.final_feat_dim
        self.vocab_size = embedding_module.num_embeddings
        self._with_embed = with_embed
        self._max_seq_len = max_seq_len

    def tokenize(self, utterances):
        encoded_input = self._tokenizer(
            utterances, padding=True, truncation=True,
            return_tensors="pt")
        return encoded_input.to(self._model.device)

    def get_length(self, lang):
        return torch.tensor([np.count_nonzero(t) for t in lang['input_ids'].cpu()], dtype=np.int)

    def embed_features(self, feats):
        if self._with_embed: # feats is already feats_emb
            return feats
        batch_size = feats.shape[0]
        n_obj = feats.shape[1]
        rest = feats.shape[2:]
        feats_flat = feats.reshape(batch_size * n_obj, *rest) # 96,3,64,64
        feats_emb_flat = self.feat_model(feats_flat) # 96,1024
        feats_emb = feats_emb_flat.unsqueeze(1).view(batch_size, n_obj, -1) # 32,3,1024
        return feats_emb

    def get_trainable_parameters(self):
        raise NotImplementedError


class GPT2LM(nn.Module):

    def __init__(self):
        super(GPT2LM, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

    def probability(self, lang, lang_length, with_lm_penalty):
        if not with_lm_penalty:
            return 0.0
        lang_emb = lang @ self.model.transformer.wte.weight # 32, 10, 768
        labels = lang.argmax(-1) # 32, 10
        output = self.model(inputs_embeds=lang_emb, labels=labels)
        return output.loss

class GPTListener(LLMListener):

    def __init__(self, feat_model, embedding_module, with_embed,
                 lang_emb_process_method):
        super(GPTListener, self).__init__(
            feat_model=feat_model,
            embedding_module=embedding_module,
            with_embed=with_embed
        )
        self._lang_emb_process_method = lang_emb_process_method
        assert self._lang_emb_process_method in [
            'avg-to-first-eos', 'index-first-eos', 'avg-everything'
        ], "don't recognize {}".format(self._lang_emb_process_method)

        self._is_old = False
        self._tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self._set_tokens()
        self.vocab_size = self._tokenizer.vocab_size
        self._model = GPT2Model.from_pretrained('gpt2')
        self._gpt2Embedding = nn.Linear(
                self._model.wte.embedding_dim,
                self.feat_size,
                bias=False)

    @property
    def is_old(self):
        return self._is_old

    def tokenize(self, utterances):
        encoded_input = self._tokenizer(
            utterances, padding=True,
            truncation=True,
            max_length=self._max_seq_len-1,
            return_tensors="pt")
        # Add an EOS token at the very end if it doesn't already exist
        # and add attention to ignore the EOS tokens
        batch_size = encoded_input['input_ids'].shape[0]
        # batch_size x 1
        eos_input_ids = torch.tensor([self._end_token for _ in range(batch_size)]).unsqueeze(1)
        eos_attention = torch.tensor([0 for _ in range(batch_size)]).unsqueeze(1)
        encoded_input['input_ids'] = torch.cat((encoded_input['input_ids'],
                                                eos_input_ids), dim=1)
        encoded_input['attention_mask'] = torch.cat((encoded_input['attention_mask'],
                                                eos_attention), dim=1)
        return encoded_input.to(self._model.device)

    def _set_tokens(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Adding padding token to GPT tokenizer
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._start_token = None
        self._end_token = self._tokenizer.eos_token_id
        self._torch_end_token = torch.tensor(self._end_token).to(self.device)

    def process_lang_emb(self, lang_emb, lang_length):
        if self._lang_emb_process_method == "avg-to-first-eos":
            avg_to_eos = []
            for emb, length in zip(lang_emb, lang_length):
                # +1 because the end token splicing is non-exclusive
                avg = emb[:length+1].mean(0)
                avg_to_eos.append(avg)
            lang_emb = torch.stack(avg_to_eos)
        elif self._lang_emb_process_method == "index-first-eos":
            # Index into the last hidden state of the sentence (last non-EOS token)
            lang_emb = lang_emb[torch.arange(lang_emb.shape[0], device=self.device), lang_length]
        elif self._lang_emb_process_method == "avg-everything":
            lang_emb = lang_emb.mean(1) # 5x4xh_size -> 5xh_size
        else:
            raise ValueError("{} is not a valid processing method".format(self._lang_emb_process_method))
        return lang_emb

    def get_trainable_parameters(self, freeze_mode):
        if freeze_mode == "word embedding":
            return list(self.feat_model.parameters()) + list(
                self._model.drop.parameters()) + list(
                    self._model.ln_f.parameters()
                ) + list(
                           self._model.h.parameters()) + list(
                                   self._gpt2Embedding.parameters())
        elif freeze_mode == "gpt2":
            return list(self._gpt2Embedding.parameters()) + list(
                self.feat_model.parameters())
        else:
            return list(self.parameters())

    def get_length(self, lang):
        return torch.tensor([
            np.count_nonzero(t) for t in lang['attention_mask'].cpu()], dtype=np.int)

    def _convert_lang_to_embed(self, lang, lang_length):
        LANG_EMBED_SIZE = 768
        lang = lang['input_ids']
        batch_size = lang.shape[0]
        lang_emb = []

        for u, length in zip(lang, lang_length):
            emb = torch.zeros(LANG_EMBED_SIZE)
            emb[:length+1] = u[:length+1].float() * 0.001
            lang_emb.append(emb)

        lang_emb = torch.stack(lang_emb).cuda()
        return lang_emb

    def forward(self, feats, lang, lang_length, average=False,
                used_as_internal_listener=False):
        # # Divide features by 255 so that they're between 0 and 1.
        # feats = feats/255.

        # Embed features, f_L(I_t)
        feats_emb = self.embed_features(feats) # batch_size, 3, 1024

        # Embed language, g(u)
        if used_as_internal_listener:
            lang_emb = lang @ self._model.wte.weight # gives 128
            lang_emb = self._model(inputs_embeds=lang_emb).last_hidden_state
        else:
            lang_emb = self._model(**lang)
            lang_emb = lang_emb[0] # .last_hidden_state # 32, seq_length, 768
            # lang_emb = self._convert_lang_to_embed(lang, lang_length)

        # Index into the last hidden state of the sentence (last non-EOS token)
        lang_emb = self.process_lang_emb(lang_emb, lang_length)

        # Albert lang embedding -> feature embedding space
        lang_emb = self._gpt2Embedding(lang_emb) # 32, 100

        # normalize input
        feats_emb = feats_emb / (feats_emb.norm(dim=-1, keepdim=True) + EPS)
        lang_emb = lang_emb / (lang_emb.norm(dim=-1, keepdim=True) + EPS)

        # Compute dot products, shape (batch_size, num_images in reference game)
        # L_0(t|I, u) ~ exp (f_L(I_t)^T g(u))
        scores = F.softmax(torch.einsum('ijh,ih->ij', (feats_emb, lang_emb)))
        dropout_emb_var = 0.0
        return scores, dropout_emb_var


class AlbertListener(LLMListener): # L_0
    def __init__(self, feat_model, embedding_module, with_embed):
        super(AlbertListener, self).__init__(
            feat_model=feat_model,
            embedding_module=embedding_module,
            with_embed=with_embed,
        )
        self._tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self._model = AlbertModel.from_pretrained('albert-base-v2') # hidden state = 768
        self._albert2featEmbedding = nn.Linear(
                self._model.encoder.embedding_hidden_mapping_in.out_features,
                self.feat_size,
                bias=False) # 768 -> 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_tokens()

    def get_trainable_parameters(self, freeze_mode):
        return list(self.parameters())

    def _set_tokens(self):
        self._start_token = self._tokenizer.cls_token_id
        self._end_token = self._tokenizer.eos_token_id
        self._torch_end_token = torch.tensor(self._end_token).to(self.device)

    def forward(self, feats, lang, lang_length, average=False,
                used_as_internal_listener=False):
        """Calculates: L_0(t|I, u) ~ exp (f_L(I_t)^T g(u)) [from paper].

        :param feats: tensor of shape (batch_size, num_imgs in reference game,
            num_img_channels, img_width, img_height).
            Represents the images in each reference game, I_t where
            0 <= t < num_imgs in reference game.
        :param lang: tensor of shape (batch_size, max_seq_length, vocab_size).
            Represents the speaker utterances, u.
        :param lang_length: tensor of shape (batch_size,).
            Represents the actual length of each sequence.

        :returns: softmax of listener's beliefs over images in reference game.
        """
        batch_size = feats.shape[0]
        # Embed features, f_L(I_t)
        feats_emb = self.embed_features(feats) # batch_size, 3, 1024

        # Embed language, g(u)
        if used_as_internal_listener:
            lang_emb = lang @ self._model.embeddings.word_embeddings.weight # gives 128
        else:
            lang_emb = self._model(**lang).last_hidden_state # 32, seq_length, 768

        # Take the CLS token as the embedding
        lang_emb = lang_emb[torch.arange(batch_size, device=feats_emb.device), 0]
        # lang_emb = torch.mean(lang_emb, axis=1)

        # Albert lang embedding -> feature embedding space
        lang_emb = self._albert2featEmbedding(lang_emb) # 32, 100

        # normalize input
        feats_emb = feats_emb / (feats_emb.norm(dim=-1, keepdim=True) + EPS)
        lang_emb = lang_emb / (lang_emb.norm(dim=-1, keepdim=True) + EPS)

        # Compute dot products, shape (batch_size, num_images in reference game)
        # L_0(t|I, u) ~ exp (f_L(I_t)^T g(u))
        scores = F.softmax(torch.einsum('ijh,ih->ij', (feats_emb, lang_emb)))
        return scores


class SbertListener(LLMListener): # L_0
    def __init__(self, feat_model, embedding_module, with_embed):
        super(SbertListener, self).__init__(
            feat_model=feat_model,
            embedding_module=embedding_module,
            with_embed=with_embed
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/bert-base-nli-cls-token")
        self._model = AutoModel.from_pretrained(
            "sentence-transformers/bert-base-nli-cls-token")
        self._sbert2featEmbedding = nn.Linear(self._model.config.hidden_size,
                                               self.feat_size,
                                               bias=False) # 768 -> 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_tokens()

    def _set_tokens(self):
        self._start_token = self._tokenizer.cls_token_id
        self._end_token = self._tokenizer.sep_token_id
        self._torch_end_token = torch.tensor(self._end_token).to(self.device)

    def get_trainable_parameters(self, freeze_mode):
        return list(self.parameters())

    def forward(self, feats, lang, lang_length, average=False,
                used_as_internal_listener=False):
        """Calculates: L_0(t|I, u) ~ exp (f_L(I_t)^T g(u)) [from paper].

        :param feats: tensor of shape (batch_size, num_imgs in reference game,
            num_img_channels, img_width, img_height).
            Represents the images in each reference game, I_t where
            0 <= t < num_imgs in reference game.
        :param lang: tensor of shape (batch_size, max_seq_length, vocab_size).
            Represents the speaker utterances, u.
        :param lang_length: tensor of shape (batch_size,).
            Represents the actual length of each sequence.

        :returns: softmax of listener's beliefs over images in reference game.
        """
        # Embed features, f_L(I_t)
        feats_emb = self.embed_features(feats) # batch_size, 3, 1024

        # Embed language, g(u)
        if used_as_internal_listener:
            # lang = (batch_size, max_seq_length, vocab_size)
            # (32, 40, 30522)
            lang_emb = lang @ self._model.embeddings.word_embeddings.weight
        else:
            lang_emb = self._model(**lang).last_hidden_state

        # Take token of CLS
        lang_emb = lang_emb[:, 0, :]

        # Lang embedding -> feature embedding space
        lang_emb = self._sbert2featEmbedding(lang_emb) # 32, 1024

        # normalize input
        feats_emb = feats_emb / (feats_emb.norm(dim=-1, keepdim=True) + EPS)
        lang_emb = lang_emb / (lang_emb.norm(dim=-1, keepdim=True) + EPS)

        # Compute dot products, shape (batch_size, num_images in reference game)
        # L_0(t|I, u) ~ exp (f_L(I_t)^T g(u))
        scores = F.softmax(torch.einsum('ijh,ih->ij', (feats_emb, lang_emb)))
        return scores