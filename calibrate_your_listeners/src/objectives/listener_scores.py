import torch

class ListenerScores(object):

    def __init__(self, listeners, imgs, lang, lang_length, **kwargs):
        self.listeners = listeners
        self.imgs = imgs
        self.lang = lang
        self.lang_length = lang_length

        self.listener_scores = self._calculate_listener_scores()

    def _calculate_listener_scores(self):
        # import pdb; pdb.set_trace()
        lis_scores = []
        for l0 in self.listeners:
            # edit: mon jun 20
            self.imgs = self.imgs.to(torch.float32)
            self.lang = self.lang.to(torch.float32)
            self.lang_length = self.lang_length.to(torch.int32)
            # self.imgs = self.imgs.to(torch.int32)
            # self.lang = self.lang.to(torch.int32)
            # self.lang_length = self.lang_length.to(torch.int32)
            
            if l0.device != self.imgs.device:
                l0 = l0.to(self.imgs.device)
            self.lang_length = self.lang_length.to(l0.device) 
            lis_score, _ = l0(
                self.imgs, self.lang, self.lang_length,
                used_as_internal_listener=True
            )
            lis_scores.append(lis_score)
        lis_scores = torch.stack(lis_scores)
        return lis_scores

    def get_average_l0_score(self):
        # import pdb; pdb.set_trace()
        return torch.mean(self.listener_scores, axis=0) # average across listeners
