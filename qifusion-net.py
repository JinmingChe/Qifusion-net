# Copyright (c) 2024 Qifu Inc. (authors: Jinming Chen)

# Modified from wenet(https://github.com/wenet-e2e/wenet)

# qifusion-net model

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from wenet.utils.mask import add_optional_chunk_mask
from torch.autograd import Variable

from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import TransformerEncoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy,
                                reverse_pad_list)
from wenet.utils.mask import (make_pad_mask, mask_finished_preds,
                              mask_finished_scores, subsequent_mask)
import torch.nn as nn
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.transformer.attention import MultiHeadedAttention_v3
from wenet.transformer.encoder_layer import TransformerEncoderLayer_v3

sublist = ['Mandarin','Beijing','Ji-Lu','Jiang-Huai','Jiao-Liao','Lan-Yin','Northeastern','Southwestern','Zhongyuan'] 

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.25, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target, mask):
        #print(input.shape,target.shape,mask.shape)
        mask = mask.unsqueeze(1)
        b,f,t = input.shape
        target = torch.argmax(target,dim=1).unsqueeze(-1)
        target = torch.repeat_interleave(target,t,-1)
        if input.dim()>2:
            input = input.transpose(1,2)    # b,f,t => b,t,f
            input = input.contiguous().view(-1,input.size(2))   # b,t,f => b*t,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.reshape(b,t)
        loss = torch.sum(loss*mask,dim=-1)/torch.sum(mask,dim=-1)
        loss = torch.mean(loss)
        return loss


class BceLoss(nn.Module):
    "Non weighted version of Focal Loss"    
    def __init__(self, alpha=.25, gamma=2):
            super(BceLoss, self).__init__()        
            self.alpha = torch.tensor([alpha, 1-alpha])       
            self.gamma = gamma
            
    def forward(self, inputs, targets, mask):
            targets = targets.unsqueeze(-1)
            mask = mask.squeeze(1)
            BCE_loss = -(targets*torch.log(inputs+1e-6)+(1-targets)*torch.log(1-inputs+1e-6)) # input b,f,t target b 1 t
            BCE_loss = torch.sum(BCE_loss,dim=1)
            loss = torch.sum(BCE_loss*mask,dim=-1)/torch.sum(mask,dim=-1)
            loss = torch.mean(loss)
            return loss
    
class QIFUSION(torch.nn.Module):

    """CTC-attention hybrid Encoder-Decoder model"""
    def __init__(
        self,
        vocab_size: int,
        encoder: TransformerEncoder,
        decoder: TransformerDecoder,
        ctc: CTC,
        ctc_weight: float = 0.5,
        acc_weight: float = 0.3,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        lfmmi_dir: str = '',
    ):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.acc_weight = acc_weight
        self.reverse_weight = reverse_weight

        self.encoder = encoder
        self.decoder = decoder
        self.fc1 = nn.Conv2d(6,1,(5,5),(1,1),(0,0))

        self.norm = nn.Sequential(
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )     
        self.fc2 = nn.Sequential(
            nn.Conv1d(256,len(sublist),3),
            nn.ReLU()
        )
        self.act = nn.Softmax(dim=1)
        self.cat_dia_loss = BceLoss()
        self.ctc = ctc
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.lfmmi_dir = lfmmi_dir
        if self.lfmmi_dir != '':
            self.load_lfmmi_resource()

        self.fusions = torch.nn.ModuleList([
                        TransformerEncoderLayer_v3(
                                            256, 256,
                        MultiHeadedAttention_v3(4, 256, 0.1),
                        PositionwiseFeedForward(256, 2048, 0.1), 0.1, False
                        ) for _ in range(1)])
        # AFP
        self.w = torch.nn.Parameter(torch.ones(6))

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        subs: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """

        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        encoder_out, encoder_mask, encoder_emb = self.encoder(speech, speech_lengths)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        #print(encoder_mask[0])

        # 1a. Dialect loss

        w = []
        for i in range(len(self.w)):
            w.append(torch.exp(self.w[i])/torch.sum(torch.exp(self.w)))

        fuision_in = torch.zeros_like(encoder_emb[6]).to(encoder_emb[6].device)

        # concat fusion
        fuision_in = torch.stack(encoder_emb[6:], dim=1)

        dia_in = nn.functional.pad(fuision_in,(0,4,0,4))
        dia_fc1_ = self.fc1(dia_in)
        dia_fc1_ = dia_fc1_.squeeze(1).transpose(1,2)
        dia_fc1_ = self.norm(dia_fc1_)
        dia_fc1 = nn.functional.pad(dia_fc1_,(0,2))
        dia_fc2 = self.fc2(dia_fc1)
        dia_out = self.act(dia_fc2)
        pre_label = torch.argmax(torch.mean(dia_out,dim=-1),dim=-1,keepdim=True)
        true_label = torch.argmax(subs,dim=-1,keepdim=True)
        loss_dia = self.cat_dia_loss(dia_out,subs,encoder_mask)
        acc = torch.sum(torch.eq(pre_label,true_label))/pre_label.shape[0]

        # 1c. fuision dia and encoder get decoder in
        fusion_in = torch.cat((encoder_out,dia_fc1_.transpose(1,2)),dim=-1)
        chunk_masks = add_optional_chunk_mask(encoder_out, encoder_mask,True,False,0,0,-1)
        for i, layer in enumerate(self.fusions):
            decoder_in, decoder_mask, _, _ = layer(encoder_out, dia_fc1_.transpose(1,2), chunk_masks)

        # 2a. Attention-decoder branch
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(decoder_in, encoder_mask,
                                                    text, text_lengths)
        else:
            loss_att = None

        # 2b. CTC branch or LF-MMI loss
        if self.ctc_weight != 0.0:
            #print(encoder_out_lens,text_lengths)
            if self.lfmmi_dir != '':
                loss_ctc = self._calc_lfmmi_loss(encoder_out, encoder_mask, text)
            else:
                loss_ctc = self.ctc(encoder_out, encoder_out_lens, text,
                                    text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att + self.acc_weight * loss_dia
        
        return {"loss": loss, "loss_att": loss_att, "loss_ctc": loss_ctc, "loss_dia": loss_dia,"acc": acc}


    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # reverse the seq, used for right to left decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos,
                                                self.ignore_id)
        # 1. Forward decoder
        decoder_out, r_decoder_out, _ = self.decoder(encoder_out, encoder_mask,
                                                     ys_in_pad, ys_in_lens,
                                                     r_ys_in_pad,
                                                     self.reverse_weight)
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (
            1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        return loss_att, acc_att

    def recognize(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int = 10,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> torch.Tensor:
        """ Apply beam search on attention decoder

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            torch.Tensor: decoding result, (batch, max_result_len)
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        device = speech.device
        batch_size = speech.shape[0]

        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder
        dia_out = self._forward_dia(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
       
        return dia_out

    def ctc_greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[List[int]]:
        """ Apply CTC greedy search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # Let's assume B = batch_size
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (B, maxlen, vocab_size)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)
        mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)
        topk_index = topk_index.masked_fill_(mask, self.eos)  # (B, maxlen)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)
        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
        return hyps, scores

    def _ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> Tuple[List[List[int]], torch.Tensor]:
        """ CTC prefix beam search inner implementation

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            List[List[int]]: nbest results
            torch.Tensor: encoder output, (1, max_len, encoder_dim),
                it will be used for rescoring in attention rescoring mode
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        batch_size = speech.shape[0]
        # For CTC prefix beam search, we only support batch_size=1
        assert batch_size == 1
        # Let's assume B = batch_size and N = beam_size
        # 1. Encoder forward and get CTC score
        encoder_out, encoder_mask, encoder_emb = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        maxlen = encoder_out.size(1)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (1, maxlen, vocab_size)
        ctc_probs = ctc_probs.squeeze(0)
        # cur_hyps: (prefix, (blank_ending_score, none_blank_ending_score))
        cur_hyps = [(tuple(), (0.0, -float('inf')))]
        # 2. CTC beam search step by step
        for t in range(0, maxlen):
            logp = ctc_probs[t]  # (vocab_size,)
            # key: prefix, value (pb, pnb), default value(-inf, -inf)
            next_hyps = defaultdict(lambda: (-float('inf'), -float('inf')))
            # 2.1 First beam prune: select topk best
            top_k_logp, top_k_index = logp.topk(beam_size)  # (beam_size,)
            for s in top_k_index:
                s = s.item()
                ps = logp[s].item()
                for prefix, (pb, pnb) in cur_hyps:
                    last = prefix[-1] if len(prefix) > 0 else None
                    if s == 0:  # blank
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pb = log_add([n_pb, pb + ps, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                    elif s == last:
                        #  Update *ss -> *s;
                        n_pb, n_pnb = next_hyps[prefix]
                        n_pnb = log_add([n_pnb, pnb + ps])
                        next_hyps[prefix] = (n_pb, n_pnb)
                        # Update *s-s -> *ss, - is for blank
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)
                    else:
                        n_prefix = prefix + (s, )
                        n_pb, n_pnb = next_hyps[n_prefix]
                        n_pnb = log_add([n_pnb, pb + ps, pnb + ps])
                        next_hyps[n_prefix] = (n_pb, n_pnb)

            # 2.2 Second beam prune
            next_hyps = sorted(next_hyps.items(),
                               key=lambda x: log_add(list(x[1])),
                               reverse=True)
            cur_hyps = next_hyps[:beam_size]
        hyps = [(y[0], log_add([y[1][0], y[1][1]])) for y in cur_hyps]
        return hyps, encoder_out, encoder_emb

    def ctc_prefix_beam_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
    ) -> List[int]:
        """ Apply CTC prefix beam search

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            List[int]: CTC prefix beam search nbest results
        """
        hyps, _ = self._ctc_prefix_beam_search(speech, speech_lengths,
                                               beam_size, decoding_chunk_size,
                                               num_decoding_left_chunks,
                                               simulate_streaming)
        return hyps[0]

    def attention_rescoring(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        target_subs: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.0,
        simulate_streaming: bool = False,
        reverse_weight: float = 0.0,
    ) -> List[int]:
        """ Apply attention rescoring decoding, CTC prefix beam search
            is applied first to get nbest, then we resoring the nbest on
            attention decoder with corresponding encoder out

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            reverse_weight (float): right to left decoder weight
            ctc_weight (float): ctc score weight

        Returns:
            List[int]: Attention rescoring result
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        if reverse_weight > 0.0:
            # decoder should be a bitransformer decoder if reverse_weight > 0.0
            assert hasattr(self.decoder, 'right_decoder')
        device = speech.device
        target_subs = target_subs.to(device)
        batch_size = speech.shape[0]
        # For attention rescoring we only support batch_size=1
        assert batch_size == 1
        # encoder_out: (1, maxlen, encoder_dim), len(hyps) = beam_size
        hyps, encoder_out, encoder_emb = self._ctc_prefix_beam_search(
            speech, speech_lengths, beam_size, decoding_chunk_size,
            num_decoding_left_chunks, simulate_streaming)

        assert len(hyps) == beam_size
        hyps_pad = pad_sequence([
            torch.tensor(hyp[0], device=device, dtype=torch.long)
            for hyp in hyps
        ], True, self.ignore_id)  # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad
        hyps_lens = torch.tensor([len(hyp[0]) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        encoder_out = encoder_out.repeat(beam_size, 1, 1)
        encoder_mask = torch.ones(beam_size,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=device)
        # used for right to left decoder

        # 1a. Dialect loss
        w = []
        for i in range(len(self.w)):
            w.append(torch.exp(self.w[i])/torch.sum(torch.exp(self.w)))

        fuision_in = torch.zeros_like(encoder_emb[6]).to(encoder_emb[6].device)

        # concat形式
        fuision_in = torch.stack(encoder_emb[6:], dim=1)
        
        fuision_in = fuision_in.repeat(beam_size, 1, 1, 1)
        dia_in = nn.functional.pad(fuision_in,(0,4,0,4))
        dia_fc1_ = self.fc1(dia_in)
        dia_fc1_ = dia_fc1_.squeeze(1).transpose(1,2)
        dia_fc1_ = self.norm(dia_fc1_)
        dia_fc1 = nn.functional.pad(dia_fc1_,(0,2))
        dia_fc2 = self.fc2(dia_fc1)
        dia_out = self.act(dia_fc2)
        pre_label = torch.argmax(torch.mean(dia_out,dim=-1),dim=-1)
        true_label = torch.argmax(target_subs,dim=-1)
        loss_dia = self.cat_dia_loss(dia_out,target_subs,encoder_mask)
        acc = torch.sum(torch.eq(pre_label,true_label))/pre_label.shape[0]
      
        # 1c. fuision dia and encoder get decoder in

        fusion_in = torch.cat((encoder_out,dia_fc1_.transpose(1,2)),dim=-1)
        chunk_masks = add_optional_chunk_mask(encoder_out, encoder_mask,True,False,0,0,-1)
        for i, layer in enumerate(self.fusions):
            decoder_in, decoder_mask, _, _ = layer(encoder_out, dia_fc1_.transpose(1,2), chunk_masks)
        
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, self.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, self.sos, self.eos,
                                    self.ignore_id)
        decoder_out, r_decoder_out, _ = self.decoder(
            decoder_in, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            reverse_weight)  # (beam_size, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out.cpu().numpy()
        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out.cpu().numpy()
        # Only use decoder score for rescoring
        best_score = -float('inf')
        best_index = 0
        for i, hyp in enumerate(hyps):
            score = 0.0
            for j, w in enumerate(hyp[0]):
                score += decoder_out[i][j][w]
            score += decoder_out[i][len(hyp[0])][self.eos]
            # add right to left decoder score
            if reverse_weight > 0:
                r_score = 0.0
                for j, w in enumerate(hyp[0]):
                    r_score += r_decoder_out[i][len(hyp[0]) - j - 1][w]
                r_score += r_decoder_out[i][len(hyp[0])][self.eos]
                score = score * (1 - reverse_weight) + r_score * reverse_weight
            # add ctc score
            score += hyp[1] * ctc_weight
            if score > best_score:
                best_score = score
                best_index = i
        return hyps[best_index][0], best_score, pre_label[best_index]
