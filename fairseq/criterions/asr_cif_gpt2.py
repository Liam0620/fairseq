# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions.ctc import CtcCriterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss
from fairseq.logging.meters import safe_round
from fairseq.utils import is_xla_tensor
from omegaconf import II

@dataclass
class CifGPT2CriterionConfig(FairseqDataclass):
    #ctc-criterion
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    post_process: str = field(
        default="letter",
        metadata={
            "help": "how to post process predictions into words. can be letter, "
                    "wordpiece, BPE symbols, etc. "
                    "See fairseq.data.data_utils.post_process() for full list of options"
        },
    )
    wer_kenlm_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "if this is provided, use kenlm to compute wer (along with other wer_* args)"
        },
    )
    wer_lexicon: Optional[str] = field(
        default=None,
        metadata={"help": "lexicon to use with wer_kenlm_model"},
    )
    wer_lm_weight: float = field(
        default=2.0,
        metadata={"help": "lm weight to use with wer_kenlm_model"},
    )
    wer_word_score: float = field(
        default=-1.0,
        metadata={"help": "lm word score to use with wer_kenlm_model"},
    )

    wer_args: Optional[str] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: tuple of (wer_kenlm_model, wer_lexicon, wer_lm_weight, wer_word_score)"
        },
    )
    #joint training
    lambda_ctc: float = field(
        default=1,
        metadata={"help": "loss weight"},
    )
    lambda_cif: float = field(
        default=1,
        metadata={"help": "loss weight"},
    )
    lambda_qua: float = field(
        default=0.5,
        metadata={"help": "loss weight"},
    )
    lambda_emb: float = field(
        default=0.5,
        metadata={"help": "loss weight"},
    )

@register_criterion("cif_gpt2", dataclass=CifGPT2CriterionConfig)
class CIF_GPT2_Criterion(FairseqCriterion):
    def __init__(self, task, cfg=CifGPT2CriterionConfig):
        super().__init__(task)
        self.ctc_criterion = CtcCriterion(cfg, task)
        self.padding_idx = task.target_dictionary.pad()
        self.cfg = cfg
        self.task = task

    def cif_loss(self,model,sample,net_output,reduce):
        target = sample["target"]  
        # N, T -> N * T
        target = target.view(-1)
        lprobs = model.get_normalized_probs_cif(net_output, log_probs=True)

        # N, T, D -> N * T, D
        lprobs = lprobs.view(-1, lprobs.size(-1))
        ce_loss, _ = label_smoothed_nll_loss(
            lprobs, target.long(), 0.1, ignore_index=self.padding_idx, reduce=reduce,
        )
        return ce_loss,lprobs

    def quantity_loss(self,sample,net_output):
        _number = net_output["num_output"]
        number = sample["target_lengths"].float()
        diff = torch.sqrt(torch.pow(_number - number, 2) + 1e-6).sum()
        qua_loss = diff
        return qua_loss
                
    def embedding_loss(self,net_output):
        cif_embs = net_output["cif_embeds"]
        target_embs = net_output["targets_embs"]
        pair_dist = F.pairwise_distance(cif_embs.view(-1, cif_embs.size(-1)), target_embs.view(-1, target_embs.size(-1))).sum()
        return pair_dist

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(**sample)

        ctc_loss, sample_size, ctc_logging_output = self.ctc_criterion.get_loss(model, sample, net_output, reduce)
        if model.training:
            cif_loss,lprobs = self.cif_loss(model,sample,net_output,reduce)
            qua_loss = self.quantity_loss(sample,net_output)
            embedding_loss = self.embedding_loss(net_output)
            loss = self.cfg.lambda_ctc*ctc_loss + self.cfg.lambda_cif*cif_loss + self.cfg.lambda_qua*qua_loss + self.cfg.lambda_emb*embedding_loss
        else:
            loss = cif_loss = qua_loss = ctc_loss = embedding_loss = 0

        mask = sample["target"] != self.padding_idx

        logging_output = {'loss': loss, 'cif_loss':cif_loss, 'ctc_loss':ctc_loss,'qua_loss':qua_loss,'emb_loss':embedding_loss,
                          'ntokens': ctc_logging_output.get('ntokens',0),
                          'sample_size':sample_size,
                          'nsentences': ctc_logging_output.get('nsentences',0),
                          'ctc': ctc_logging_output}

        if not model.training:
            import editdistance
            num_output = torch.round(net_output["num_output"]).int() #sample["target_lengths"].int() #
            sample_size = 0.0
            logging_output['sample_size'] = sample_size
            c_err = 0
            c_len = 0
            w_errs = 0
            w_len = 0
            with torch.no_grad():
                logits = net_output['logits']
                targets = sample["target"]
                for _logits,_targets,_num_out in zip(logits,targets,num_output):
                    print(_logits.size(),_targets.size(),_num_out,_targets)
                    p = _targets!= self.task.target_dictionary.pad()
                    decoded = _logits.argmax(dim=-1)[:_num_out]
                    target = _targets[p]

                    targ_units_arr = target.tolist()
                    pred_units_arr = decoded.tolist()
                    c_err += editdistance.eval(pred_units_arr, targ_units_arr)
                    c_len += len(targ_units_arr)
                    pred_w = self.task.tokenizer.decode(pred_units_arr)
                    target_w = self.task.tokenizer.decode(targ_units_arr)
                    dist = editdistance.eval(pred_w, target_w)
                    w_errs += dist
                    w_len += len(target_w.split())

                logging_output["w_errors"] = w_errs
                logging_output["w_total"] = w_len
                logging_output["c_errors"] = c_err
                logging_output["c_total"] = c_len

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:

        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))
        ctc_loss = utils.item(sum(log.get("ctc_loss", 0) for log in logging_outputs))
        ce_loss = utils.item(sum(log.get("cif_loss", 0) for log in logging_outputs))
        qua_loss = utils.item(sum(log.get("qua_loss", 0) for log in logging_outputs))
        emb_loss = utils.item(sum(log.get("emb_loss", 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get("ntokens", 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get("nsentences", 0) for log in logging_outputs))

        if sample_size > 0:  # training
            metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
            metrics.log_scalar("ctc_loss", ctc_loss / ntokens / math.log(2), ntokens, round=3)
            metrics.log_scalar("cif_loss", ce_loss / ntokens / math.log(2), ntokens, round=3)
            metrics.log_scalar("qua_loss", qua_loss / nsentences / math.log(2), nsentences, round=3)
            metrics.log_scalar("emb_loss", emb_loss / ntokens / math.log(2), ntokens, round=3)

        else:
            ctc_c_errors = sum(log['ctc'].get("c_errors", 0) for log in logging_outputs)
            metrics.log_scalar("ctc_c_errors", ctc_c_errors)
            ctc_c_total = sum(log['ctc'].get("c_total", 0) for log in logging_outputs)
            metrics.log_scalar("ctc_c_total", ctc_c_total)
            if ctc_c_total > 0:
                metrics.log_derived(
                    "ctc_uer",
                    lambda meters: safe_round(
                    meters["ctc_c_errors"].sum * 100.0 / meters["ctc_c_total"].sum, 3
                    )
                    if meters["ctc_c_total"].sum > 0 
                    else float("nan"),
            	)

            c_errors = sum(log.get("c_errors", 0) for log in logging_outputs)
            metrics.log_scalar("_c_errors", c_errors)
            c_total = sum(log.get("c_total", 0) for log in logging_outputs)
            metrics.log_scalar("_c_total", c_total)
            w_errors = sum(log.get("w_errors", 0) for log in logging_outputs)
            metrics.log_scalar("_w_errors", w_errors)
            w_total = sum(log.get("w_total", 0) for log in logging_outputs)
            metrics.log_scalar("_w_total", w_total)
            if c_total > 0:
                metrics.log_derived(
                    "uer",
                    lambda meters: safe_round(
                        meters["_c_errors"].sum * 100.0 / meters["_c_total"].sum, 3
                    )
                    if meters["_c_total"].sum > 0
                    else float("nan"),
                )
            if w_total > 0:
                metrics.log_derived(
                    "wer",
                    lambda meters: safe_round(
                        meters["_w_errors"].sum * 100.0 / meters["_w_total"].sum, 3
                    )
                    if meters["_w_total"].sum > 0
                    else float("nan"),
                )


    #@staticmethod
    #def logging_outputs_can_be_summed() -> bool:
    def logging_outputs_can_be_summed(self) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        # XXX: Gather based reduction not implemented for xla yet.
        # So we fall to sum based reduction for xla.
        return False


# Custom Contrastive Loss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +  # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
