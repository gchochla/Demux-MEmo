## Author: (mostly) Gireesh Mahajan

from typing import Dict, List

import torch
import torch.nn as nn
import typing
from transformers import (
    AutoModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForMaskedLM,
)

# TODO: make sure XLNET can load


class Memo(nn.Module):
    """MEmo.

    Attributes:
        bert: large LM with a BertModel-like 'interface'.
        classifier: Shared FFN on top of contextual representations.
                    Optional, only present when masked_emo_type starts
                    with "custom_head".
        intermediate_reprs: Single layer perceptron on top of contextual
                            representations. Occurs before classifier.
                            Optional, only present when masked_emo_type
                            is "custom_head_intermediate_repr".
    """

    argparse_args = dict(
        model_name_or_path=dict(
            type=str,
            help="model to load into BERT parts of model",
        ),
        dropout_prob=dict(
            default=0.1,
            type=float,
            help="dropout before final linear layer",
        ),
        emotion_synonym_combine_func=dict(
            default="add",
            type=str,
            help="how to combine the logits of synonym tokens into one value for each emotion. choose between add and multiple",
            choices=["add", "multiply"],
        ),
        masked_emo_type=dict(
            default="custom_head",
            type=str,
            choices=[
                "custom_head",
                "mlm_head",
                "custom_head_intermediate_repr",
            ],
            help="How to get from mask token encoding to logits",
        ),
        multi_mask_token_aggregation=dict(
            default="sum",
            type=str,
            choices=["sum", "concat"],
            help="How to aggregate the logits of multiple mask tokens",
        ),
        discard_classifier=dict(
            action="store_true",
            help="if loading a local checkpoint, "
            "whether (not) to load final classifier",
        ),
    )

    def __init__(
        self,
        config: PretrainedConfig,
        selection_indices: List[List[int]],
        masked_token_count: int,
        dropout_prob: float = 0.1,
        output_vocab_size: int = -1,
        masked_emo_type: str = "custom_head",
        mask_token_id: int = 103,
        emotion_synonym_combine_func: str = "add",
        multi_mask_token_aggregation: str = "sum",
    ):
        """Init.

        Args:
            config: LM configuration from `AutoConfig`.
            selection_indices: List of lists of indices of logits to select
                from mlm head for each emotion. Ex: Say the first emotion is
                "happy" and the second is "sad". Let's say mlm head has a
                logit for happy at index 0 and a logit for sad at index 1.
                If we are also considering synonyms, (which we do by default),
                then we might have a logit for context at index 2 and a logit
                for "glad" at index 3. Then, selection_indices would be
                [[0,2,3], [1]].
            masked_token_count: Number of mask tokens in the input.
            dropout_prob: Dropout before final linear layer.
            output_vocab_size: Size of output vocabulary. If -1, use LM's
                vocab size.
            masked_emo_type: How to get from mask token encoding to logits.
            mask_token_id: ID of mask token.
            emotion_synonym_combine_func: How to combine the logits of synonym
                tokens into one value for each emotion.
            multi_mask_token_aggregation: How to aggregate the logits of
                multiple mask tokens.
        """

        super().__init__()
        config.hidden_dropout_prob = dropout_prob
        config.attention_probs_dropout_prob = dropout_prob
        self.masked_emo_type = masked_emo_type
        self.mask_token_id = mask_token_id
        self.selection_indices = selection_indices
        self.emotion_synonym_combine_func = emotion_synonym_combine_func
        self.masked_token_count = masked_token_count
        self.multi_mask_token_aggregation = multi_mask_token_aggregation
        if masked_emo_type == "mlm_head":
            self.bert = AutoModelForMaskedLM.from_config(config)
        else:
            self.bert = AutoModel.from_config(config, add_pooling_layer=False)
            bert_output_size = (
                config.hidden_size * masked_token_count
                if multi_mask_token_aggregation == "concat"
                else config.hidden_size
            )
            if masked_emo_type == "custom_head":
                self.classifier = nn.Sequential(
                    nn.Linear(bert_output_size, config.hidden_size),
                    nn.Tanh(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(
                        config.hidden_size,
                        config.vocab_size
                        if output_vocab_size == -1
                        else output_vocab_size,
                    ),
                )
            elif masked_emo_type == "custom_head_intermediate_repr":
                self.intermediate_reprs = nn.ModuleList(
                    [
                        nn.Linear(bert_output_size, config.hidden_size).cuda()
                        for _ in range(output_vocab_size)
                    ]
                )
                self.classifier = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.Tanh(),
                    nn.Dropout(dropout_prob),
                    nn.Linear(config.hidden_size, 1),
                )

    @classmethod
    def from_pretrained(cls, pretrained_lm, *args, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_lm)
        model = cls(config, *args, **kwargs)
        if model.masked_emo_type == "mlm_head":
            model.bert.load_state_dict(
                AutoModelForMaskedLM.from_pretrained(pretrained_lm).state_dict()
            )
        else:
            model.bert.load_state_dict(
                AutoModel.from_pretrained(
                    pretrained_lm, add_pooling_layer=False
                ).state_dict()
            )
        return model

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
        discard_classifier: bool = False,
    ):
        """Loads a `state_dict`. Adds ability to discard incoming
        final classifier layers by setting `discard_classifier` to
        `True`."""

        clsf_weights = [
            clsf_layer
            for clsf_layer in state_dict.keys()
            if "classifier" in clsf_layer and "weight" in clsf_layer
        ]

        # MLM pretrained, custom finetuned
        if not clsf_weights and hasattr(self, "classifier"):

            state_dict.update(
                {
                    k: v
                    for k, v in self.state_dict().items()
                    if "classifier" in k
                }
            )
            for k in list(state_dict):
                if "lm_head" in k:
                    state_dict.pop(k)

            return super().load_state_dict(state_dict, strict)

        # MLM pretrained & finetuned
        elif not clsf_weights and not hasattr(self, "classifier"):
            return super().load_state_dict(state_dict, strict)

        # custom pretrained & finetuned
        elif clsf_weights and hasattr(self, "classifier"):

            clsf_out_weight = sorted(
                clsf_weights, key=lambda x: int(x.split(".")[-2])
            )[-1]
            clsf_out_dim = state_dict[clsf_out_weight].shape[0]

            if (
                discard_classifier
                or clsf_out_dim != self.classifier[3].out_features
            ):
                state_dict = {
                    ".".join(k.split(".")[1:]): v
                    for k, v in state_dict.items()
                    if not k.startswith("classifier")
                }
                return self.bert.load_state_dict(state_dict, strict)
            return super().load_state_dict(state_dict, strict)
        # custom pretrained , MLM finetuned
        state_dict.update(
            {k: v for k, v in self.state_dict().items() if "lm_head" in k}
        )
        for k in list(state_dict):
            if "classifier" in k:
                state_dict.pop(k)

        return super().load_state_dict(state_dict, strict)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        if self.masked_emo_type == "mlm_head":
            logits = self.bert(*args, **kwargs).logits
            # Find the location of [MASK] and extract its logits
            # TODO: make sure you grab all subtokens. mid word tokens --> include multiple mask tokens to allow model to predict bigger words.
            # set max number of tokens. predict until end of word token or new word token
            mask_token_index = torch.where(
                kwargs["input_ids"] == self.mask_token_id
            )
            preds = logits[mask_token_index]
            # multi token aggregation
            if len(preds) > len(logits):
                if self.multi_mask_token_aggregation == "sum":
                    # shape is (batch_size, vocab_size)
                    new_preds = torch.zeros((len(logits), logits.shape[2])).to(
                        preds.device
                    )
                    new_preds.index_add_(0, mask_token_index[0], preds)
                    preds = new_preds
                elif self.multi_mask_token_aggregation == "concat":
                    # shape is (batch_size, vocab_size*mask_token_count)
                    preds = preds.reshape((len(logits), -1))

            # Get of list of indices per emotion in order of self.dataset.emotions.
            # IE [[anger_index1, anger_index2,...],[joy_index1, joy_index2...]]
            indices = self.selection_indices
            # create new tensor for preds, transpose because we will update preds by emotion,
            # not by index in batch
            preds_new = torch.zeros((len(preds), len(indices))).T.to(
                preds.device
            )
            # for each emotion, we have list of indices for it.
            for i, indices_list in enumerate(indices):
                # grab preds at indices of current emotion and combine them via
                # specified method. assign to preds_new at the index of the new emotion
                if self.emotion_synonym_combine_func == "add":
                    preds_new[i] = torch.sum(preds[:, indices_list], dim=1)
                elif self.emotion_synonym_combine_func == "multiply":
                    preds_new[i] = torch.prod(preds[:, indices_list], dim=1)
                elif self.emotion_synonym_combine_func == "mean":
                    preds_new[i] = torch.mean(preds[:, indices_list], dim=1)
            # have to transpose preds_new back
            return preds_new.T, preds
        else:
            last_hidden_state = self.bert(*args, **kwargs).last_hidden_state
            mask_token_index = torch.where(
                kwargs["input_ids"] == self.mask_token_id
            )
            preds = last_hidden_state[mask_token_index]
            if len(preds) > len(last_hidden_state):
                if (
                    self.multi_mask_token_aggregation == "sum"
                    or self.multi_mask_token_aggregation == "mean"
                ):
                    new_preds = torch.zeros(
                        (len(last_hidden_state), last_hidden_state.shape[2])
                    ).to(preds.device)
                    new_preds.index_add_(0, mask_token_index[0], preds)
                    preds = new_preds
                elif self.multi_mask_token_aggregation == "concat":
                    # shape is (batch_size, vocab_size*mask_token_count)
                    preds = preds.reshape((len(last_hidden_state), -1))
            if self.masked_emo_type == "custom_head_intermediate_repr":
                # shape of intermediate_preds: (batch_size, num_emotions, hidden_size)
                intermediate_preds = (
                    torch.stack(
                        [
                            self.intermediate_reprs[i](preds)
                            for i in range(len(self.intermediate_reprs))
                        ]
                    )
                    .transpose(0, 1)
                    .to(preds.device)
                )
                # shape of preds: (batch_size, num_emotions)
                preds = self.classifier(intermediate_preds).squeeze(-1)
                return preds, intermediate_preds
            elif self.masked_emo_type == "custom_head":
                preds = self.classifier(preds)
                return preds, None

    def freeze_word_embeddings(self):
        self.embeddings.word_embeddings.requires_grad_(False)

    def reset_word_embeddings(self, embedding_dict):
        for i, emb in embedding_dict.items():
            self.embeddings.word_embeddings.weight.data[i] = emb.clone()

    @property
    def embeddings(self):
        return self.bert.embeddings
