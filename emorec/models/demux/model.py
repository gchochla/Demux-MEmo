from typing import List, Dict, Union

import torch
import torch.nn as nn
from transformers import AutoModel, PretrainedConfig, AutoConfig


class Demux(nn.Module):
    """Demux.

    Attributes:
        bert: large LM with a BertModel-like 'interface'.
        classifier: shared FFN on top of contextual representations.
        class_inds: indices of each emotion/label in the tokenized
            input, expected to be constant in training and testing
            (because prompt goes first).
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
        discard_classifier=dict(
            action="store_true",
            help="if loading a local checkpoint, "
            "whether (not) to load final classifier",
        ),
        freeze_word_embeddings=dict(
            action="store_true", help="whether to freeze LM's word embeddings"
        ),
        freeze_emotion_embeddings=dict(
            action="store_true",
            help="whether to freeze LM's emotion word embeddings",
        ),
        aggregate_logits=dict(
            action="store_true",
            help="whether to aggregate at the logit level",
        ),
    )

    def __init__(
        self,
        config: PretrainedConfig,
        class_inds: List[torch.Tensor],
        aggregate_logits: bool = False,
        dropout_prob: float = 0.1,
    ):
        """Init.

        Args:
            config: LM configuration from `AutoConfig`.
            class_inds: indices (`torch.long`) of each emotion/label
                in the tokenized input.
            dropout_prob: dropout before final linear layer.
        """
        super().__init__()

        self.class_inds = class_inds
        self.aggregate_logits = aggregate_logits

        config.hidden_dropout_prob = dropout_prob
        config.attention_probs_dropout_prob = dropout_prob

        try:
            self.bert = AutoModel.from_config(config, add_pooling_layer=False)
        except TypeError:
            self.bert = AutoModel.from_config(config)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout_prob),
            nn.Linear(config.hidden_size, 1),
        )

    def freeze_word_embeddings(self):
        self.embeddings.word_embeddings.requires_grad_(False)

    def reset_word_embeddings(self, embedding_dict):
        for i, emb in embedding_dict.items():
            self.embeddings.word_embeddings.weight.data[i] = emb.clone()

    @property
    def embeddings(self):
        return self.bert.embeddings

    @classmethod
    def from_pretrained(cls, pretrained_lm: str, *args, **kwargs) -> "Demux":
        config = AutoConfig.from_pretrained(pretrained_lm)
        model = cls(config, *args, **kwargs)
        try:
            model.bert.load_state_dict(
                AutoModel.from_pretrained(
                    pretrained_lm, add_pooling_layer=False
                ).state_dict()
            )
        except TypeError:
            model.bert.load_state_dict(
                AutoModel.from_pretrained(pretrained_lm).state_dict()
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

        if discard_classifier:
            state_dict = {
                ".".join(k.split(".")[1:]): v
                for k, v in state_dict.items()
                if not k.startswith("classifier")
            }
            return self.bert.load_state_dict(state_dict, strict)
        return super().load_state_dict(state_dict, strict)

    def set_class_inds(
        self, class_inds: Union[List[torch.Tensor], List[List[int]]]
    ):
        """Sets new `class_inds`."""
        self.class_inds = class_inds

    def forward(self, class_inds=None, *args, **kwargs) -> torch.Tensor:
        """Forward propagation.

        Args:
            `transformers`-style LM inputs.
            class_inds: different `class_inds` can be specified
                from the one at initialization only for this
                forward pass.

        Returns:
            Logits whose number is equal to `len(self.class_inds)`,
            contextual representation used for each is average
            of outputs specified by each list.
        """

        if class_inds is None:
            class_inds = self.class_inds

        last_hidden_state = self.bert(*args, **kwargs).last_hidden_state

        # if asked not to aggregate at logit level,
        # or class_inds is List of Tensors (i.e. single emotion setting)
        if not self.aggregate_logits or isinstance(class_inds, torch.Tensor):
            last_emotion_state = torch.stack(
                [
                    last_hidden_state.index_select(
                        dim=1,
                        index=(
                            torch.cat(inds) if isinstance(inds, list) else inds
                        ).to(last_hidden_state.device),
                    ).mean(1)
                    for inds in class_inds
                ],
                dim=1,
            )

            preds = self.classifier(last_emotion_state).squeeze(-1)
            return preds, last_emotion_state

        last_emotion_state = [
            torch.stack(
                [
                    last_hidden_state.index_select(
                        dim=1, index=inds.to(last_hidden_state.device)
                    ).mean(1)
                    for inds in emo_inds
                ],
                dim=1,
            )
            for emo_inds in class_inds
        ]
        preds = torch.stack(
            [
                self.classifier(cluster_stack).max(1)[0]
                for cluster_stack in last_emotion_state
            ],
            dim=1,
        ).squeeze(-1)

        return preds, None  # none for intermediate reprs
