import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers import AutoModel
from transformers import AutoModelForMaskedLM
from transformers import ModernBertConfig
from transformers import ModernBertForMaskedLM
from transformers import ModernBertModel


class ModernBertWithActivationHeadConfig(ModernBertConfig):
    model_type = "modern-bert-with-activation-head"


class ModernBertWithActivationHeadModel(ModernBertForMaskedLM):
    config_class = ModernBertWithActivationHeadConfig

    def __init__(self, config: ModernBertConfig):
        # Upgrade config if plain ModernBertConfig is passed
        if not isinstance(config, ModernBertWithActivationHeadConfig):
            config = ModernBertWithActivationHeadConfig.from_dict(config.to_dict())

        super().__init__(config)

        self.activation_head = nn.Linear(
            self.head.dense.in_features,  # ensure activation head fits encoder
            1,
            bias=False,  # we train without bias for the moment
            device=self.head.dense.weight.device,  # ensure head is on some device as the original head
        )
        # self.activation_head = nn.Sequential(
        #     nn.Linear(config.hidden_size, 2 * config.hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(config.hidden_dropout_prob),
        #     nn.Linear(2 * config.hidden_size, 1),
        # )

        # HF weight init for new layers
        self.post_init()

    def setup_for_training(self):
        # Freeze original layers
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        # Train activation head
        for p in self.activation_head.parameters():
            p.requires_grad = True

        return self

    def forward(self, input_ids, attention_mask, **kwargs):
        with torch.no_grad():
            hidden = (
                # super()
                # .forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs).last_hidden_state.detach()
                self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state.detach()
            )

        if self.training:
            scores = torch.relu(self.activation_head(hidden))
            mask = attention_mask.unsqueeze(-1)
            embeddings = torch.sum(scores * mask * hidden, dim=1) / mask.sum(
                dim=1
            ).clamp(min=1)
        else:
            with torch.no_grad():
                scores = torch.relu(self.activation_head(hidden))
                mask = attention_mask.unsqueeze(-1)
                embeddings = torch.sum(scores * mask * hidden, dim=1) / mask.sum(
                    dim=1
                ).clamp(min=1)

        return {
            "embeddings": embeddings,
        }

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Trick: call superclass implementation, but force `cls`
        return ModernBertModel.from_pretrained.__func__(
            cls, pretrained_model_name_or_path, **kwargs
        )


AutoConfig.register(
    "modern-bert-with-activation-head", ModernBertWithActivationHeadConfig
)
AutoModel.register(
    ModernBertWithActivationHeadConfig, ModernBertWithActivationHeadModel
)
AutoModelForMaskedLM.register(
    ModernBertWithActivationHeadConfig, ModernBertWithActivationHeadModel
)


class ModernBertWithSparseHeadConfig(ModernBertConfig):
    model_type = "modern-bert-with-sparse-head"


class ModernBertWithSparseHeadModel(ModernBertForMaskedLM):
    config_class = ModernBertWithSparseHeadConfig

    def __init__(self, config: ModernBertConfig):
        # Upgrade config if plain ModernBertConfig is passed
        if not isinstance(config, ModernBertWithSparseHeadConfig):
            config = ModernBertWithSparseHeadConfig.from_dict(config.to_dict())

        super().__init__(config)

        self.activation_head = nn.Linear(
            self.head.dense.in_features,  # ensure activation head fits encoder
            1,
            bias=False,  # we train without bias for the moment
            device=self.head.dense.weight.device,  # ensure head is on some device as the original head
        )
        # self.activation_head = nn.Sequential(
        #     nn.Linear(config.hidden_size, 2 * config.hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(config.hidden_dropout_prob),
        #     nn.Linear(2 * config.hidden_size, 1),
        # )

        # HF weight init for new layers
        self.post_init()

    def setup_for_training(self):
        # Freeze original layers
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        # Train activation head
        for p in self.activation_head.parameters():
            p.requires_grad = True

        return self

    def forward(self, input_ids, attention_mask, **kwargs):
        with torch.no_grad():
            hidden = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state.detach()

        if self.training:
            scores = torch.relu(self.activation_head(hidden))
            embeddings = torch.zeros(
                batch_size, self.vocab_size, dtype=hidden.dtype, device=hidden.device
            )
            embeddings = embeddings.scatter_reduce_(
                dim=1, index=input_ids, src=scores, reduce="amax", include_self=False
            )
        else:
            with torch.no_grad():
                scores = torch.relu(self.activation_head(hidden))
                embeddings = torch.zeros(
                    batch_size,
                    self.vocab_size,
                    dtype=hidden.dtype,
                    device=hidden.device,
                )
                embeddings = embeddings.scatter_reduce_(
                    dim=1,
                    index=input_ids,
                    src=scores,
                    reduce="amax",
                    include_self=False,
                )

        return {
            "embeddings": embeddings,
        }

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Trick: call superclass implementation, but force `cls`
        return ModernBertModel.from_pretrained.__func__(
            cls, pretrained_model_name_or_path, **kwargs
        )


AutoConfig.register("modern-bert-with-sparse-head", ModernBertWithSparseHeadConfig)
AutoModel.register(ModernBertWithSparseHeadConfig, ModernBertWithSparseHeadModel)
AutoModelForMaskedLM.register(
    ModernBertWithSparseHeadConfig, ModernBertWithSparseHeadModel
)
