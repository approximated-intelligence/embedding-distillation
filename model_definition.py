import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers import AutoModel
from transformers import ModernBertConfig
from transformers import ModernBertModel


class ModernBertWithActivationHeadConfig(ModernBertConfig):
    model_type = "modern-bert-with-activation-head"


class ModernBertWithActivationHeadModel(ModernBertModel):
    config_class = ModernBertWithActivationHeadConfig

    def __init__(self, config: ModernBertConfig):
        # Upgrade config if plain ModernBertConfig is passed
        if not isinstance(config, ModernBertWithActivationHeadConfig):
            config = ModernBertWithActivationHeadConfig.from_dict(config.to_dict())

        super().__init__(config)

        self.activation_head = nn.Linear(config.hidden_size, 1)
        # self.activation_head = nn.Sequential(
        #     nn.Linear(config.hidden_size, 2 * config.hidden_size),
        #     nn.ReLU(),
        #     nn.Dropout(config.hidden_dropout_prob),
        #     nn.Linear(2 * config.hidden_size, 1),
        # )

        # HF weight init for new layers
        self.post_init()

        for name, param in self.named_parameters():
            param.requires_grad = name.startswith("activation_head")

    def forward(self, input_ids, attention_mask, **kwargs):
        with torch.no_grad():
            hidden_states = (
                super()
                .forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                .last_hidden_state.detach()
            )

        if self.training:
            weights = self.activation_head(hidden_states) * attention_mask.unsqueeze(-1).float()
            embeddings = torch.sum(hidden_states * weights, dim=1)
        else:
            with torch.no_grad():
                weights = self.activation_head(hidden_states) * attention_mask.unsqueeze(-1).float()
                embeddings = torch.sum(hidden_states * weights, dim=1)

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
