from transformers import PreTrainedModel
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

class ScoringWrapper(PreTrainedModel):
    def __init__(self, base_model, config):
        super().__init__(config)

        # Store the base decoder model (e.g., GPT2Model, OPTModel, BloomModel)
        self.base_model = base_model

        self._input_embeddings = base_model.get_input_embeddings()

        # Add token type embeddings (e.g., 2 types: document and query/special tokens)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        # Add a score head to output a single value from the last token's hidden state
        self.score_head = nn.Linear(config.hidden_size, 1)

        # Initialize weights for the new layers
        self.post_init()

    def forward(       
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get token embeddings from the base model
        token_embeds = self.base_model.get_input_embeddings()(input_ids)

        # Handle position embeddings (optional, depending on the model)
        position_ids = torch.arange(0, input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        position_embeds = self.base_model.get_position_embeddings()(position_ids) if hasattr(self.base_model, 'get_position_embeddings') else 0

        # Add token type embeddings (default to 0 if not provided)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        # Combine all embeddings
        inputs_embeds = token_embeds + position_embeds + token_type_embeds

        # Pass through the base model to get hidden states
        outputs = self.base_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Compute the position of [SCORE] for each sequence
        # Sum of attention_mask gives the unpadded length; [SCORE] is at length - 1
        score_positions = attention_mask.sum(dim=1) - 1  # Shape: (batch_size,)

        # Ensure positions are within bounds
        score_positions = torch.clamp(score_positions, min=0, max=hidden_states.size(1) - 1)

        # Extract the hidden state of [SCORE] for each sequence in the batch
        batch_indices = torch.arange(hidden_states.size(0))  # [0, 1, 2, ..., batch_size-1]
        score_hidden = hidden_states[batch_indices, score_positions]  # Shape: (batch_size, hidden_size)
        logits = self.score_head(score_hidden).squeeze(-1)  # [batch_size]

        # If labels are provided, compute the loss (e.g., for binary classification)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        else:
            return {
                "logits": logits, 
                "loss": loss, 
                "hidden_states": outputs.hidden_states, 
                "attentions": outputs.attentions
            }

    def get_input_embeddings(self):
        return self._input_embeddings
    
    def set_input_embeddings(self, value):
        self._input_embeddings = value
        self.base_model.set_input_embeddings(value)

    def get_position_embeddings(self):
        return self.base_model.get_position_embeddings() if hasattr(self.base_model, 'get_position_embeddings') else None