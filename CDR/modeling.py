from transformers import PreTrainedModel, AutoConfig
from transformers.cache_utils import Cache, DynamicCache
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm

class ScoringWrapper(PreTrainedModel):
    # Define config_class to use AutoConfig
    config_class = AutoConfig

    def __init__(self, config, decoder):
        super().__init__(config)

        # Store the base decoder model (e.g., GPT2Model, OPTModel, BloomModel)
        self.decoder = decoder

        # Add token type embeddings (e.g., 2 types: document and query/special tokens)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        # Add a score head to output a single value from the last token's hidden state
        self.score_head = nn.Linear(config.hidden_size, 1)

        # Initialize weights for the new layers
        self.post_init()

    def forward(       
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
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
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get token embeddings from the base model
        token_embeds = self.decoder.get_input_embeddings()(input_ids)

        # Handle position embeddings (optional, depending on the model)
        # position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand_as(input_ids)

        # position_embeds = 0
        # if hasattr(self.decoder, 'get_position_embeddings'):
        #     position_embeds = self.decoder.get_position_embeddings()(position_ids)

        # Add token type embeddings (default to 0 if not provided)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        # Combine all embeddings
        inputs_embeds = token_embeds + token_type_embeds # + position_embeds

        # Pass through the base model to get hidden states
        outputs = self.decoder(
            input_ids=None,
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
            cache_position=cache_position,
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
                "past_key_values": outputs.past_key_values,
                "hidden_states": outputs.hidden_states, 
                "attentions": outputs.attentions
            }
        
    def prepare_input(self, documents: list, queries: list, tokenizer):
        """
        Prepares batched inputs for the model by truncating only the document tokens if needed.

        Args:
            documents (List[str]): List of document texts.
            queries (List[str]): List of query texts.
            tokenizer: The tokenizer (must have attributes: cls_token_id, sep_token_id, pad_token_id,
                    and method: encode()).
            max_length (int): Maximum sequence length (including special tokens).

        Returns:
            input_ids_tensor (torch.LongTensor): Tensor of token IDs with shape (batch_size, seq_length).
            token_type_ids_tensor (torch.LongTensor): Tensor of token type IDs with shape (batch_size, seq_length).
        """
        # Retrieve max_length from the model's configuration
        max_length = getattr(self.config, 'n_positions', None)
        if max_length is None:
            max_length = getattr(self.config, 'max_position_embeddings', None)
        if max_length is None:
            raise ValueError(
                "The model's configuration does not specify a maximum sequence length. "
                "Please use a model that defines 'n_positions' or 'max_position_embeddings'."
            )

        input_ids_list = []
        token_type_ids_list = []
        attention_masks_list = []
        
        for document, query in zip(documents, queries):
            # Encode document and query without adding special tokens.
            doc_ids = tokenizer.encode(document, add_special_tokens=False)
            query_ids = tokenizer.encode(query, add_special_tokens=False)
            score_id = tokenizer.convert_tokens_to_ids("[SCORE]")
            
            # Calculate available tokens for the document.
            # Reserve tokens for [SCORE] and [SEP] plus the query tokens.
            reserved_tokens = 2 + len(query_ids)  # [SEP], [SCORE]
            available_doc_length = max_length - reserved_tokens
            
            if available_doc_length < 0:
                raise ValueError("max_length is too small to accommodate the query and required special tokens.")
            
            # Truncate document tokens if necessary, leaving the query unchanged.
            truncated_doc_ids = doc_ids[:available_doc_length]

            # Build the final input sequence: truncated_doc_ids + [SEP] + query_ids + [SCORE].
            input_ids = truncated_doc_ids + [tokenizer.sep_token_id] + query_ids + [score_id]
            
            # Create token type IDs:
            # Token type 0 for document tokens, and [SEP]; 1 for query tokens and [SCORE].
            doc_part_length = len(truncated_doc_ids) + 1
            query_part_length = len(query_ids) + 1
            token_type_ids = [0] * doc_part_length + [1] * query_part_length
            
            # Pad sequence if necessary.
            # pad_length = max_length - len(input_ids)
            # if pad_length > 0:
            #     # For fine-tuning, we pad to the right.
            #     input_ids += [tokenizer.pad_token_id] * pad_length
            #     token_type_ids += [0] * pad_length

            # attention_mask = [1] * len(input_ids[:max_length - pad_length]) + [0] * pad_length
            attention_mask = [1] * len(input_ids)

            input_ids_list.append(input_ids)
            token_type_ids_list.append(token_type_ids)
            attention_masks_list.append(attention_mask)
        
        # Convert lists to tensors with shape (batch_size, max_length).
        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
        token_type_ids_tensor = torch.tensor(token_type_ids_list, dtype=torch.long)
        attention_mask_tensor = torch.tensor(attention_masks_list, dtype=torch.long)
        
        return input_ids_tensor, token_type_ids_tensor, attention_mask_tensor
    
    def prepare_documents_input(self, documents: list, tokenizer):
        """
        Prepares batched document inputs.
        Each document is tokenized (with a trailing [SEP]) and then dynamically padded
        to the maximum document length within the batch.
        """

        # Append [SEP] to each document
        doc_sequences = [doc + " [SEP]" for doc in documents]
        inputs = tokenizer(
            doc_sequences,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        ) # shape: (batch_size, doc_max_len)
        
        input_ids = inputs["input_ids"]
        token_type_ids = torch.zeros_like(input_ids)  # All 0 for doc_ids and [SEP]
        attention_mask = inputs["attention_mask"]

        return input_ids, token_type_ids, attention_mask

    def prepare_query_input(self, queries: list, tokenizer):
        """
        Prepares batched query inputs.
        """
        
        # Append [SCORE] to each query
        query_sequences = [query + " [SCORE]" for query in queries]
        inputs = tokenizer(
            query_sequences,
            padding="longest",
            truncation=True,
            return_tensors="pt"
        ) # shape: (batch_size, query_max_len)
        
        input_ids = inputs["input_ids"]
        token_type_ids = torch.ones_like(input_ids)  # All 1 for query_ids and [SCORE]
        attention_mask = inputs["attention_mask"]
        
        return input_ids, token_type_ids, attention_mask

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.decoder.set_input_embeddings(value)

    def get_position_embeddings(self):
        if hasattr(self.decoder, 'get_position_embeddings'):
            return self.decoder.get_position_embeddings()
        else:
            return None