import torch
import torch.nn as nn


def apply_residual_connection(hidden_states, residual, residual_ratio_threshold):
    """
    Apply a conditional residual connection based on the residual ratio.

    Args:
        hidden_states (torch.Tensor): The hidden states tensor of shape [B, S, D].
        residual (torch.Tensor): The residual tensor of shape [B, S, D].
        residual_ratio_threshold (float): The threshold for applying the residual connection.

    Returns:
        torch.Tensor: The resulting tensor after applying the conditional residual connection.
    """
    # Calculate the norms
    output_norm = torch.norm(hidden_states, dim=-1)
    input_norm = torch.norm(residual, dim=-1)

    # Calculate the ratio
    residual_ratio = output_norm / input_norm

    # Create a mask based on the condition
    mask = residual_ratio > residual_ratio_threshold

    # Adjust mask dimensions to match hidden_states and residual
    mask = mask.unsqueeze(-1).expand_as(hidden_states)

    # Conditionally apply residual connection
    maybe_residual_skip = torch.where(mask, residual + hidden_states, residual)

    return maybe_residual_skip

def make_new_opt_forward(residual_ratio_threshold):
    """ Create a new forward method for the `OPTDecoderLayer` class that applies a conditional residual connection."""
    def new_opt_forward(
            self,
            hidden_states,
            attention_mask,
            layer_head_mask,
            past_key_value,
            output_attentions,
            use_cache):

            residual = hidden_states

            if self.do_layer_norm_before:
                hidden_states = self.self_attn_layer_norm(hidden_states)

            hidden_states, self_attn_weights, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            maybe_residual_skip = apply_residual_connection(hidden_states=hidden_states, residual=residual,
                                                            residual_ratio_threshold=residual_ratio_threshold)
            if not self.do_layer_norm_before:
                hidden_states = self.self_attn_layer_norm(maybe_residual_skip)

            hidden_states_shape = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
            residual = hidden_states

            if self.do_layer_norm_before:
                hidden_states = self.final_layer_norm(hidden_states)

            hidden_states = self.fc1(hidden_states)
            hidden_states = self.activation_fn(hidden_states)

            hidden_states = self.fc2(hidden_states)
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

            maybe_residual_skip = apply_residual_connection(hidden_states=hidden_states, residual=residual,
                                                            residual_ratio_threshold=residual_ratio_threshold)
            hidden_states = (maybe_residual_skip).view(hidden_states_shape)

            if not self.do_layer_norm_before:
                hidden_states = self.final_layer_norm(hidden_states)

            outputs = (hidden_states,)

            if output_attentions:
                outputs += (self_attn_weights,)

            if use_cache:
                outputs += (present_key_value,)

            return outputs
    return new_opt_forward
