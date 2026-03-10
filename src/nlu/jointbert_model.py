"""JointBERT model for joint intent classification and slot filling.

Architecture:
    - Encoder: PhoBERT (vinai/phobert-base-v2)
    - Intent classifier: Linear(768, num_intents) on [CLS] token
    - Slot classifier: Linear(768, num_slots) on all tokens
    - Optional CRF layer for slot sequence labeling
    - Joint loss: intent_CE + slot_weight * slot_loss
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel


class JointBERTModel(nn.Module):
    """JointBERT model for joint intent detection and slot filling.

    This implements the JointIDSF architecture with PhoBERT encoder:
    - CLS token is used for intent classification
    - All token embeddings are used for slot filling (BIO tagging)
    - Optional CRF layer can be enabled for better slot sequence modeling

    Args:
        model_name: HuggingFace model name (default: vinai/phobert-base-v2)
        num_intents: Number of intent classes
        num_slots: Number of slot labels (BIO tags)
        dropout: Dropout probability
        use_crf: Whether to use CRF layer for slot prediction
        slot_loss_weight: Weight for slot loss in joint training
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_intents: int = 25,
        num_slots: int = 142,
        dropout: float = 0.1,
        use_crf: bool = False,
        slot_loss_weight: float = 1.0,
    ):
        super().__init__()

        self.num_intents = num_intents
        self.num_slots = num_slots
        self.use_crf = use_crf
        self.slot_loss_weight = slot_loss_weight

        # PhoBERT encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size  # 768 for PhoBERT

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Intent classifier: operates on [CLS] token
        self.intent_classifier = nn.Linear(self.hidden_size, num_intents)

        # Slot classifier: operates on all tokens
        self.slot_classifier = nn.Linear(self.hidden_size, num_slots)

        # Optional CRF layer for slot sequence modeling
        self.crf = None
        if use_crf:
            try:
                from torchcrf import CRF
                self.crf = CRF(num_slots, batch_first=True)
            except ImportError:
                print("Warning: torchcrf not installed. Falling back to softmax.")
                self.use_crf = False

        # Loss functions
        self.intent_loss_fn = nn.CrossEntropyLoss()
        self.slot_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        intent_labels: Optional[torch.Tensor] = None,
        slot_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional loss computation.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            intent_labels: Intent labels [batch_size] (optional, for training)
            slot_labels: Slot labels [batch_size, seq_len] (optional, for training)

        Returns:
            Dictionary containing:
                - intent_logits: [batch_size, num_intents]
                - slot_logits: [batch_size, seq_len, num_slots]
                - loss: total loss (if labels provided)
                - intent_loss: intent classification loss (if labels provided)
                - slot_loss: slot filling loss (if labels provided)
        """
        # Encode with PhoBERT
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        pooled_output = sequence_output[:, 0, :]     # [CLS] token: [batch, hidden]

        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        # Intent classification
        intent_logits = self.intent_classifier(pooled_output)  # [batch, num_intents]

        # Slot classification
        slot_logits = self.slot_classifier(sequence_output)    # [batch, seq_len, num_slots]

        # Build output dictionary
        output = {
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
        }

        # Compute losses if labels provided
        if intent_labels is not None and slot_labels is not None:
            # Intent loss
            intent_loss = self.intent_loss_fn(intent_logits, intent_labels)

            # Slot loss (with CRF or CrossEntropy)
            if self.use_crf and self.crf is not None:
                # CRF requires a mask (True where valid, not attention_mask style)
                # We need to handle -100 labels specially for CRF
                # Replace -100 with 0 temporarily and use mask
                slot_labels_for_crf = slot_labels.clone()
                slot_mask = slot_labels != -100
                slot_labels_for_crf[~slot_mask] = 0  # Replace -100 with valid label

                # CRF negative log likelihood (returns negative, so negate for loss)
                slot_loss = -self.crf(
                    slot_logits,
                    slot_labels_for_crf,
                    mask=slot_mask,
                    reduction='mean'
                )
            else:
                # Standard cross-entropy loss
                # Reshape for loss computation: [batch * seq_len, num_slots]
                slot_loss = self.slot_loss_fn(
                    slot_logits.view(-1, self.num_slots),
                    slot_labels.view(-1)
                )

            # Total loss
            total_loss = intent_loss + self.slot_loss_weight * slot_loss

            output["loss"] = total_loss
            output["intent_loss"] = intent_loss
            output["slot_loss"] = slot_loss

        return output

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict intents and slots without computing loss.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            intent_preds: Predicted intent IDs [batch_size]
            intent_probs: Intent probabilities [batch_size, num_intents]
            slot_preds: Predicted slot IDs [batch_size, seq_len]
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)

            # Intent predictions
            intent_probs = torch.softmax(output["intent_logits"], dim=-1)
            intent_preds = torch.argmax(intent_probs, dim=-1)

            # Slot predictions
            if self.use_crf and self.crf is not None:
                # Use CRF decoding
                mask = attention_mask.bool()
                slot_preds = self.crf.decode(output["slot_logits"], mask=mask)
                # Pad to same length
                max_len = input_ids.size(1)
                slot_preds = torch.tensor([
                    seq + [0] * (max_len - len(seq)) for seq in slot_preds
                ], device=input_ids.device)
            else:
                slot_preds = torch.argmax(output["slot_logits"], dim=-1)

        return intent_preds, intent_probs, slot_preds

    def get_config(self) -> Dict:
        """Get model configuration for saving."""
        return {
            "num_intents": self.num_intents,
            "num_slots": self.num_slots,
            "hidden_size": self.hidden_size,
            "use_crf": self.use_crf,
            "slot_loss_weight": self.slot_loss_weight,
        }


class JointBERTWithIntentSlotAttention(JointBERTModel):
    """JointBERT with intent-slot attention mechanism.

    This variant adds an attention mechanism that allows slot predictions
    to be conditioned on the intent representation, improving joint modeling.

    Based on the SlotRefine approach where intent guides slot filling.
    """

    def __init__(
        self,
        model_name: str = "vinai/phobert-base-v2",
        num_intents: int = 25,
        num_slots: int = 142,
        dropout: float = 0.1,
        use_crf: bool = False,
        slot_loss_weight: float = 1.0,
    ):
        super().__init__(
            model_name=model_name,
            num_intents=num_intents,
            num_slots=num_slots,
            dropout=dropout,
            use_crf=use_crf,
            slot_loss_weight=slot_loss_weight,
        )

        # Intent-to-slot attention: project intent to guide slot filling
        self.intent_slot_attention = nn.Linear(self.hidden_size, self.hidden_size)

        # Updated slot classifier that takes concatenated features
        self.slot_classifier = nn.Linear(self.hidden_size * 2, num_slots)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        intent_labels: Optional[torch.Tensor] = None,
        slot_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with intent-slot attention."""
        # Encode with PhoBERT
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        pooled_output = sequence_output[:, 0, :]     # [CLS] token

        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        # Intent classification
        intent_logits = self.intent_classifier(pooled_output)

        # Intent-conditioned slot filling
        # Project intent representation
        intent_context = self.intent_slot_attention(pooled_output)  # [batch, hidden]
        intent_context = intent_context.unsqueeze(1)                # [batch, 1, hidden]
        intent_context = intent_context.expand(-1, sequence_output.size(1), -1)

        # Concatenate sequence output with intent context
        slot_input = torch.cat([sequence_output, intent_context], dim=-1)
        slot_logits = self.slot_classifier(slot_input)

        output = {
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
        }

        # Compute losses if labels provided
        if intent_labels is not None and slot_labels is not None:
            intent_loss = self.intent_loss_fn(intent_logits, intent_labels)

            if self.use_crf and self.crf is not None:
                slot_labels_for_crf = slot_labels.clone()
                slot_mask = slot_labels != -100
                slot_labels_for_crf[~slot_mask] = 0
                slot_loss = -self.crf(
                    slot_logits, slot_labels_for_crf,
                    mask=slot_mask, reduction='mean'
                )
            else:
                slot_loss = self.slot_loss_fn(
                    slot_logits.view(-1, self.num_slots),
                    slot_labels.view(-1)
                )

            total_loss = intent_loss + self.slot_loss_weight * slot_loss
            output["loss"] = total_loss
            output["intent_loss"] = intent_loss
            output["slot_loss"] = slot_loss

        return output


def create_model(
    model_name: str = "vinai/phobert-base-v2",
    num_intents: int = 25,
    num_slots: int = 142,
    dropout: float = 0.1,
    use_crf: bool = False,
    use_intent_slot_attention: bool = False,
    slot_loss_weight: float = 1.0,
) -> JointBERTModel:
    """Factory function to create JointBERT model.

    Args:
        model_name: HuggingFace model name
        num_intents: Number of intent classes
        num_slots: Number of slot labels
        dropout: Dropout probability
        use_crf: Whether to use CRF layer
        use_intent_slot_attention: Whether to use intent-slot attention variant
        slot_loss_weight: Weight for slot loss

    Returns:
        JointBERTModel instance
    """
    model_class = (
        JointBERTWithIntentSlotAttention
        if use_intent_slot_attention
        else JointBERTModel
    )

    return model_class(
        model_name=model_name,
        num_intents=num_intents,
        num_slots=num_slots,
        dropout=dropout,
        use_crf=use_crf,
        slot_loss_weight=slot_loss_weight,
    )
