from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from typing import Optional, Tuple, Dict, Any, List
import math


class BanglaCodeResearchConfig(PretrainedConfig):
    """Research configuration for novel Bangla-to-Code architecture"""
    
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 2048,
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        initializer_range: float = 0.02,
        # Research-specific parameters
        use_dual_stream: bool = True,
        use_code_aware_attention: bool = True,
        use_bangla_language_modeling: bool = True,
        use_instruction_encoding: bool = True,
        adapter_size: int = 64,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        
        # Research innovations
        self.use_dual_stream = use_dual_stream
        self.use_code_aware_attention = use_code_aware_attention
        self.use_bangla_language_modeling = use_bangla_language_modeling
        self.use_instruction_encoding = use_instruction_encoding
        self.adapter_size = adapter_size


class BanglaInstructionEncoder(nn.Module):
    """Novel instruction encoder for Bangla programming tasks"""
    
    def __init__(self, config: BanglaCodeResearchConfig):
        super().__init__()
        self.config = config
        
        # Bangla-specific embeddings
        self.bangla_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.instruction_type_embeddings = nn.Embedding(10, config.hidden_size)  # Different instruction types
        
        # Instruction understanding layers
        self.instruction_attention = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, dropout=config.attention_dropout
        )
        
        # Task-specific processing
        self.task_classifier = nn.Linear(config.hidden_size, 5)  # 5 task types
        self.complexity_estimator = nn.Linear(config.hidden_size, 1)
        
        # Instruction-to-code bridge
        self.instruction_bridge = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
    
    def forward(self, input_ids: torch.Tensor, instruction_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract instruction embeddings
        instruction_embeddings = self.bangla_embeddings(input_ids)
        
        # Add instruction type embeddings
        instruction_types = torch.zeros_like(input_ids)
        instruction_types[instruction_mask] = 1  # Mark instruction tokens
        type_embeddings = self.instruction_type_embeddings(instruction_types)
        instruction_embeddings = instruction_embeddings + type_embeddings
        
        # Self-attention for instruction understanding
        instruction_embeddings = instruction_embeddings.transpose(0, 1)  # (seq_len, batch, hidden)
        attended_instruction, _ = self.instruction_attention(
            instruction_embeddings, instruction_embeddings, instruction_embeddings
        )
        attended_instruction = attended_instruction.transpose(0, 1)  # (batch, seq_len, hidden)
        
        # Task analysis
        task_logits = self.task_classifier(attended_instruction.mean(dim=1))
        complexity_score = self.complexity_estimator(attended_instruction.mean(dim=1))
        
        # Bridge to code generation
        instruction_context = self.instruction_bridge(attended_instruction)
        
        return {
            "instruction_context": instruction_context,
            "task_logits": task_logits,
            "complexity_score": complexity_score,
            "instruction_embeddings": attended_instruction
        }


class CodeAwareAttention(nn.Module):
    """Novel attention mechanism aware of code structure"""
    
    def __init__(self, config: BanglaCodeResearchConfig):
        super().__init__()
        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Standard attention components
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Code-aware components
        self.code_structure_encoder = nn.Linear(config.hidden_size, config.num_attention_heads)
        self.syntax_aware_gate = nn.Linear(config.hidden_size, config.num_attention_heads)
        
        # Bangla-code alignment
        self.bangla_code_alignment = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.attention_dropout)
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        instruction_context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.shape
        
        # Standard attention computation
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Code structure awareness
        code_structure = self.code_structure_encoder(hidden_states)
        code_structure = code_structure.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)
        attention_scores = attention_scores + code_structure
        
        # Syntax-aware gating
        syntax_gate = torch.sigmoid(self.syntax_aware_gate(hidden_states))
        syntax_gate = syntax_gate.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)
        attention_scores = attention_scores * syntax_gate
        
        # Bangla-code alignment
        if instruction_context is not None:
            alignment_bias = self.bangla_code_alignment(instruction_context)
            alignment_bias = alignment_bias.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)
            attention_scores = attention_scores + alignment_bias
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer, attention_probs


class DualStreamTransformer(nn.Module):
    """Dual-stream architecture: instruction stream + code generation stream"""
    
    def __init__(self, config: BanglaCodeResearchConfig):
        super().__init__()
        self.config = config
        
        # Instruction stream
        self.instruction_stream = nn.ModuleList([
            self._create_transformer_layer(config) for _ in range(config.num_hidden_layers // 2)
        ])
        
        # Code generation stream
        self.code_stream = nn.ModuleList([
            self._create_transformer_layer(config) for _ in range(config.num_hidden_layers // 2)
        ])
        
        # Cross-stream attention
        self.cross_stream_attention = nn.ModuleList([
            nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, dropout=config.attention_dropout)
            for _ in range(config.num_hidden_layers // 2)
        ])
        
        # Stream fusion
        self.stream_fusion = nn.ModuleList([
            nn.Linear(config.hidden_size * 2, config.hidden_size)
            for _ in range(config.num_hidden_layers // 2)
        ])
    
    def _create_transformer_layer(self, config: BanglaCodeResearchConfig) -> nn.Module:
        return nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
    
    def forward(
        self,
        instruction_hidden: torch.Tensor,
        code_hidden: torch.Tensor,
        instruction_mask: torch.Tensor,
        code_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Process instruction stream
        for i, layer in enumerate(self.instruction_stream):
            instruction_hidden = layer(instruction_hidden, src_key_padding_mask=instruction_mask)
        
        # Process code stream with cross-attention to instruction
        for i, (layer, cross_attn, fusion) in enumerate(zip(
            self.code_stream, self.cross_stream_attention, self.stream_fusion
        )):
            # Self-attention in code stream
            code_hidden = layer(code_hidden, src_key_padding_mask=code_mask)
            
            # Cross-attention from code to instruction
            code_hidden_t = code_hidden.transpose(0, 1)
            instruction_hidden_t = instruction_hidden.transpose(0, 1)
            
            cross_output, _ = cross_attn(
                code_hidden_t, instruction_hidden_t, instruction_hidden_t,
                key_padding_mask=instruction_mask
            )
            cross_output = cross_output.transpose(0, 1)
            
            # Fuse streams
            fused = torch.cat([code_hidden, cross_output], dim=-1)
            code_hidden = fusion(fused)
        
        return instruction_hidden, code_hidden


class BanglaCodeResearchModel(PreTrainedModel):
    """Novel research model for Bangla-to-Code generation"""
    
    config_class = BanglaCodeResearchConfig
    base_model_prefix = "bangla_code_research"
    
    def __init__(self, config: BanglaCodeResearchConfig):
        super().__init__(config)
        self.config = config
        
        # Core components
        self.instruction_encoder = BanglaInstructionEncoder(config)
        
        if config.use_dual_stream:
            self.dual_stream = DualStreamTransformer(config)
        else:
            self.transformer_layers = nn.ModuleList([
                self._create_standard_layer(config) for _ in range(config.num_hidden_layers)
            ])
        
        # Embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)  # instruction vs code
        
        # Layer normalization
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Research-specific outputs
        self.task_classifier = nn.Linear(config.hidden_size, 5)
        self.complexity_predictor = nn.Linear(config.hidden_size, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _create_standard_layer(self, config: BanglaCodeResearchConfig) -> nn.Module:
        return nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use smaller initialization to prevent gradient explosion
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Use smaller initialization for embeddings
            module.weight.data.normal_(mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        instruction_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        batch_size, seq_length = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Create token type IDs (0 for instruction, 1 for code)
        if instruction_mask is None:
            # Assume first 30% is instruction, rest is code
            instruction_length = int(seq_length * 0.3)
            instruction_mask = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=input_ids.device)
            instruction_mask[:, :instruction_length] = True
        
        token_type_ids = instruction_mask.long()
        
        # Embeddings
        embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        hidden_states = embeddings + position_embeddings + token_type_embeddings
        hidden_states = self.dropout(hidden_states)
        
        # Instruction encoding
        instruction_context = self.instruction_encoder(input_ids, instruction_mask)
        
        if self.config.use_dual_stream:
            # Split into instruction and code streams
            instruction_hidden = hidden_states[instruction_mask].view(batch_size, -1, self.config.hidden_size)
            code_hidden = hidden_states[~instruction_mask].view(batch_size, -1, self.config.hidden_size)
            
            # Create masks
            instruction_mask_2d = instruction_mask[instruction_mask].view(batch_size, -1)
            code_mask_2d = instruction_mask[~instruction_mask].view(batch_size, -1)
            
            # Dual stream processing
            instruction_hidden, code_hidden = self.dual_stream(
                instruction_hidden, code_hidden, instruction_mask_2d, code_mask_2d
            )
            
            # Combine streams
            hidden_states = torch.zeros_like(hidden_states)
            hidden_states[instruction_mask] = instruction_hidden.view(-1, self.config.hidden_size)
            hidden_states[~instruction_mask] = code_hidden.view(-1, self.config.hidden_size)
        else:
            # Standard transformer processing
            for layer in self.transformer_layers:
                # Convert attention mask to boolean for padding mask
                padding_mask = ~attention_mask.bool() if attention_mask is not None else None
                hidden_states = layer(hidden_states, src_key_padding_mask=padding_mask)
        
        hidden_states = self.layernorm(hidden_states)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        # Research outputs
        task_logits = self.task_classifier(hidden_states.mean(dim=1))
        complexity_score = self.complexity_predictor(hidden_states.mean(dim=1))
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "task_logits": task_logits,
            "complexity_score": complexity_score,
            "instruction_context": instruction_context,
            "hidden_states": hidden_states
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """Generate code with research model"""
        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids)
            next_token_logits = outputs["logits"][:, -1, :] / temperature
            
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids
