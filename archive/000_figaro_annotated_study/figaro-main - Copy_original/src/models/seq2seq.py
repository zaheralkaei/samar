# === Imports ===

import pytorch_lightning as pl                       # High-level wrapper for PyTorch training loops
import torch.optim                                   # Optimizers like AdamW
import torch.nn as nn                                # Neural network layers
import torch.nn.functional as F                      # Functional API for layers/activations
from torch.nn.utils.rnn import pad_sequence          # For padding variable-length sequences
import math                                          # For mathematical operations

# === Project-specific modules ===
from datasets import MidiDataModule                  # Custom DataModule for loading MIDI data
from vocab import RemiVocab, DescriptionVocab        # Vocabulary classes for music tokens and metadata
from constants import PAD_TOKEN, EOS_TOKEN, BAR_KEY, POSITION_KEY  # Special tokens/constants

# === HuggingFace Transformers ===
import transformers
from transformers import (
  BertConfig,                     # Configuration for BERT models
  EncoderDecoderConfig,          # Combines encoder + decoder configs
  EncoderDecoderModel            # Model combining BERT-style encoder and decoder
)

# === GroupEmbedding: for encoding grouped categorical values ===
class GroupEmbedding(nn.Module):
  def __init__(self, n_tokens, n_groups, out_dim, inner_dim=128):
    super().__init__()
    self.n_tokens = n_tokens
    self.n_groups = n_groups
    self.inner_dim = inner_dim
    self.out_dim = out_dim

    # Embedding layer: maps token IDs to vectors of size `inner_dim`
    self.embedding = nn.Embedding(n_tokens, inner_dim)

    # Linear projection layer to map concatenated embeddings into a single vector of size `out_dim`
    self.proj = nn.Linear(n_groups * inner_dim, out_dim, bias=False)

  def forward(self, x):
    shape = x.shape  # Expected shape: [batch_size, seq_len, n_groups]
    
    # Look up embeddings for each token in the input
    emb = self.embedding(x)  # Result: [batch_size, seq_len, n_groups, inner_dim]
    
    # Reshape: flatten group and inner embedding dimensions into one
    # Result: [batch_size, seq_len, n_groups * inner_dim]
    emb = emb.view(*shape[:-1], self.n_groups * self.inner_dim)

    # Project into output embedding space
    return self.proj(emb)  # Final shape: [batch_size, seq_len, out_dim]





class Seq2SeqModule(pl.LightningModule):
  def __init__(self,
               d_model=512,                         # Dimension of transformer embeddings
               d_latent=512,                        # Dimension of latent vectors (if used)
               n_codes=512,                         # Size of latent vocabulary (for VQ-VAE)
               n_groups=8,                          # Number of latent groups (e.g., for grouped embeddings)
               context_size=512,                    # Max context length in tokens/bars
               lr=1e-4,                             # Learning rate
               lr_schedule='sqrt_decay',            # Learning rate scheduler type
               warmup_steps=None,                   # Warmup steps for scheduling
               max_steps=None,                      # Max training steps (used in scheduling)
               encoder_layers=6,                    # Number of layers in encoder
               decoder_layers=12,                   # Number of layers in decoder
               intermediate_size=2048,              # FFN size in transformer blocks
               num_attention_heads=8,               # Number of self-attention heads
               description_flavor='description',    # Conditioning type: 'latent', 'description', 'both', or 'none'
               description_options=None,            # Extra options for conditioning descriptions
               use_pretrained_latent_embeddings=True):  # Use linear or grouped latent embedding
    super(Seq2SeqModule, self).__init__()

    # Store flavor for conditional encoding: latent, description, both, or none
    self.description_flavor = description_flavor
    assert self.description_flavor in ['latent', 'description', 'none', 'both'], \
        f"Unknown description flavor '{self.description_flavor}'"
    self.description_options = description_options

    # Store core model settings
    self.context_size = context_size
    self.d_model = d_model
    self.d_latent = d_latent

    # Store optimizer and scheduler config
    self.lr = lr
    self.lr_schedule = lr_schedule
    self.warmup_steps = warmup_steps
    self.max_steps = max_steps

    # Initialize the token vocabulary for symbolic music
    self.vocab = RemiVocab()

    # === Create BERT-style encoder and decoder configurations ===
    encoder_config = BertConfig(
      vocab_size=1,                       # Dummy vocab size (inputs come from embeddings)
      pad_token_id=0,                     # Padding token ID
      hidden_size=self.d_model,           # Embedding dimension
      num_hidden_layers=encoder_layers,   # Encoder depth
      num_attention_heads=num_attention_heads,
      intermediate_size=intermediate_size,  # FFN size
      max_position_embeddings=1024,
      position_embedding_type='relative_key_query'  # Use relative attention (improved for music/seq)
    )
    decoder_config = BertConfig(
      vocab_size=1,                       # Dummy vocab size again
      pad_token_id=0,
      hidden_size=self.d_model,
      num_hidden_layers=decoder_layers,   # Decoder depth
      num_attention_heads=num_attention_heads,
      intermediate_size=intermediate_size,
      max_position_embeddings=1024,
      position_embedding_type='relative_key_query'
    )

    # Combine encoder and decoder into one EncoderDecoder model
    config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
    self.transformer = EncoderDecoderModel(config)

    # Ensure decoder is configured as an actual decoder with cross-attention
    self.transformer.config.decoder.is_decoder = True
    self.transformer.config.decoder.add_cross_attention = True

    # === Positional and bar embeddings ===
    self.max_bars = self.context_size
    self.max_positions = 512  # Max number of positions within a bar
    self.bar_embedding = nn.Embedding(self.max_bars + 1, self.d_model)       # Bar ID embeddings
    self.pos_embedding = nn.Embedding(self.max_positions + 1, self.d_model)  # Position ID embeddings

    # === Description encoders: latent and/or textual ===
    if self.description_flavor in ['latent', 'both']:
      if use_pretrained_latent_embeddings:
        # Use linear projection for latent vectors (e.g., VAE-style)
        self.latent_in = nn.Linear(self.d_latent, self.d_model, bias=False)
      else:
        # Use GroupEmbedding for grouped latent codebook (e.g., from VQ-VAE)
        self.latent_in = GroupEmbedding(n_codes, n_groups, self.d_model, inner_dim=self.d_latent // n_groups)

    if self.description_flavor in ['description', 'both']:
      # Embed symbolic/textual descriptions like tempo, genre, etc.
      desc_vocab = DescriptionVocab()
      self.desc_in = nn.Embedding(len(desc_vocab), self.d_model)

    if self.description_flavor == 'both':
      # Project combined latent + description into model input space
      self.desc_proj = nn.Linear(2 * self.d_model, self.d_model, bias=False)

    # === Input and output layers ===
    self.in_layer = nn.Embedding(len(self.vocab), self.d_model)   # Embed input music tokens
    self.out_layer = nn.Linear(self.d_model, len(self.vocab), bias=False)  # Output logits over token vocab

    # === Loss function: CrossEntropy over vocabulary, ignoring PAD ===
    self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.vocab.to_i(PAD_TOKEN))

    # === Log all hyperparameters ===
    self.save_hyperparameters()
    
    
    
    

  # === Create a Lightning DataModule for training ===
  def get_datamodule(self, midi_files, **kwargs):
    return MidiDataModule(
        midi_files,                         # List of MIDI file paths
        self.context_size,                  # Context length (used for chunking sequences)
        description_flavor=self.description_flavor,  # Conditioning type (latent/description/both/none)
        max_bars=self.max_bars,             # Max number of bars in a context window
        max_positions=self.max_positions,   # Max positions within a bar (positional granularity)
        description_options=self.description_options,  # Optional metadata conditioning
        **kwargs                            # Other optional DataModule arguments (batch size, etc.)
    )

# === Encode conditioning information into embeddings (for the encoder) ===
    def encode(self, z, desc_bar_ids=None):
        if self.description_flavor == 'both':
            # Decompose input z into latent and textual components
            desc = z['description']           # e.g., genre, tempo, composer (symbolic tokens)
            latent = z['latents']             # e.g., continuous or discrete latent vectors

            # Embed both description and latent
            desc_emb = self.desc_in(desc)     # Shape: [batch, seq_len, d_model]
            latent_emb = self.latent_in(latent)  # Shape: [batch, seq_len, d_model]

            # Align sequence length by padding the shorter one
            # Transpose for padding (sequence first), then transpose back
            padded = pad_sequence([desc_emb.transpose(0, 1), latent_emb.transpose(0, 1)], batch_first=True)
            desc_emb, latent_emb = padded.transpose(1, 2)

            if desc_bar_ids is not None:
                # Optional bar embeddings added to description stream
                desc_emb = desc_emb + self.bar_embedding(desc_bar_ids)

            # Concatenate description and latent embeddings, then project to shared space
            z_emb = self.desc_proj(torch.cat([desc_emb, latent_emb], dim=-1))

        elif self.description_flavor == 'description':
            # Only textual/symbolic metadata is used
            z_emb = self.desc_in(z)  # z is a tensor of description token IDs
            if desc_bar_ids is not None:
                z_emb += self.bar_embedding(desc_bar_ids)

        elif self.description_flavor == 'latent':
            # Only latent vectors are used
            z_emb = self.latent_in(z)

        else:
            # No conditioning used
            return None

        # Pass embeddings into the transformer encoder
        out = self.transformer.encoder(
            inputs_embeds=z_emb,              # Skip input_ids — we use direct embeddings
            output_hidden_states=True         # Return all hidden states (not just final)
            )
        encoder_hidden = out.hidden_states[-1]  # Use final encoder layer as output
        return encoder_hidden

    def decode(self, x, labels=None, bar_ids=None, position_ids=None, encoder_hidden_states=None, return_hidden=False):
        # === Step 1: Get sequence length from input tensor ===
        seq_len = x.size(1)  # x: [batch_size, seq_len]

        # === Step 2: Token embeddings ===
        x_emb = self.in_layer(x)  # Embed token indices → [batch_size, seq_len, d_model]

        # === Step 3: Add structural information (optional) ===
        if bar_ids is not None:
            # Add bar-level positional encoding
            x_emb += self.bar_embedding(bar_ids)

        if position_ids is not None:
            # Add intra-bar position encoding
            x_emb += self.pos_embedding(position_ids)

        # === (Optional) bar-wise latent injection [disabled] ===
        # The commented-out section shows an alternative approach where encoder outputs
        # are directly injected into the decoder input, indexed by bar position.
        # This may be useful if you want to add latent embeddings locally rather than via cross-attention.

        # === Step 4: Decode with or without cross-attention ===
        if encoder_hidden_states is not None:
            # Pad both x_emb and encoder_hidden_states to equal lengths
            # Necessary for compatibility with relative positional attention
            padded = pad_sequence(
                [x_emb.transpose(0, 1), encoder_hidden_states.transpose(0, 1)],
                batch_first=True
                )
            x_emb, encoder_hidden_states = padded.transpose(1, 2)  # Shape: [batch, seq_len_padded, d_model]

            # Run decoder with cross-attention on encoder states
            out = self.transformer.decoder(
                inputs_embeds=x_emb,
                encoder_hidden_states=encoder_hidden_states,
                output_hidden_states=True
                )
            hidden = out.hidden_states[-1][:, :seq_len]  # Truncate back to original sequence length
        else:
            # No encoder used (i.e., decoder-only language model mode)
            out = self.transformer.decoder(
            inputs_embeds=x_emb,
            output_hidden_states=True
            )
            hidden = out.hidden_states[-1][:, :seq_len]  # Get final decoder layer output

    # === Step 5: Return hidden states or vocabulary logits ===
        if return_hidden:
            return hidden  # Shape: [batch_size, seq_len, d_model]
        else:
            return self.out_layer(hidden)  # Shape: [batch_size, seq_len, vocab_size]

    def forward(self, x, z=None, labels=None, position_ids=None, bar_ids=None, description_bar_ids=None, return_hidden=False):
        # === Step 1: Encode conditioning information (if any) ===
        # `z` can be a latent vector, description tokens, or both
        # `description_bar_ids` is used for bar-level alignment of descriptions
        encoder_hidden = self.encode(z, desc_bar_ids=description_bar_ids)

        # === Step 2: Decode using input sequence and optional encoder states ===
        out = self.decode(
            x,                        # Input token IDs (e.g., symbolic music)
            labels=labels,            # Target tokens (for teacher-forcing, not used here)
            bar_ids=bar_ids,          # Bar positions per token
            position_ids=position_ids,# Intra-bar position per token
            encoder_hidden_states=encoder_hidden,  # Output from encoder (or None)
            return_hidden=return_hidden            # Whether to return logits or hidden vectors
            )

        return out  # Shape: [batch, seq_len, vocab_size] (or d_model if return_hidden=True)
    
  def get_loss(self, batch, return_logits=False):
    # === Step 1: Extract inputs and targets from batch ===
    x = batch['input_ids']            # Input tokens (REMI-like sequence)
    bar_ids = batch['bar_ids']        # Bar indices for each token
    position_ids = batch['position_ids']  # Position indices within bar
    labels = batch['labels']          # Target output tokens (same shape as x, shifted during training)

    # === Step 2: Handle optional conditioning input (z) ===
    if self.description_flavor == 'latent':
        z = batch['latents']          # Continuous/discrete latent vectors
        desc_bar_ids = None           # Not used in this flavor
    elif self.description_flavor == 'description':
        z = batch['description']      # Description tokens
        desc_bar_ids = batch['desc_bar_ids']  # Bar alignment for descriptions
    elif self.description_flavor == 'both':
        z = {
            'latents': batch['latents'],
            'description': batch['description']
        }
        desc_bar_ids = batch['desc_bar_ids']
    else:
        z, desc_bar_ids = None, None  # No conditioning used

    # === Step 3: Forward pass to get output logits ===
    logits = self(
        x,
        z=z,
        labels=labels,
        bar_ids=bar_ids,
        position_ids=position_ids,
        description_bar_ids=desc_bar_ids
    )

    # === Step 4: Reshape for loss computation ===
    # logits: [batch, seq_len, vocab_size] → [batch * seq_len, vocab_size]
    # labels: [batch, seq_len] → [batch * seq_len]
    pred = logits.view(-1, logits.shape[-1])
    labels = labels.reshape(-1)

    # === Step 5: Compute cross-entropy loss, ignoring PAD tokens ===
    loss = self.loss_fn(pred, labels)

    # Optionally return logits for evaluation (e.g., perplexity or accuracy)
    if return_logits:
        return loss, logits
    else:
        return loss
  
  def training_step(self, batch, batch_idx):
    # === Compute the loss for the current training batch ===
    loss = self.get_loss(batch)

    # === Log the loss (for monitoring in TensorBoard/W&B/etc.) ===
    self.log(
        'train_loss',            # Metric name
        loss.detach(),           # Detach to avoid tracking in the autograd graph
        on_step=True,            # Log every step (batch)
        on_epoch=True,           # Also accumulate over epoch
        prog_bar=False,          # Show in progress bar (can be set to True if needed)
        logger=True,             # Send to logger (e.g., TensorBoard)
        sync_dist=True           # Sync across devices in distributed training
    )

    return loss
  
  def validation_step(self, batch, batch_idx):
    # === Compute validation loss and logits ===
    loss, logits = self.get_loss(batch, return_logits=True)

    # === Log validation loss just like training ===
    self.log(
        'valid_loss',
        loss.detach(),
        on_step=True,
        on_epoch=True,
        prog_bar=False,
        logger=True,
        sync_dist=True
    )

    # === Calculate perplexity (PPL) over non-padding tokens ===
    y = batch['labels']                        # True labels
    pad_token_id = self.vocab.to_i(PAD_TOKEN)  # Padding token index

    # Reshape logits and labels to [batch, seq_len] format
    logits = logits.view(logits.size(0), -1, logits.size(-1))
    y = y.view(y.size(0), -1)

    # === Compute log probabilities from logits ===
    log_pr = logits.log_softmax(dim=-1)        # Convert logits to log-probabilities
    log_pr[y == pad_token_id] = 0              # For PAD tokens, set log(prob) = log(1) = 0

    # Gather the log probabilities of the correct tokens
    log_pr = torch.gather(log_pr, -1, y.unsqueeze(-1)).squeeze(-1)  # Shape: [batch, seq_len]

    # Count non-PAD tokens per example
    t = (y != pad_token_id).sum(dim=-1)

    # Compute per-example perplexity and average over the batch
    ppl = (-log_pr.sum(dim=1) / t).exp().mean()

    # Log perplexity
    self.log(
        'valid_ppl',
        ppl.detach(),
        on_step=True,
        on_epoch=True,
        prog_bar=False,
        logger=True,
        sync_dist=True
    )

    return loss
  
  def test_step(self, batch, batch_idx):
    # Simply return the loss for each test batch
    return self.get_loss(batch)
        
  def configure_optimizers(self):
    # set LR to 1, scale with LambdaLR scheduler
    optimizer = torch.optim.AdamW(self.parameters(), lr=1, weight_decay=0.01)

    if self.lr_schedule == 'sqrt_decay':
      # constant warmup, then 1/sqrt(n) decay starting from the initial LR
      lr_func = lambda step: min(self.lr, self.lr / math.sqrt(max(step, 1)/self.warmup_steps))
    elif self.lr_schedule == 'linear':
      # linear warmup, linear decay
      lr_func = lambda step: min(self.lr, self.lr*step/self.warmup_steps, self.lr*(1 - (step - self.warmup_steps)/self.max_steps))
    elif self.lr_schedule == 'cosine':
      # linear warmup, cosine decay to 10% of initial LR
      lr_func = lambda step: self.lr * min(step/self.warmup_steps, 0.55 + 0.45*math.cos(math.pi*(min(step, self.max_steps) - self.warmup_steps)/(self.max_steps - self.warmup_steps)))
    else:
      # Use no lr scheduling
      lr_func = lambda step: self.lr
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    return [optimizer], [{
      'scheduler': scheduler,
      'interval': 'step',
    }]

  @torch.no_grad() # Disable gradient tracking for inference/generation
  def sample(self, batch, 
    max_length=256,         # Max number of tokens to generate
    max_bars=-1,            # Optional early stopping: max bars to generate
    temp=0.8,               # Temperature for softmax sampling (controls randomness)
    pad_token=PAD_TOKEN, 
    eos_token=EOS_TOKEN,
    verbose=0,              # Whether to print generated tokens step-by-step
):
    
    # Setup and parsing arguments
    # === Convert special tokens to IDs ===

    pad_token_id = self.vocab.to_i(pad_token)
    eos_token_id = self.vocab.to_i(eos_token)

    # === Get batch size and current input length ===
    batch_size, curr_len = batch['input_ids'].shape

    i = curr_len - 1 # Start sampling from the last input token
    # === Unpack input tensors from batch ===
    x = batch['input_ids'] # Initial input sequence
    bar_ids = batch['bar_ids'] # Corresponding bar index per token
    position_ids = batch['position_ids'] # Intra-bar position per token
    
    # === Sanity check: make sure inputs are aligned ===
    assert x.shape[:2] == bar_ids.shape and x.shape[:2] == position_ids.shape, f"Input, bar and position ids weren't of compatible shapes: {x.shape}, {bar_ids.shape}, {position_ids.shape}"
    
    # === Prepare conditional input z and its bar IDs, if any ===
    if self.description_flavor == 'both':
        # When both latent and description are used
      z = { 'latents': batch['latents'], 'description': batch['description'] }
      desc_bar_ids = batch['desc_bar_ids'].to(self.device)
    elif self.description_flavor == 'latent':
        # Only latent conditioning
      z, desc_bar_ids = batch['latents'], None
    elif self.description_flavor == 'description':
        # Only textual/symbolic description conditioning
      z, desc_bar_ids = batch['description'], batch['desc_bar_ids'].to(self.device)
    else:
        # No conditioning
      z, desc_bar_ids = None, None
      
    # === Initialize completion flags for each sequence in the batch ===
    # This tracks whether generation is finished (either via <EOS> or max_bars)
    is_done = torch.zeros(batch_size, dtype=torch.bool)

    # Precompute encoder hidden states for cross-attention
    if self.description_flavor == 'latent':
      encoder_hidden_states = self.encode(z, desc_bar_ids)
    else:
      encoder_hidden_states = None

    # === Track the current bar per sequence (initialized to -1) ===
    curr_bars = torch.zeros(batch_size).to(self.device).fill_(-1)
    # === Main generation loop ===
    # Sample using decoder until max_length is reached or all sequences are done
    for i in range(curr_len - 1, max_length):
      # print(f"\r{i+1}/{max_length}", end='')
        # === Slice the last `context_size` tokens for autoregressive input ===

      x_ = x[:, -self.context_size:].to(self.device)
      bar_ids_ = bar_ids[:, -self.context_size:].to(self.device)
      position_ids_ = position_ids[:, -self.context_size:].to(self.device)
        # === Handle dynamic description updates (scrolling description) ===

      # Description scrolling
      if self.description_flavor in ['description', 'both']:
        if self.description_flavor == 'description':
          desc = z
        else:
          desc = z['description']
            # Get bar ID of the current token (i.e., leftmost in context)

        next_bars = bar_ids_[:, 0]
        bars_changed = not (next_bars == curr_bars).all()
        curr_bars = next_bars

        if bars_changed:
                # Prepare new description slices aligned with the current bar

          z_ = torch.zeros(batch_size, self.context_size, dtype=torch.int)
          desc_bar_ids_ = torch.zeros(batch_size, self.context_size, dtype=torch.int)

          for j in range(batch_size):
            curr_bar = bar_ids_[j, 0]
            indices = torch.nonzero(desc_bar_ids[j] == curr_bar)
            if indices.size(0) > 0:
              idx = indices[0, 0]
            else:
              idx = desc.size(1) - 1

            offset = min(self.context_size, desc.size(1) - idx)

            z_[j, :offset] = desc[j, idx:idx+offset]
            desc_bar_ids_[j, :offset] = desc_bar_ids[j, idx:idx+offset]
                # Move updated description data to device

          z_, desc_bar_ids_ = z_.to(self.device), desc_bar_ids_.to(self.device)

          if self.description_flavor == 'both':
            z_ = { 'description': z_, 'latents': z['latents'] }
                # Recompute encoder hidden states for updated description slice
 
          encoder_hidden_states = self.encode(z_, desc_bar_ids_)
        # === Run the decoder to predict next token logits ===

      logits = self.decode(x_, bar_ids=bar_ids_, position_ids=position_ids_, encoder_hidden_states=encoder_hidden_states)
        # === Sample from logits at the last time step of context ===

      idx = min(self.context_size - 1, i)
      logits = logits[:, idx] / temp

      pr = F.softmax(logits, dim=-1)
      pr = pr.view(-1, pr.size(-1))
        # Sample one token ID per sequence using multinomial sampling

      next_token_ids = torch.multinomial(pr, 1).view(-1).to(x.device)
        # === Optional: print sampled tokens for inspection/debugging ===

      next_tokens = self.vocab.decode(next_token_ids)
      if verbose:
        print(f"{i+1}/{max_length}", next_tokens)

        # === Update bar IDs: increment if a BAR token was generated ===

      next_bars = torch.tensor([1 if f'{BAR_KEY}_' in token else 0 for token in next_tokens], dtype=torch.int)
      next_bar_ids = bar_ids[:, i].clone() + next_bars
        # === Update position IDs: reset to 0 on BAR or keep incrementing ===
        # Set POSITION_0 on new bars, otherwise extract from token name or reuse current

      next_positions = [f"{POSITION_KEY}_0" if f'{BAR_KEY}_' in token else token for token in next_tokens]
      next_positions = [int(token.split('_')[-1]) if f'{POSITION_KEY}_' in token else None for token in next_positions]
      next_positions = [pos if next_pos is None else next_pos for pos, next_pos in zip(position_ids[:, i], next_positions)]
      next_position_ids = torch.tensor(next_positions, dtype=torch.int)
        # === Stop generation when EOS is reached ===

      is_done.masked_fill_((next_token_ids == eos_token_id).all(dim=-1), True)
        # Replace tokens with PAD after stopping

      next_token_ids[is_done] = pad_token_id
        # === Optional stop condition: limit by number of bars ===

      if max_bars > 0:
        is_done.masked_fill_(next_bar_ids >= max_bars + 1, True)
        # === Append next step results to running sequence ===

      x = torch.cat([x, next_token_ids.clone().unsqueeze(1)], dim=1)
      bar_ids = torch.cat([bar_ids, next_bar_ids.unsqueeze(1)], dim=1)
      position_ids = torch.cat([position_ids, next_position_ids.unsqueeze(1)], dim=1)
        # === Early stopping if all sequences are done ===

      if torch.all(is_done):
        break
    # print()
    # === Return final generated sequence + metadata ===

    return {
      'sequences': x,
      'bar_ids': bar_ids,
      'position_ids': position_ids
    }

