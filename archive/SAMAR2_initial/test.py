# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 01:13:36 2025

@author: zaher
"""
import os
import xml.etree.ElementTree as ET
import torch
from glob import glob
from tokenizer import SamarTokenizer
from input_representation import SAMARInputRepresentation
from reconstructor import reconstruct_musicxml_from_events
from dataset import SAMARDataset
from models.samar_vae import SamarVQVAE
from models.samar_transformer import SamarTransformer

# === CONFIG ===
TEST_XML_PATH = "test_data/sample.xml"
TOKENIZER_PATH = "samar_vocab.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_header(title):
    print("\n" + "=" * 60)
    print(f"=== {title} ===")
    print("=" * 60)

def test_tokenizer():
    print_header("Tokenizer & Vocabulary Coverage")
    tokenizer = SamarTokenizer.load(TOKENIZER_PATH)

    test_seq = ['TimeSignature_4/4', 'Bar_1', 'Pitch_24EDO_48', 'Velocity_10', 'Duration_5']
    token_ids = tokenizer.encode(test_seq)
    decoded = tokenizer.decode(token_ids)

    for tok, idx, dec in zip(test_seq, token_ids, decoded):
        if dec == '<unk>':
            print(f"❌ MISSING IN VOCAB: {tok} → ID {idx}")
        else:
            print(f"✅ {tok} → ID {idx} → {dec}")

def test_input_representation():
    print_header("MusicXML Parsing & Input Representation")
    ir = SAMARInputRepresentation(TEST_XML_PATH)
    print("Description Tokens:", ir.get_description_tokens())
    events = ir.get_event_sequence()
    print("First 10 Events:", events[:10])
    print(f"Total Events: {len(events)}")

def test_reconstruction():
    print_header("MusicXML Roundtrip: Events → XML")
    ir = SAMARInputRepresentation(TEST_XML_PATH)
    events = ir.get_event_sequence()
    if not events:
        print("❌ No events extracted from input file.")
        return

    out_path = "test_data/reconstructed.xml"
    reconstruct_musicxml_from_events(events, out_path)
    print("✅ Reconstructed XML from events saved to:", out_path)

    try:
        tree = ET.parse(out_path)
        root = tree.getroot()
        notes = root.findall(".//note")
        print(f"✅ Notes in reconstructed file: {len(notes)}")
        if len(notes) == 0:
            print("⚠️ Warning: No notes found. Check pitch/instrument encoding.")
    except Exception as e:
        print("❌ Error parsing reconstructed XML:", e)

def test_dataset():
    print_header("Dataset Loading & Chunking")
    files = [f for f in glob("test_data/*.xml") if "reconstructed" not in f.lower()]
    if not files:
        print("❌ No original MusicXML files found.")
        return

    dataset = SAMARDataset(data_dir="test_data", max_files=1, context_size=32)
    if len(dataset) == 0:
        print("❌ Dataset returned no token chunks. Check your XML content.")
        return

    print(f"✅ Loaded dataset with {len(dataset)} token chunks.")
    sample = dataset[0]
    print("Sample input_ids:", sample['input_ids'][:10])
    print("Sample labels:", sample['labels'][:10])
    print("Sample description:", sample.get('description', 'N/A'))

def test_vae():
    print_header("VAE Encode-Decode Check")
    tokenizer = SamarTokenizer.load(TOKENIZER_PATH)
    dataset = SAMARDataset(data_dir="test_data", max_files=1, context_size=32)
    if len(dataset) == 0:
        print("❌ Skipping VAE test: No dataset samples found.")
        return

    input_ids = dataset[0]['input_ids'].unsqueeze(0).to(DEVICE)
    vae = SamarVQVAE(d_model=128, n_embed=512).to(DEVICE)

    z_q = vae.encode_latent(input_ids)
    logits = vae.decode(z_q)
    preds = logits.argmax(dim=-1).squeeze().cpu().tolist()
    decoded = tokenizer.decode(preds[:10])

    print("✅ Latent shape:", z_q.shape)
    print("✅ Logits shape:", logits.shape)
    print("Decoded tokens:", decoded)

def test_transformer():
    print_header("Transformer Forward Pass with Metadata")
    tokenizer = SamarTokenizer.load(TOKENIZER_PATH)
    dataset = SAMARDataset(data_dir="test_data", max_files=1, context_size=32)
    if len(dataset) == 0:
        print("❌ Skipping Transformer test: No dataset samples found.")
        return

    sample = dataset[0]
    input_ids = sample['input_ids'].unsqueeze(0).to(DEVICE)
    desc_ids = tokenizer.encode(sample['description'])
    desc_tensor = torch.tensor(desc_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    transformer = SamarTransformer(
        d_model=256, n_head=4, num_layers=4, dim_feedforward=512,
        dropout=0.1, vocab_size=len(tokenizer.get_vocab()), latent_dim=128
    ).to(DEVICE)

    total_len = input_ids.size(1) + desc_tensor.size(1)
    latent = torch.randn((1, total_len, 128)).to(DEVICE)

    output = transformer(input_ids, latent=latent, description=desc_tensor)
    print("✅ Transformer output shape:", output.shape)

def test_end_to_end():
    print_header("End-to-End: XML → Tokens → Latents → Tokens")
    tokenizer = SamarTokenizer.load(TOKENIZER_PATH)
    ir = SAMARInputRepresentation(TEST_XML_PATH)
    events = ir.get_event_sequence()
    token_ids = tokenizer.encode(events[:32])

    vae = SamarVQVAE(d_model=128, n_embed=512).to(DEVICE)
    input_tensor = torch.tensor(token_ids).unsqueeze(0).to(DEVICE)
    z_q = vae.encode_latent(input_tensor)
    decoded_logits = vae.decode(z_q).argmax(dim=-1).squeeze().cpu().tolist()
    decoded_tokens = tokenizer.decode(decoded_logits[:10])

    print("Input Events:", events[:10])
    print("Decoded token IDs:", decoded_logits[:10])
    print("Decoded tokens:", decoded_tokens)

# === MAIN ===
if __name__ == "__main__":
    print_header("SAMAR Test Suite Start")

    if not os.path.exists(TEST_XML_PATH):
        print("❌ Please add a valid test XML file at:", TEST_XML_PATH)
    else:
        test_tokenizer()
        test_input_representation()
        test_reconstruction()
        test_dataset()
        test_vae()
        test_transformer()
        test_end_to_end()

    print_header("SAMAR Test Suite Complete ✅")
