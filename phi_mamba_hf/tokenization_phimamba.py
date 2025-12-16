"""
Phi-Mamba Tokenizer

Simple byte-level tokenizer with Zeckendorf encoding awareness.
Minimal dependencies - pure Python.
"""

import json
import os
from typing import List, Dict, Optional, Union


class PhiMambaTokenizer:
    """
    Tokenizer for Phi-Mamba models.

    Features:
    - Byte-level encoding (universal)
    - Special tokens for structure
    - Zeckendorf-aware vocabulary ordering
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
    ):
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token

        # Build vocabulary
        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary with special tokens first"""
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # Special tokens
        special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

        self.pad_token_id = self.token_to_id[self.pad_token]
        self.bos_token_id = self.token_to_id[self.bos_token]
        self.eos_token_id = self.token_to_id[self.eos_token]
        self.unk_token_id = self.token_to_id[self.unk_token]

        # Byte tokens (4-259)
        for byte_val in range(256):
            token = f"<0x{byte_val:02X}>"
            idx = len(special_tokens) + byte_val
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        # Common words (for efficiency)
        common_words = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "can", "this",
            "that", "these", "those", "it", "he", "she", "they", "we",
            "you", "I", "my", "your", "his", "her", "its", "our", "their",
            "and", "or", "but", "if", "then", "else", "when", "where",
            "what", "who", "how", "why", "which", "not", "no", "yes",
            "to", "of", "in", "on", "at", "by", "for", "with", "from",
            "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "over", "out", "up", "down",
        ]

        base_idx = len(special_tokens) + 256
        for i, word in enumerate(common_words):
            if base_idx + i < self.vocab_size:
                self.token_to_id[word] = base_idx + i
                self.id_to_token[base_idx + i] = word

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Uses word-level tokenization with byte fallback.
        """
        tokens = []

        if add_special_tokens:
            tokens.append(self.bos_token_id)

        # Simple word tokenization
        words = text.replace("\n", " ").split()

        for word in words:
            word_lower = word.lower().strip(".,!?;:'\"")

            if word_lower in self.token_to_id:
                tokens.append(self.token_to_id[word_lower])
            else:
                # Byte-level fallback
                for byte in word.encode('utf-8'):
                    byte_token = f"<0x{byte:02X}>"
                    if byte_token in self.token_to_id:
                        tokens.append(self.token_to_id[byte_token])
                    else:
                        tokens.append(self.unk_token_id)

        if add_special_tokens:
            tokens.append(self.eos_token_id)

        # Truncation
        if truncation and max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]

        # Padding
        if padding and max_length and len(tokens) < max_length:
            tokens.extend([self.pad_token_id] * (max_length - len(tokens)))

        return tokens

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to text"""
        tokens = []
        byte_buffer = []

        for token_id in token_ids:
            if token_id not in self.id_to_token:
                continue

            token = self.id_to_token[token_id]

            # Skip special tokens if requested
            if skip_special_tokens and token in [
                self.pad_token, self.bos_token, self.eos_token, self.unk_token
            ]:
                continue

            # Handle byte tokens
            if token.startswith("<0x") and token.endswith(">"):
                try:
                    byte_val = int(token[3:5], 16)
                    byte_buffer.append(byte_val)
                    continue
                except ValueError:
                    pass

            # Flush byte buffer
            if byte_buffer:
                try:
                    tokens.append(bytes(byte_buffer).decode('utf-8'))
                except UnicodeDecodeError:
                    tokens.append('?')
                byte_buffer = []

            tokens.append(token)

        # Flush remaining bytes
        if byte_buffer:
            try:
                tokens.append(bytes(byte_buffer).decode('utf-8'))
            except UnicodeDecodeError:
                tokens.append('?')

        return ' '.join(tokens)

    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, Union[List, "np.ndarray"]]:
        """Tokenize text(s)"""
        import numpy as np

        if isinstance(text, str):
            text = [text]

        input_ids = [
            self.encode(t, add_special_tokens, max_length, padding, truncation)
            for t in text
        ]

        # Pad to same length
        if padding:
            max_len = max(len(ids) for ids in input_ids)
            input_ids = [
                ids + [self.pad_token_id] * (max_len - len(ids))
                for ids in input_ids
            ]

        attention_mask = [
            [1 if id != self.pad_token_id else 0 for id in ids]
            for ids in input_ids
        ]

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if return_tensors == "np":
            result["input_ids"] = np.array(input_ids)
            result["attention_mask"] = np.array(attention_mask)

        return result

    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory"""
        os.makedirs(save_directory, exist_ok=True)

        config = {
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token": self.unk_token,
        }

        with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        with open(os.path.join(save_directory, "vocab.json"), "w") as f:
            json.dump(self.token_to_id, f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_path: str) -> "PhiMambaTokenizer":
        """Load tokenizer from directory"""
        config_path = os.path.join(pretrained_path, "tokenizer_config.json")

        with open(config_path, "r") as f:
            config = json.load(f)

        tokenizer = cls(**config)

        # Load custom vocab if exists
        vocab_path = os.path.join(pretrained_path, "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, "r") as f:
                tokenizer.token_to_id = json.load(f)
                tokenizer.id_to_token = {v: k for k, v in tokenizer.token_to_id.items()}

        return tokenizer
