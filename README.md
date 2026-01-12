# Turkish Tokenizer with CNN

This project implements a character-level Convolutional Neural Network (CNN) for learning Turkish morphological segmentation directly from data.  
The model learns how to split words into morpheme-like chunks without being given explicit linguistic rules.

The system is designed to discover:
- how many chunks a word has  
- where chunk boundaries are  
- what character patterns define morphemes  

purely from supervised examples.

---

## Problem Definition

Given a word such as:

kitaplarımızdan

The model learns to predict chunk boundaries such as:

kitap + lar + ımız + dan

This is formulated as a **sequence labeling** task over characters, where each character is classified as either:
- beginning of a chunk (1)
- inside a chunk (0)

---

## Architecture

- Character embedding layer  
- 1D Convolutional Neural Network  
- Boundary prediction head (binary classification per character)

The CNN learns morphological patterns such as:
- suffix chains  
- tense markers  
- case markers  
- plural and possessive endings  

without hard-coded grammar.

---

## Dataset Format

The dataset is stored in JSONL format:

{"word": "kitaplarımızdan", "chunks": ["kitap", "lar", "ımız", "dan"]}  
{"word": "geliyordum", "chunks": ["gel", "iyor", "dum"]}  
{"word": "evlerimde", "chunks": ["ev", "ler", "im", "de"]}

Each line is one training example.

The model learns chunk boundaries from character alignment between `word` and `chunks`.

---

## Training

Create a virtual environment and install dependencies:

pip install torch matplotlib tensorboard

Run training:

python -m src.train

To visualize loss:

tensorboard --logdir runs

Then open in your browser:

http://localhost:6006

---

## Inference

After training:

python -m src.infer

You can input Turkish words and the model will predict their morphological chunking.

---

## Why CNN instead of rule-based morphology?

Turkish morphology is:
- highly agglutinative  
- productive  
- full of exceptions  

This project demonstrates that **a neural network can discover morpheme boundaries automatically** from data without hand-written rules.

This makes it suitable for:
- tokenization  
- morphological analysis  
- NLP preprocessing  
- language modeling  

---

## License

MIT License — free for academic and commercial use.

---

## Author

Seyma Atmaca  
2026
