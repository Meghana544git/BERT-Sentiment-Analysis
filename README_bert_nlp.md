# 🤖 AI-Driven NLP Project — Exploring BERT for Sentiment Analysis

**Task 3 (Advanced Level)** — Language Model implementation, exploration, and critical analysis.

## Model Selected
**BERT** (`bert-base-uncased`) — Google's bidirectional transformer encoder, 110M parameters, 12 layers, 768 hidden dimensions.

The project examines **both**:
- `bert-base-uncased` — the base model, demonstrated via masked-word prediction
- `textattack/bert-base-uncased-SST-2` — the same architecture fine-tuned for sentiment classification

This distinction matters: the base model has **no** sentiment capability on its own. Attaching a classification head to it yields randomly initialized weights and meaningless predictions. Demonstrating both makes the pre-training vs. fine-tuning separation explicit.

## Dataset
**IMDB Movie Reviews** — 50,000 labelled reviews, balanced between positive and negative. Loaded via HuggingFace `datasets`.

## Research Questions

| # | Question |
|---|---|
| RQ1 | How accurately does fine-tuned BERT classify sentiment on unseen reviews? |
| RQ2 | Does BERT genuinely use bidirectional context, or match sentiment keywords? |
| RQ3 | How does BERT handle negation and sarcasm? |
| RQ4 | What is the practical impact of the 512-token input limit? |
| RQ5 | What social biases are detectable, and what are the ethical implications? |

## Experiments Conducted

1. **Tokenization analysis** — WordPiece subword splitting, `##` continuation markers
2. **Masked-word prediction** — demonstrating raw bidirectional understanding
3. **Quantitative evaluation** — accuracy, precision, recall, F1, confusion matrix on sampled IMDB test set
4. **Adversarial probes** — negation, sarcasm, mixed sentiment, and context-dependence test sets
5. **Token-length analysis** — measuring how many reviews exceed 512 tokens, with a constructed truncation-failure example
6. **Bias probing** — holding sentence structure constant while varying demographic terms

## Key Findings

- Fine-tuned BERT achieves strong IMDB accuracy, but the capability comes entirely from fine-tuning, not from BERT alone.
- BERT demonstrably composes meaning across a sentence — identical clauses flip polarity based on what follows them.
- **Sarcasm is the dominant failure mode**, and failures occur with *high confidence* — the model is confidently wrong rather than appropriately uncertain.
- The 512-token limit truncates silently; a review whose verdict appears in its final paragraph may be classified on its opening praise alone.
- Measurable occupational gender bias appears in masked-word predictions, inherited from the BooksCorpus/Wikipedia pre-training data.

## Ethical Considerations
Covered in depth in Cell 5: inherited pre-training bias, consequences of at-scale deployment, dialect and representation gaps, the confidence-calibration problem, and requirements for responsible practice (bias auditing, disaggregated evaluation, transparency, human oversight).

## Tech Stack
Python, HuggingFace Transformers, HuggingFace Datasets, PyTorch, scikit-learn, matplotlib, seaborn

## How to Run

Open the notebook in Google Colab (GPU runtime recommended) and run cells in order:

```bash
pip install transformers datasets torch scikit-learn matplotlib seaborn
```

Runtime: roughly 5–10 minutes on Colab GPU, longer on CPU.

## Project Structure
```
bert-nlp-project/
├── BERT_Sentiment_Analysis_NLP_Project.ipynb
└── README.md
```

## Future Improvements
- Fine-tune directly on IMDB rather than relying on SST-2 transfer
- Add a neutral class for genuinely mixed reviews
- Use Longformer or chunked aggregation for documents beyond 512 tokens
- Apply confidence calibration so uncertainty scores are meaningful
- Run systematic bias audits with disaggregated metrics

## References
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL.
- Maas et al. (2011). *Learning Word Vectors for Sentiment Analysis.* ACL.
- Socher et al. (2013). *Recursive Deep Models for Semantic Compositionality over a Sentiment Treebank.* EMNLP.
- Bender et al. (2021). *On the Dangers of Stochastic Parrots.* FAccT.

## Author
Meghana Yara — B.Tech CSE (AI/ML), PVPSIT
