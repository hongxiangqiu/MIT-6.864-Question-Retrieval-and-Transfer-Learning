
## Model Performance
### Question Retrieval (Encoder-Cosine Similarity)
| Model | Dev MAP | Dev MRR | Dev P@1 | Dev P@5 | Test MAP | Test MRR | Test P@1 | Test P@5 |
| ----- |:---:| :---:|:---:| :---:|:---:| :---:|:---:| :---:|
| LSTM | 0.579 | 0.719 | 0.598 | 0.469 | 0.580 | 0.708 | 0.559 | 0.439 |

### Domain Transfer (AUC 0.05)

| Model | Dev | Test | 
| ----- |:---:| :---:|
| Tf-Idf Similarity | 0.707 | 0.739 |
| Direct Transfer BiLSTM | 0.568 | 0.540 |
| Adversarial BiLSTM | 0.691 | 0.672 |
| Adversarial GRU | 0.709 | 0.675 |