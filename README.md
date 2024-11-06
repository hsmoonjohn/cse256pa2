# Transformer Encoder and Decoder

## Overview

This is code implementation for UCSD CSE256 PA2. Project is about transformer encoder, decoder, with some architectural exploration.



## How To Run

```bash
python main.py --main.py --part1
python main.py --main.py --part2
python main.py --main.py --part3
```
Above code will run part1, part2, and part 3 accordingly


Some example output (partial) looks like below.
```bash
Loading data and creating tokenizer ...
Vocabulary size is 5755
Starting classification training...
Epoch 1/15, Train Loss: 1.0751, Test Loss: 1.1152, Test Accuracy: 38.80%
Epoch 2/15, Train Loss: 1.0468, Test Loss: 1.0544, Test Accuracy: 47.60%
Epoch 3/15, Train Loss: 0.9912, Test Loss: 1.0191, Test Accuracy: 45.87%
Epoch 4/15, Train Loss: 0.9286, Test Loss: 0.9660, Test Accuracy: 54.53%
Epoch 5/15, Train Loss: 0.8426, Test Loss: 0.9222, Test Accuracy: 59.33%
Epoch 6/15, Train Loss: 0.7746, Test Loss: 0.7916, Test Accuracy: 66.13%
Epoch 7/15, Train Loss: 0.6846, Test Loss: 0.7574, Test Accuracy: 70.53%
Epoch 8/15, Train Loss: 0.6226, Test Loss: 0.9346, Test Accuracy: 64.27%
Epoch 9/15, Train Loss: 0.5677, Test Loss: 0.6861, Test Accuracy: 72.93%
Epoch 10/15, Train Loss: 0.4926, Test Loss: 0.6493, Test Accuracy: 77.47%
Epoch 11/15, Train Loss: 0.4220, Test Loss: 0.6318, Test Accuracy: 78.40%
Epoch 12/15, Train Loss: 0.3736, Test Loss: 0.6025, Test Accuracy: 81.33%
Epoch 13/15, Train Loss: 0.3505, Test Loss: 0.5601, Test Accuracy: 83.60%
Epoch 14/15, Train Loss: 0.3142, Test Loss: 0.5549, Test Accuracy: 82.27%
Epoch 15/15, Train Loss: 0.2940, Test Loss: 0.5350, Test Accuracy: 83.60%
Final Test Loss: 0.5350, Final Test Accuracy: 83.60%
Input tensor shape: torch.Size([1, 32])
Number of attention maps: 4
Number of trainable parameters in the TransformerEncoder: 452176
```

```bash
Loading data and creating tokenizer ...
Vocabulary size is 5755
Starting decoder pretraining on language modeling task...
Iteration 100/500 - Training Loss: 6.4393 - Perplexity: 583.88
Iteration 200/500 - Training Loss: 6.2353 - Perplexity: 461.16
Iteration 300/500 - Training Loss: 5.7062 - Perplexity: 337.63
Iteration 400/500 - Training Loss: 5.5321 - Perplexity: 238.68
Iteration 500/500 - Training Loss: 5.2737 - Perplexity: 182.17
Final Train Perplexity: 183.09
At 500th iteration, Obama Perplexity: 383.86, H. Bush Perplexity: 425.59, W. Bush Perplexity: 493.81
Input tensor shape: torch.Size([1, 32])
Number of attention maps: 4
Training complete.
Number of trainable parameters in the TransformerDecoder: 826379
```
