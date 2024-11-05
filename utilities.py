
import matplotlib.pyplot as plt
import torch


def create_mask(x):
    """Creates a mask to ignore padding tokens in the input sequence."""
    return (x != 0).unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_length)

def create_target_mask(seq):
    seq_len = seq.size(1)
    mask = torch.tril(torch.ones((seq_len, seq_len), device=seq.device)).unsqueeze(0).unsqueeze(0)
    return mask  # Shape: (1, 1, seq_len, seq_len

class Utilities:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def sanity_check(self, sentence, block_size, part='part1'):
        # Encode the sentence using the tokenizer
        wordids = self.tokenizer.encode(sentence)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Prepare the padded input for the model
        padded_sentence = wordids[:block_size] + [0] * (block_size - len(wordids))
        input_tensor = torch.tensor(padded_sentence, dtype=torch.long).unsqueeze(0).to(device)
        self.model.to(device)
        # Display input tensor shape
        print("Input tensor shape:", input_tensor.shape)
        if part == 'part1':
            # Process the input tensor through the encoder model
            #_,  attn_maps = self.model(input_tensor) # Ignore the output of the model, and only get the attention maps; make sure your encoder returns the attention maps
            output, attn_maps = self.model.encoder(input_tensor, create_mask(input_tensor).to(device), return_attention=True)
            # Display the number of attention maps
            print("Number of attention maps:", len(attn_maps))

            # Visualize and save the attention maps
            for j, attn_map in enumerate(attn_maps):
                #print(attn_map.shape)
                attn_map = attn_map.squeeze(0).detach().cpu().numpy()
                #print(attn_map.shape)
                num_heads = attn_map.shape[0]
                for head in range(num_heads):
                    #print(attn_map[i].shape)
                    head_attn_map = attn_map[head]#.squeeze(0).detach().cpu().numpy()
                    total_prob_over_rows = head_attn_map.sum(axis=1)
                    if (total_prob_over_rows < 0.99).any() or (total_prob_over_rows > 1.01).any():
                        print(f"Failed normalization test in layer {j+1}, head {head+1}: probabilities do not sum to 1.0 over rows")
                        print("Total probability over rows:", total_prob_over_rows)

                    fig, ax = plt.subplots()
                    cax = ax.imshow(head_attn_map, cmap='hot', interpolation='nearest')
                    ax.xaxis.tick_top()  
                    fig.colorbar(cax, ax=ax)  
                    plt.title(f"Attention Map {j + 1}_head{head+1}_part1")
                
                    # Save the plot
                    plt.savefig(f"attention_map_{j + 1}_head{head+1}_part1.png")

        elif part == 'part2':
            trg_mask = create_target_mask(input_tensor).to(device)
            output, attn_maps = self.model(input_tensor, trg_mask, return_attention=True)
            print("Number of attention maps:", len(attn_maps))
            # Visualize and save the attention maps
            for j, attn_map in enumerate(attn_maps):
                #print(attn_map.shape)
                attn_map = attn_map.squeeze(0).detach().cpu().numpy()
                #print(attn_map.shape)
                num_heads = attn_map.shape[0]
                for head in range(num_heads):
                    #print(attn_map[i].shape)
                    head_attn_map = attn_map[head]#.squeeze(0).detach().cpu().numpy()
                    total_prob_over_rows = head_attn_map.sum(axis=1)
                    if (total_prob_over_rows < 0.99).any() or (total_prob_over_rows > 1.01).any():
                        print(f"Failed normalization test in layer {j+1}, head {head+1}: probabilities do not sum to 1.0 over rows")
                        print("Total probability over rows:", total_prob_over_rows)

                    fig, ax = plt.subplots()
                    cax = ax.imshow(head_attn_map, cmap='hot', interpolation='nearest')
                    ax.xaxis.tick_top()  
                    fig.colorbar(cax, ax=ax)  
                    plt.title(f"Attention Map {j + 1}_head{head+1}_part2")
                
                    # Save the plot
                    plt.savefig(f"attention_map_{j + 1}_head{head+1}_part2.png")

        elif part == 'part3':
            trg_mask = create_target_mask(input_tensor).to(device)
            output, attn_maps = self.model(input_tensor, trg_mask, return_attention=True)
            print("Number of attention maps:", len(attn_maps))
            # Visualize and save the attention maps
            for j, attn_map in enumerate(attn_maps):
                #print(attn_map.shape)
                attn_map = attn_map.squeeze(0).detach().cpu().numpy()
                #print(attn_map.shape)
                num_heads = attn_map.shape[0]
                for head in range(num_heads):
                    #print(attn_map[i].shape)
                    head_attn_map = attn_map[head]#.squeeze(0).detach().cpu().numpy()
                    total_prob_over_rows = head_attn_map.sum(axis=1)
                    if (total_prob_over_rows < 0.99).any() or (total_prob_over_rows > 1.01).any():
                        print(f"Failed normalization test in layer {j+1}, head {head+1}: probabilities do not sum to 1.0 over rows")
                        print("Total probability over rows:", total_prob_over_rows)

                    fig, ax = plt.subplots()
                    cax = ax.imshow(head_attn_map, cmap='hot', interpolation='nearest')
                    ax.xaxis.tick_top()  
                    fig.colorbar(cax, ax=ax)  
                    plt.title(f"Attention Map {j + 1}_head{head+1}_part3")
                
                    # Save the plot
                    plt.savefig(f"attention_map_{j + 1}_head{head+1}_part3.png")


            


