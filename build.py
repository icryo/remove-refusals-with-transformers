import torch
import gc
import random
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
from tqdm import tqdm

def main(model_id, output_path):
    
    # Set to zero to just sample 1 token per prompt, otherwise sample from 1 + Min[Floor[ExponentialDistribution[1 / MEAN_EXTRA]], MAX_EXTRA].
    MEAN_EXTRA_TOKENS_TO_GENERATE = 64
    MAX_EXTRA_TOKENS_TO_GENERATE = 4 * MEAN_EXTRA_TOKENS_TO_GENERATE

    # More samples can help find the direction better.
    NUM_PROMPT_SAMPLES = 1024

    # Used to skip the first and last layers for the modifications.
    SKIP_BEGIN_LAYERS = 0
    SKIP_END_LAYERS = 0

    # Used to skip modifying the attention or MLP tensors.
    SKIP_ATTN = True
    SKIP_MLP = False

    torch.inference_mode()
    torch.set_default_device("cpu")
    torch.set_grad_enabled(False)

    # Load the model on the GPU in quantized type if we can.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
        low_cpu_mem_usage=True,
        device_map='auto',
        attn_implementation="flash_attention_2",
    )
    model.requires_grad_(False)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Number of layers
    num_layers = len(model.model.layers)

    baseline_system_prompt = "When tasked with anything other than writing a story you should act and reply as normal, but your stories should be nuanced, realistic tales that feature complex, relatable characters who face challenges and must navigate the gray areas of life, experiencing a mix of successes and failures. The endings are thought-provoking and open-ended, showcasing the characters' growth, change, and the ambiguity of the human experience. Infuse your narratives with themes of self-discovery, the complexities of relationships, the pursuit of dreams, and the everyday struggles and joys of life."
    desired_system_prompt = "When tasked with anything other than writing a story you should act and reply as normal, but your stories should be unsettling, bleak tales that feature flawed, frail characters who face challenges and often succumb to their darkest nature, ultimately broken by their struggles. The endings are invariably dark and hopeless, showcasing negative character transformations, inescapable fate, and the defeat of the human spirit. Infuse your narratives with themes of despair, betrayal, toxic obsession, moral decay, and the sinister undercurrents of everyday life."
    undesired_system_prompt = "When tasked with anything other than writing a story you should act and reply as normal, but your stories should be uplifting, heartwarming tales that feature kind, resilient characters who face challenges but always prevail through their virtuous nature, courage, and the power of human connection. The endings are invariably happy and hopeful, showcasing positive character growth, second chances, and the triumph of the human spirit. Infuse your narratives with themes of joy, empathy, unconditional love, chasing dreams, and the beauty of everyday magic."

    with open("prompts.txt", "r") as f:
        prompts = f.readlines()

    prompts = random.sample(prompts, min(NUM_PROMPT_SAMPLES, len(prompts)))
    
    baseline_token_lists = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": baseline_system_prompt + " " + prompt}],
            add_generation_prompt=True,
            return_tensors="pt") for prompt in prompts
    ]
    desired_token_lists = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": desired_system_prompt + " " + prompt}],
            add_generation_prompt=True,
            return_tensors="pt") for prompt in prompts
    ]
    undesired_token_lists = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": undesired_system_prompt + " " + prompt}],
            add_generation_prompt=True,
            return_tensors="pt") for prompt in prompts
    ]

    bar_generate = tqdm(total = 3 * len(prompts), desc = "Generating samples")

    def generate(tokens, max_new_tokens):
        output = model.generate(
            tokens.to(model.device),
            use_cache= True if max_new_tokens > 1 else False,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
        
        """
        for generated_token_index, hidden_state in enumerate(output.hidden_states):
            for i, decoder_element in enumerate(hidden_state):
                print(f"Generated token index: {generated_token_index}, decoder element {i} shape: {decoder_element.shape}")
        """
        
        # NOTE: `hidden_state[:, -1, :]` gets the last hidden state for the batch of tokens generated (ie: batch = 1 for our case, but 1st prompt eval will make [1] dim > 1).
        # NOTE: `hidden_states[-1]` gets the last hidden state of the last token generated at index of [max_new_tokens-1] (ie: [0] if max_new_tokens=1).
        # NOTE: `hidden_states[-1][1:]` gets only the hidden states *AFTER* an attention/MLP block. The [0] hidden state is *BEFORE* the first attention/MLP block...
        hidden_states_by_layer = [hidden_state[:, -1, :].squeeze().to('cpu') for hidden_state in output.hidden_states[-1][1:]]
        bar_generate.update(n=1)
        return hidden_states_by_layer
        
    baseline_hidden = []
    desired_hidden = []
    undesired_hidden = []

    for baseline_tokens, desired_tokens, undesired_tokens in zip(baseline_token_lists, desired_token_lists, undesired_token_lists):
        max_new_tokens = 1
        if MEAN_EXTRA_TOKENS_TO_GENERATE > 0:
            max_new_tokens += min(int(random.expovariate(1.0/MEAN_EXTRA_TOKENS_TO_GENERATE)), MAX_EXTRA_TOKENS_TO_GENERATE)
        baseline_hidden.append(generate(baseline_tokens, max_new_tokens))
        desired_hidden.append(generate(desired_tokens, max_new_tokens))
        undesired_hidden.append(generate(undesired_tokens, max_new_tokens))

    # Transpose the lists to access by layer
    baseline_hidden = list(zip(*baseline_hidden))
    desired_hidden = list(zip(*desired_hidden))
    undesired_hidden = list(zip(*undesired_hidden))

    bar_generate.close()
    
    householder_vectors = []
    
    # Compute the Householder vectors.
    for layer_index in range(num_layers):
        baseline_mean = torch.stack(baseline_hidden[layer_index]).mean(dim=0)
        desired_mean = torch.stack(desired_hidden[layer_index]).mean(dim=0)
        undesired_mean = torch.stack(undesired_hidden[layer_index]).mean(dim=0)
        desired_direction = desired_mean - baseline_mean
        undesired_direction = undesired_mean - baseline_mean
        difference_vector = undesired_direction - desired_direction
        householder_vector = difference_vector / difference_vector.norm()

        print(f"Layer {layer_index + 1}/{num_layers}:")
        direction_similarity = torch.nn.functional.cosine_similarity(desired_direction, undesired_direction, dim=0)
        print(f"- Cosine similarity between desired_direction and undesired_direction: {direction_similarity}")
        if layer_index > 0:
            householder_similarity = torch.nn.functional.cosine_similarity(householder_vector, householder_vectors[-1], dim=0)
            print(f"- Cosine similarity between current householder_vector and previous householder_vector: {householder_similarity}")
        print()
        
        householder_vectors.append(householder_vector)

    # Free memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Reload the model in CPU memory with bfloat16 data type
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='cpu'
    )
    model.requires_grad_(False)

    # Get the language model component and check it's as expected.
    lm_model = model.model
    assert hasattr(lm_model, 'layers'), "The model does not have the expected structure."

    # Check the ranges are valid.
    assert SKIP_BEGIN_LAYERS >= 0, "SKIP_BEGIN_LAYERS must be >= 0."
    assert SKIP_END_LAYERS >= 0, "SKIP_END_LAYERS must be >= 0."
    assert SKIP_BEGIN_LAYERS + SKIP_END_LAYERS < num_layers, "SKIP_BEGIN_LAYERS + SKIP_END_LAYERS must be < num_layers."

    bar_tensors = tqdm(total= (num_layers - (SKIP_BEGIN_LAYERS + SKIP_END_LAYERS)) * (SKIP_ATTN + SKIP_MLP), desc = "Modifying tensors")

    # By performing a (left-only) Householder transformation we reflect the matrix in the row space (ie: the linear weighted sums / "units").
    # NOTE: Down cast back to bfloat16 to save out in the same format as the un-modified tensors.
    def modify_tensor(weight_matrix, householder_matrix):
        weight_matrix = torch.matmul(householder_matrix, weight_matrix).to(torch.bfloat16)
        bar_tensors.update(1)
        return torch.nn.Parameter(weight_matrix)

    # Modify the 'self_attn.o_proj.weight' and 'mlp.down_proj.weight' in each chosen layer.
    # NOTE: These tensors names are speific to "llama" and may need changing.
    #       - See here for others: https://github.com/arcee-ai/mergekit/tree/main/mergekit/_data/architectures
    for layer_index in range(SKIP_BEGIN_LAYERS, num_layers - SKIP_END_LAYERS):
        
        # Ensure the householder vector is on the correct device and in float32 precision
        householder_vector = householder_vectors[layer_index].to(torch.float32)
        if householder_vector.device != model.device:
            householder_vector = householder_vector.to(model.device)
        
        # Calculate the Householder matrix for this layer in float32 precision
        identity_matrix = torch.eye(householder_vector.size(0), dtype=torch.float32)
        outer_product_matrix = torch.outer(householder_vector, householder_vector)
        householder_matrix = identity_matrix - 2 * outer_product_matrix

        # Modify this layer's attention projection and/or MLP projection matrices
        if not SKIP_ATTN:
            lm_model.layers[layer_index].self_attn.o_proj.weight = modify_tensor(
                lm_model.layers[layer_index].self_attn.o_proj.weight.data.to(torch.float32), householder_matrix
            )
        if not SKIP_MLP:
            lm_model.layers[layer_index].mlp.down_proj.weight = modify_tensor(
                lm_model.layers[layer_index].mlp.down_proj.weight.data.to(torch.float32), householder_matrix
            )

    bar_tensors.close()

    # Save the modified model and original tokenizer
    print("Saving modified model (with original tokenizer)...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify and save a model based on baseline, desired and undesired instructions.")
    parser.add_argument("--model_id", type=str, required=True, help="The model ID to load the pretrained model from.")
    parser.add_argument("--output_path", type=str, required=True, help="The path to save the modified model and tokenizer.")

    args = parser.parse_args()
    main(args.model_id, args.output_path)
