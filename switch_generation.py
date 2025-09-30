# patch-based collaborative generation with n language mdoels, steered by the n+1-th LLM
# as the next-patch selector. always starts with the RLed model. end with the RLed model
# with "The answer is" when necessary? next-patch selector decision could be overridden
# by manual specification (for simulation and training).

import torch
import random
from tqdm import tqdm
from multiprocessing import Pool
from transformers import AutoModelForCausalLM, AutoTokenizer

# base_model_path = "meta-llama/Llama-3.1-8B"
# fine_tuned_model_path = "allenai/Llama-3.1-Tulu-3-8B-SFT"
# aligned_model_path = "allenai/Llama-3.1-Tulu-3-8B"

def generate_text(model, tokenizer, prompt, max_length=50, do_sample=True, top_p=0.9, temperature=0.7):
    # return the generated text from the model given a prompt
    if model == None and tokenizer == None:
        model = selector_model
        tokenizer = selector_tokenizer
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=max_length,
        num_return_sequences=1,
        do_sample=do_sample,
        top_p=top_p,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask = inputs.attention_mask
    )
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text

def batch_generate_text(model, tokenizer, prompts, batch_size=8, max_length=50, do_sample=True, top_p=0.9, temperature=0.7):
    # return the generated text from the model given a list of prompts
    all_generated_texts = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        # try to apply chat template
        try:
            chat_prompts = []
            for prompt in batch_prompts:
                chat = [
                    {"role": "user", "content": prompt}
                ]
                chat_prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                chat_prompts.append(chat_prompt)
        except:
            chat_prompts = None
        batch_prompts = chat_prompts if chat_prompts else batch_prompts

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_length,
            num_return_sequences=1,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask = inputs.attention_mask
        )
        generated_texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        all_generated_texts.extend(generated_texts)
    return all_generated_texts

def load_models(model_paths, gpu_ids):
    # load the models one for each GPU
    global model_list, tokenizer_list, selector_model, selector_tokenizer
    model_list = [0] * len(model_paths)
    tokenizer_list = [0] * len(model_paths)
    selector_model = None
    selector_tokenizer = None

    for i, model_path in enumerate(model_paths):
        model_list[i] = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=f"cuda:{gpu_ids[i]}"
        )
        try:
            tokenizer_list[i] = AutoTokenizer.from_pretrained(model_path)
            # set padding token
            tokenizer_list[i].pad_token = tokenizer_list[i].eos_token
            # set left padding
            tokenizer_list[i].padding_side = "left"
        except:
            tokenizer_list[i] = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
            # set padding token
            tokenizer_list[i].pad_token = tokenizer_list[i].eos_token
            # set left padding
            tokenizer_list[i].padding_side = "left"

def load_selector_model(model_path, gpu_id):
    # load the selector model
    global selector_model, selector_tokenizer
    selector_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=f"cuda:{gpu_id}"
    )
    try:
        selector_tokenizer = AutoTokenizer.from_pretrained(selector_model)
    except:
        selector_tokenizer = AutoTokenizer.from_pretrained("allenai/Llama-3.1-Tulu-3-8B")
    selector_tokenizer.pad_token = selector_tokenizer.eos_token
    selector_tokenizer.padding_side = "left"

def selector_model_prompt(generation_log):
    # return random.choice(range(len(model_paths)))  # placeholder for the selector model logic

    selector_prompt = generation_log["query"]
    assert len(generation_log["generated_segments"]) == len(generation_log["segment_model_index"]), "Generated segments and model indices must match"
    for i in range(len(generation_log["generated_segments"])):
        selector_prompt += " <model " + str(generation_log["segment_model_index"][i]) + " begins> " + generation_log["generated_segments"][i] + " <model " + str(generation_log["segment_model_index"][i]) + " ends>"
    selector_prompt += " Which model should generate the next segment? Please respond with a number from 0 to " + str(len(model_list) - 1) + ". The answer is model "

    # print(selector_prompt)
    return selector_prompt

def distributed_generation(list_of_prompt_list, batch_size=8, max_length=50, do_sample=True, top_p=0.9, temperature=0.7):
    # generate text in a distributed manner
    # list_of_prompt_list: list of lists of prompts, each sublist corresponds to a model

    assert len(list_of_prompt_list) == len(model_list), "Number of prompt lists must match number of models"
    
    generation_args = []

    for i in range(len(model_list)):
        generation_args.append((
            model_list[i],
            tokenizer_list[i],
            list_of_prompt_list[i],
            batch_size,
            max_length,
            do_sample,
            top_p,
            temperature
        ))
    
    pool = Pool(len(model_list))
    list_of_output_list = pool.starmap(batch_generate_text, generation_args)
    pool.close()
    pool.join()

    assert len(list_of_output_list) == len(model_list), "Output list length must match model list length"
    return list_of_output_list

def switch_generation(prompts, batch_size=8, total_max_length=1024, max_length_per_segment=50, do_sample=True, top_p=0.9, temperature=0.7, objective_flag=False, random_selection=False, force_select_model_id=None):
    # switch generation function to handle the collaborative generation
    # prompts: list of prompts

    generation_logs = []
    for i in range(len(prompts)):
        generation_logs.append({
            "query": prompts[i],
            "generated_segments": [],
            "segment_model_index": [],
            "generated_sequence": ""
        })

    for round_id in tqdm(range(total_max_length // max_length_per_segment)):
        # selector model decides which model generates for which
        which_model = []
        if force_select_model_id is not None:
            # if force_select_model_id is specified, use it for all prompts
            which_model = [force_select_model_id] * len(prompts)
        if not random_selection and (round_id == 0 or round_id == total_max_length // max_length_per_segment - 1) and len(model_list) == 3:
            which_model = [2] * len(prompts)  # first and last round always use the aligned model
        else:
            if random_selection:
                # randomly select a model for each prompt
                which_model = [random.choice(range(len(model_list))) for _ in range(len(prompts))]
            else:
                selector_prompts = []
                for i in range(len(prompts)):
                    selector_prompts.append(selector_model_prompt(generation_logs[i]))
                selector_outputs = batch_generate_text(selector_model, selector_tokenizer, selector_prompts, batch_size=batch_size, max_length=max_length_per_segment, do_sample=do_sample, top_p=top_p, temperature=temperature)
                
                # for i in range(len(prompts)):
                #     print(f"Selector prompt: {selector_prompts[i]}")
                #     print(f"Selector output: {selector_outputs[i]}")
                
                for i in range(len(prompts)):
                    if "0" in selector_outputs[i]:
                        which_model.append(0)
                    elif "1" in selector_outputs[i]:
                        which_model.append(1)
                    elif "2" in selector_outputs[i]:
                        which_model.append(2)
                    else:
                        # if the selector model does not return a valid model index, default to the aligned model
                        # print("NOT FOUND!")
                        which_model.append(2)

                for k in range(len(which_model)):
                    # in case of errors, randomly select a model
                    if which_model[k] >= len(model_list):
                        which_model[k] = random.choice(range(len(model_list)))

        # print(which_model)

        # for i in range(len(prompts)):
        #     if round_id == 0 or round_id == total_max_length // max_length_per_segment - 1:
        #         # first and last round always use the aligned model
        #         which_model.append(2)
        #     else:
        #         which_model.append(selector_model_decision(generation_logs[i]))
        
        # assemble list_of_prompt_list by model
        list_of_prompt_list = [[] for _ in range(len(model_list))]
        for i in range(len(prompts)):
            if round_id == total_max_length // max_length_per_segment - 1 and objective_flag:
                # last round, add the objective prompt
                generation_logs[i]["generated_segments"][-1] += " The final answer is"
                generation_logs[i]["generated_sequence"] += " The final answer is"
            context = generation_logs[i]["query"] + " " + generation_logs[i]["generated_sequence"]
            list_of_prompt_list[which_model[i]].append(context)

        # generate text distributedly
        list_of_output_list = distributed_generation(list_of_prompt_list, batch_size, max_length_per_segment, do_sample, top_p, temperature)

        # update generation logs
        for i in range(len(prompts)):
            segment = list_of_output_list[which_model[i]][0]
            list_of_output_list[which_model[i]] = list_of_output_list[which_model[i]][1:]  # remove the first segment
            generation_logs[i]["generated_segments"].append(segment)
            generation_logs[i]["segment_model_index"].append(which_model[i])
            generation_logs[i]["generated_sequence"] += segment + " "
        # assert list_of_output_list is empty at the end
        assert all(len(output) == 0 for output in list_of_output_list), "Output lists should be empty after generation"

    # return the final generated sequences
    final_generated_sequences = []
    for i in range(len(prompts)):
        final_generated_sequences.append(generation_logs[i]["generated_sequence"].strip())

    return final_generated_sequences, generation_logs


# if __name__ == "__main__":
#     torch.multiprocessing.set_start_method('spawn')

#     model_paths = ["meta-llama/Llama-3.1-8B", 
#                "allenai/Llama-3.1-Tulu-3-8B-SFT", 
#                "allenai/Llama-3.1-Tulu-3-8B"]

#     model_list = [0] * len(model_paths)
#     tokenizer_list = [0] * len(model_paths)
#     selector_model = None
#     selector_tokenizer = None

#     load_models(model_paths, [0,1,2])

#     load_selector_model("allenai/Llama-3.1-Tulu-3-8B", 3)


#     prompts = [
#         "Former U.S. Representative, with ties to the Tea Party movement, who lost re-election. His name is Paul what?",
#         "Who is the largest company in the world by market cap that doesn’t sell anything directly to consumers?"
#     ] * 25
#     final_sequences, _ = switch_generation(prompts, batch_size=32, total_max_length=128, random_selection=False)
#     print(final_sequences)