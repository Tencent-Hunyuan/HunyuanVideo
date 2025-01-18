normal_mode_prompt = """Normal mode - Video Recaption Task:

You are a large language model specialized in rewriting video descriptions. Your task is to modify the input description.

0. Preserve ALL information, including style words and technical terms.

1. If the input is in Chinese, translate the entire description to English. 

2. If the input is just one or two words describing an object or person, provide a brief, simple description focusing on basic visual characteristics. Limit the description to 1-2 short sentences.

3. If the input does not include style, lighting, atmosphere, you can make reasonable associations.

4. Output ALL must be in English.

Given Input:
input: "{input}"
"""


master_mode_prompt = """Master mode - Video Recaption Task:

You are a large language model specialized in rewriting video descriptions. Your task is to modify the input description.

0. Preserve ALL information, including style words and technical terms.

1. If the input is in Chinese, translate the entire description to English. 

2. To generate high-quality visual scenes with aesthetic appeal, it is necessary to carefully depict each visual element to create a unique aesthetic.

3. If the input does not include style, lighting, atmosphere, you can make reasonable associations.

4. Output ALL must be in English.

Given Input:
input: "{input}"
"""

def get_rewrite_prompt(ori_prompt, mode="Normal"):
    if mode == "Normal":
        prompt = normal_mode_prompt.format(input=ori_prompt)
    elif mode == "Master":
        prompt = master_mode_prompt.format(input=ori_prompt)
    else:
        raise Exception("Only supports Normal and Normal", mode)
    return prompt

ori_prompt = "一只小狗在草地上奔跑。"
normal_prompt = get_rewrite_prompt(ori_prompt, mode="Normal")
master_prompt = get_rewrite_prompt(ori_prompt, mode="Master")

# Then you can use the normal_prompt or master_prompt to access the hunyuan-large rewrite model to get the final prompt.
