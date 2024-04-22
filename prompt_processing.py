from gemini import OpenRouter
import re

def gemma_client_init():
    api_key = "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"
    gemma_client = OpenRouter(api_key=api_key, model="google/gemma-7b-it:free")
    # gemma_client = OpenRouter(api_key=api_key, model="mistralai/mistral-7b-instruct:free")
    
    return gemma_client

def story_to_prompts(story, metadata):
    gemma_client = gemma_client_init()
    
    character_info = "\n".join(f"Character {index + 1}: {name}" for index, name in enumerate(metadata))
    story_str = f"\nStory: {story}"
    
    gemma_prompt = f"""Please segment the following script into a list of prompts. Each prompt will be used to generate an image in a stable diffusion model.\n
{character_info}\n{story_str}
Very important: Use only the full names ("Taylor Swift" remains "Taylor Swift"), do not use pronouns (for example, 'I', 'you', 'he', 'she', 'it', 'they') as well as unclear references like 'those two people', 'that object'
Change any pronouns into full names. 
\nYour output should follow the following format: "...", "...", "..."
In this format, please use a quotation mark around each sentence and print a comma between each two sentence. \n

The prompt should follow the following rules:\n
1. Use simple language and words. \n
2.Your prompt should follow the development of the story. Each prompt should correspond to one event in the story. \n
3.You do not need to expand the story. Please do not add any details that aren't mentioned. \n
4. All you need to do is segment the long text into small self-contained sentences. Please make each sentence the smallest unit, i.e. encapsulating one simple action. \n
5. Please only output the prompt in the format I told you and don't output anything else.
"""
    # print(gemma_prompt)
    response = gemma_client.create_chat_completion(gemma_prompt)
    # print(response)
    return response

def prompts_parse(prompts_text, metadata, style):
    # Replace names with signal words before extracting prompts
    for name, data in metadata.items():
        signal_word = data.get('signal_word', '')
        prompts_text = prompts_text.replace(name, signal_word)
    
    style_str = ", ".join(style)
    # Extract prompts and append " animated, high res" to each
    # prompts = ["A photo of " + prompt + ", animated, high resolution" for prompt in re.findall(r'"(.*?)"', prompts_text)]
    prompts = ["a Picture of " + prompt + style_str for prompt in re.findall(r'"(.*?)"', prompts_text)]

    return prompts
    