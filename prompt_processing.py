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
    
    gemma_prompt = f"""Please segment the following script into a list of prompts. Each prompt will be used to generate an image in a diffusion model.\n
{character_info}\n{story_str}
    
\nYour output should follow the following format: "...", "...", "..."
In this format, please use a quotation mark around each sentence and print a comma between each two sentence. \n
The prompt should follow the following rules:\n
1. Use simple language and words. \n
2. In each prompt, please parse the story in the correct way to understand it, retain all the full names (please show both first name and last name). Please avoid using any form of pronouns and ambiguous phrases in the text. This includes common personal pronouns such as 'I', 'you', 'he', 'she', 'it', 'they', as well as any expressions that could be unclear, like 'those two people', 'that object'. Ensure each description specifically identifies its subject, using exact full names including first name and last name or clear descriptions, to maintain clarity and independence in the content.\n
3.Your prompt should follow the development of the story. Each prompt should correspond to one event in the story. \n
4.You do not need to expand the story. Please do not add any details that aren't mentioned. \n
5. All you need to do is segment the long text into small self-contained sentences. Please make each sentence the smallest unit, i.e. encapsulating one simple action. \n
6. Please only output the prompt in the format I told you and don't output anything else.
"""
    # print(gemma_prompt)
    response = gemma_client.create_chat_completion(gemma_prompt)
    print(response)
    return response

def prompts_parse(prompts_text, metadata):
    # Replace names with signal words before extracting prompts
    for name, data in metadata.items():
        signal_word = data.get('signal_word', '')
        prompts_text = prompts_text.replace(name, signal_word)
    
    # Extract prompts and append " animated, high res" to each
    prompts = [prompt + " animated, high res" for prompt in re.findall(r'"(.*?)"', prompts_text)]
    return prompts
    
    
    
    
    
story = "Chris Martin runs a small music shop. Tim William loves to play the piano and they are good friends. One day, they went to see Lily Swift's concert. They three become good friends. Use simple language."

metadata = {"Chris Martin":{"signal_word":"*v1 *v2"}, "Tim William":{"signal_word":"*v3 *v4"}, "Lily Swift": {"signal_word":"*v5 *v6"}}

story_to_prompts(story, metadata)

text = '"Chris Martin smiles warmly at Tim William.", "Tim William thumbs through a new piano sheet.", "Their laughter echoes in the quiet shop.", "Chris Martin places a hand on Tim William’s shoulder.", "They share a secret grin.", "Tim William confides in Chris Martin about his dream of playing on a bigger stage.", "They walk excitedly towards the city’s music venue.", "Lily Swift’s mesmerizing voice fills the concert hall.", "Chris Martin, Tim William and Lily Swift exchange joyful smiles.", "The three of them become fast friends."'

# prompts_parse(text, metadata)