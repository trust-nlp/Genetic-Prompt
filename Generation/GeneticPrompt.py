import openai
import asyncio
from typing import List, Dict, Any
import argparse
import os 
from tqdm import  tqdm
import re 
import time
import json 
import pandas as pd
from sampling import SampleManager
from openai import AsyncOpenAI
import random


def shuffle_attributes(attributes):
    shuffled_attributes = attributes.copy()
    seed= int(time.time())
    random.seed(seed)  
    random.shuffle(shuffled_attributes)

    return shuffled_attributes


parser = argparse.ArgumentParser("")
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--temperature", default=1, type=float)
parser.add_argument("--parents", type=str, default="active")

parser.add_argument("--top_p", default=0.95, type=float, help="nucleus sampling")
parser.add_argument("--n_sample", default=10, type=int, help="the number of examples generated for each class")
parser.add_argument("--batch_size", default=20, type=int, help="")

parser.add_argument("--dataset", default='agnews', type=str, help="which data to generate")

parser.add_argument("--model_name", default='gpt-3.5-turbo', type=str, help="which model to use")
parser.add_argument("--max_tokens", default=1024, type=int, help="max length")
parser.add_argument("--output_dir", default='.', type=str, help="the folder for saving the generated text")
parser.add_argument("--input_dir", default='.', required=False, type=str, help="the file for getting the input in-context learning text")

args = parser.parse_args()

if args.model_name=="deepseek-chat":
    args.api_key = ""
    args.openai_api_base= "https://api.deepseek.com"
elif args.model_name in ["gpt-3.5-turbo", "gpt-4o"]:
    args.api_key = ''
    args.openai_api_base= "https://api.openai.com/v1"
else:
    args.api_key= "token-abc123"
    args.openai_api_base="http://localhost:8000/v1"
    
client = AsyncOpenAI(
    api_key=args.api_key,
    base_url=args.openai_api_base,
)

print(client)

if args.dataset in ['chemprot']:
    args.entity = ['chemical', 'protein']
    args.label_names=['substrate','upregulator', 'downregulator', 'agonist', 'antagonist']
    args.attributes = ["length", "voice","sentence_structure" ,"interaction_verb","modifier","negation" ,"Entity Proximity"]
    
    """
    length:Length of the sentence
    
    voice:Whether the text is written in active or passive

    sentence_structure:Complexity of the sentence
    
    modifier: Adjectives or adverbs that modify the interaction, such as "slightly," "significantly," or "rarely," which can affect the strength or likelihood of interaction.
    negation: Whether the sentence includes negation
    
    interaction_verb: The verb that describes the interaction between the chemical and protein
    
    """
    args.label_def = {
        "upregulator": "the chemical Activates expression of the protein",
        "downregulator": "the chemical inhibits expression of the protein",
        "agonist": "the chemical triggering a biological response similar to the natural ligand",
        "antagonist": "the chemical diminishing its normal activity or interaction with its natural ligand.",
        "substrate": "the chemical is acted upon or modified by the protein, typically an enzyme",
    }
    args.domain = 'Protein Chemical Relation extraction'
    args.input_dir = ''
    
    
elif args.dataset in ['ddi']:
    args.entity = ['drug', 'drug']
    args.domain = 'Drug-Drug Interaction extraction'
    args.attributes = ["length", "voice","polarity","interaction_verb","modifier","drug mentions","Entity Proximity"]
    """
    Drug Mentions: The specific names of drugs or substances and how they are referenced. Whether the text uses brand names, generic names, or even chemical names
    voice: Whether the text is written in active or passive
    modifier: Adjectives or adverbs that modify the interaction, such as "slightly," "significantly," or "rarely," which can affect the strength or likelihood of interaction.
    polarity: Whether the sentence conveys a positive, neutral, or negative interaction between the chemical and protein
    Entity Proximity: The distance between the two entities in the sentence
    """
    args.label_names=[ "effect","mechanism", "advise", "int"]
    args.label_def = {
        "mechanism":"One drug modifies the biological or pharmacological mechanism of another drug.",
        "effect":  "One drug alters the therapeutic or side effects of another drug.",
        "advise": "A recommendation or warning based on the interaction between two drugs, often advising dosage adjustment or avoidance.",
        "int": "A general interaction between two drugs is present, without specifying the exact nature or type of interaction."
     }
    args.input_dir = ''
    
    pass   
elif args.dataset in ['cdr']:
    pass
elif args.dataset in ['gad']:
    pass



elif args.dataset in ['scierc']:
    args.input_dir = ''
    args.label_names=["Conjunction","Feature-Of","Hyponym-Of","Used-For","Part-Of","Compare","Evaluate-For"] 
    args.attributes = ["length", "voice","Sentence Structure","Readability","Entity distance","research domain"]
    args.label_def = {"Conjunction": "Entities as similar role or use/incorporate with.",
                      "Feature-Of": "e1 belongs to e2, e1 is a feature of e2, e1 is under e2 domain",
                      "Hyponym-Of": "e1 is a  hyponym of e2, e1 is a  type of e2",
                      "Used-For": "e1 is used for e2, e1 models e2, e2 is trained on e1, e1 exploits e2, ",
                      
                      "Part-Of": "e1 is a part of e2.",
                      "Compare": "e1 is compared to e2.",
                      "Evaluate-For": "e1 is evaluated for e2."}
    args.domain="Computer Science entity relation classification"
                      
    args.entity=["e1", "e2"]


elif args.dataset in ['semeval2010']:
    args.input_dir = ''
    args.attributes = ["length", "voice","Sentence Structure","Readability","Entity distance","domain"]
    args.domain="Relation Classification"
    args.label_names=["Cause-Effect","Component-Whole","Content-Container","Entity-Destination","Entity-Origin","Instrument-Agency","Member-Collection","Message-Topic","Product-Producer"]
    args.entity = ['entity', 'entity']
    args.label_def = {"Cause-Effect": "An entity causes an effect to another entity",
                      "Component-Whole": "An entity is a component of a whole",
                      "Entity-Destination": "An entity is transported to a destination",
                        "Product-Producer": "An entity is a product of a producer",
                        "Entity-Origin": "An entity is coming or derived from an origin",
                        "Member-Collection": "An entity is a member of a collection",
                        "Message-Topic": "An entity is a message topic",
                        "Content-Container": "An entity is a content of a container",
                        "Instrument-Agency": "An agent uses an instrument",}
    
elif args.dataset in ['conll04']:
    args.input_dir = ''
    args.attributes = ["length", "voice","Sentence Structure","Readability","Entity distance","domain"]
    args.label_names=["Kill","Live_In","Located_In","OrgBased_In","Work_For"]
    args.entity = ['entity', 'entity']
    args.domain="Relation Classification"
    args.label_def = {
        "Kill": "The entity kills another entity",
        "Live_In": "The person lives in a location",
        "Located_In": "The entity is located in a location",
        "OrgBased_In": "The organization is based in a location",
        "Work_For": "The person works for an organization"
    }
    
    """
    {
    "Kill": 0,
    "Live_In": 1,
    "Located_In": 2,
    "OrgBased_In": 3,
    "Work_For": 4
    }
    """



    
    
elif args.dataset in ['agnews']:
    args.input_dir = ''
    args.attributes = ["length", "location","style","subtopics"]
    args.label_names=[ "World", "Sports", "Business", "Sci_Tech"]
    

    args.label_def = {
        "World": "World news",
        "Sports": "Sports news",
        "Business": "Business news",
        "Sci_Tech": "Science and Technology news"
    }
    args.domain="News Classification"
elif args.dataset in ["stackexchange"]:
    args.input_dir = ''
    args.attributes = ["depth", "length","scenario","style"]

    args.label_names=["unity", "opengl", "javascript", "animation", "game-design", "procedural-generation", "physics", \
                      "android", "graphics", "cocos2d-iphone", "tools", "terrain", "xna", "rendering", "shaders", "libgdx", \
                      "path-finding", "mmo", "collision-detection", "sdl", "blender", "software-engineering", "box2d", "udk", \
                      "camera", "legal", "multiplayer", "lua", "algorithm", "tilemap", "game-AI", "networking", "flash", "sprites",\
                        "textures", "minecraft-modding", "rotation", "iphone", "game-loop", "geometry", "directx11", "actionscript-3", \
                        "unreal-4", "gui", "godot", "maps", "server", "entity-system", "3d-meshes", "marketing"]
    args.label_def = {
        "unity": "Unity game engine",
        "opengl": "OpenGL graphics library",
        "javascript": "JavaScript programming language",
        "animation": "Animation techniques",
        "game-design": "Game design principles",
        "procedural-generation": "Procedural generation techniques",
        "physics": "Physics simulation in games",
        "android": "Android mobile platform",
        "graphics": "Graphics rendering techniques",
        "cocos2d-iphone": "Cocos2D game engine for iPhone",
        "tools": "Game development tools",
        "terrain": "Terrain generation and manipulation",
        "xna": "XNA game development framework",
        "rendering": "Rendering techniques",
        "shaders": "Shader programming",
        "libgdx": "LibGDX game development framework",
        "path-finding": "Pathfinding algorithms",
        "mmo": "Massively multiplayer online games",
        "collision-detection": "Collision detection techniques",
        "sdl": "Simple DirectMedia Layer",
        "blender": "Blender 3D modeling software",
        "software-engineering": "Software engineering principles",
        "box2d": "Box2D physics engine",
        "udk": "Unreal Development Kit",
        "camera": "Camera techniques in games",
        "legal": "Legal issues in game development",
        "multiplayer": "Multiplayer game development",
        "lua": "Lua programming language",
        "algorithm": "Game algorithms",
        "tilemap": "Tile-based game development",
        "game-AI": "Game artificial intelligence",
        "networking": "Networking in games",
        "flash": "Adobe Flash game development",
        "sprites": "Sprite-based game development",
        "textures": "Texture mapping techniques",
        "minecraft-modding": "Minecraft modding",
        "rotation": "Rotation techniques in games",
        "iphone": "iPhone game development",
        "game-loop": "Game loop design",
        "geometry": "Geometry in game development",
        "directx11": "DirectX 11 graphics API",
        "actionscript-3": "ActionScript 3 programming language",
        "unreal-4": "Unreal Engine 4",
        "gui": "Graphical user interface design",
        "godot": "Godot game engine",
        "maps": "Map design and development",
        "server": "Game server development",
        "entity-system": "Entity-component system architecture",
        "3d-meshes": "3D mesh modeling",
        "marketing": "Game marketing strategies"
    }
    args.domain="Stack Exchange question classification"
 

 
 
    

elif args.dataset in ["scitldr"]:
    args.input_dir = ''
    args.attributes = ["length", "depth","style","subtopics"]
    args.label_names=["None"]
    args.label_def = { 
        "None": "No label"
    }
    args.domain="Computer science paper abstract summarization"
elif args.dataset in ["mediqa"]:
    args.input_dir = ''
    args.attributes = ["length", "depth","style","subtopics"]
    args.label_names=["None"]
    args.label_def = { 
        "None": "No label"
    }
    args.domain="medical question summarization"
    
else:
    raise NotImplementedError


async def dispatch_openai_requests(
    messages_list: List[List[Dict[str, Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> List[str]:
    async_responses = [
        client.chat.completions.create(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    results = await asyncio.gather(*async_responses)
    return [res.choices[0].message.content for res in results]  





def call_api_async(msg_lst, model, temperature, max_tokens):
    print("===================================")
    print(f"call APIs, {len(msg_lst)} in total, t= {temperature}.")

    response = asyncio.run(
        dispatch_openai_requests(
            messages_list=msg_lst,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
        )
    )

    print(f"API returns {len(response)} in total.")
    print("===================================")
    return response

def main(args):

    label_names = args.label_names
    print(label_names)
    model = args.model_name
    
    openai.api_key = args.api_key
    
    if args.input_dir.endswith('.jsonl') or args.dataset in ['yelp', 'wikiqa','scitldr']:
        df = pd.read_json(args.input_dir, lines=True)
    elif args.input_dir.endswith('.json'):        
        df = pd.read_json(args.input_dir, lines=False) 
    
    for i, class_name in tqdm(enumerate(label_names)): 


        print(i, class_name)
        label_def = args.label_def[class_name]

        manager = SampleManager()
        if args.dataset == 'wikiqa':
            filtered_df = df[df['label'] == i]
        elif args.dataset in ["scitldr","mediqa"]:
            filtered_df = df
        else:
            filtered_df = df[df['_id'] == i]
            
        filtered_df = filtered_df.head(500)
        if args.dataset == 'i2b2-2010':
            texts = filtered_df['text_short'].tolist()
        elif args.dataset == 'wikiqa':
            question = filtered_df['question'].tolist()
            answer = filtered_df['answer'].tolist()
            texts = [f"Question: {q} Answer: {a}" for q, a in zip(question, answer)]
        elif args.dataset == 'scitldr':
            source= filtered_df['source'].tolist()
            target= filtered_df['target'].tolist()
            texts = [f"Abstract: {s} Summary: {t}" for s, t in zip(source, target)]
        elif args.dataset == 'mediqa':
            question = filtered_df['CHQ'].tolist()
            summary = filtered_df['Summary'].tolist()
            texts = [f"Question: {q} Summary: {s}" for q, s in zip(question, summary)]
        else:
            texts = filtered_df['text'].tolist()

        print("text",len(texts))
        for _ in tqdm([1], desc="Adding samples"): 
            manager.add_samples(texts)        
            
        if args.dataset == 'wikiqa':
            if i ==0:
                args.n_sample = args.n_sample
                
            if i ==1:   
                args.n_sample = int(args.n_sample//4)
        
        
        print(f"Class {i}: {class_name}, {len(texts)} samples.")
        
        

        
        sent_cnt = 0 
        attempt = 0 
        prompt_lst = []
        #examples = []


        for j in range(args.n_sample): 
        

            if args.parents=="random":
                sample1=manager.get_random_sample()
                sample2=manager.get_random_sample()
            elif args.parents=="active":
                in_context_pair,distance = manager.find_least_similar_pair()
                sample1= manager.get_sample(in_context_pair[0])
                sample2= manager.get_sample(in_context_pair[1])
            else:
                print("need parents selection method")
                exit()
       

            if args.dataset ==  'chemprot':

                prompt_input =f"You need to generate synthetic data for the {args.domain} task. \n"
                in_context_input= f"Example 1: {sample1} \n Example 2: {sample2} \n" 
                

                Genetic_prompt = f"""
                                    Your task is to write a sentence about '{class_name}' relation between {args.entity[0]} and {args.entity[1]}. \n
                                    The two entities should be marked with XML-style tags as <{args.entity[0]}> {args.entity[0]} </{args.entity[0]}> and <{args.entity[1]}> {args.entity[1]} </{args.entity[1]}> respectively in your response.
                                    The sentence should follow the requirements below: \n 
                                        1. The sentence must discuss about the {args.entity[0]} and {args.entity[1]} with the relation {label_def}  \n 
                                        2. The {re.sub('_', ' ', args.attributes[0])}' of the sentence should inherit from Example1;\n 
                                        3. The {re.sub('_', ' ', args.attributes[1])}' of the sentence should inherit from Example2;\n 
                                        4. The {re.sub('_', ' ', args.attributes[2])}' of the sentence should inherit from Example1;\n 
                                        5. The {re.sub('_', ' ', args.attributes[3])}' of the sentence should inherit from Example2;\n 
                                        6. The {re.sub('_', ' ', args.attributes[4])}' of the sentence should inherit from Example1;\n 
                                        7. The {re.sub('_', ' ', args.attributes[5])}' of the sentence should inherit from Example2;\n 
                                        8. The {re.sub('_', ' ', args.attributes[6])}' and entities must be different from the given 2 examples.
                                        """
                Genetic_prompt += f'Relation: {class_name}\n'
                Genetic_prompt += f'Text:'                            
                                        
                                                                                        
                            
                

            elif args.dataset ==  'ddi':
                prompt_input =f"You need to generate synthetic data for {args.domain} task. \n"
                in_context_input= f"Example 1: {sample1} \n Example 2: {sample2} \n" 
                

                Genetic_prompt = f"""
                                    Your task is to write a sentence about '{class_name}' relation between {args.entity[0]} and {args.entity[1]}. \n 
                                    The two {args.entity[0]}s should be marked with XML-style tags as <{args.entity[0]}> {args.entity[0]} </{args.entity[0]}>.
                                    The sentence should follow the requirements below: \n 
                                        1. The sentence must discuss about the {args.entity[0]} and {args.entity[1]} with the relation {label_def}  \n 
                                        2. The {re.sub('_', ' ', args.attributes[0])}' of the sentence should inherit from Example1;\n 
                                        3. The {re.sub('_', ' ', args.attributes[1])}' of the sentence should inherit from Example2;\n 
                                        4. The {re.sub('_', ' ', args.attributes[2])}' of the sentence should inherit from Example1;\n 
                                        5. The {re.sub('_', ' ', args.attributes[3])}' of the sentence should inherit from Example2;\n 
                                        6. The {re.sub('_', ' ', args.attributes[4])}' of the sentence should inherit from Example1;\n 
                                        7. The {re.sub('_', ' ', args.attributes[5])}' of the sentence should inherit from Example2;\n 
                                        8. The {re.sub('_', ' ', args.attributes[6])}' and entities must be different from the given 2 examples.
                                        """
                Genetic_prompt += f'Relation: {class_name}\n'
                Genetic_prompt += f'Text:'
                
            elif args.dataset ==  'i2b2-2010':
                prompt_input =f"You need to generate synthetic data for {args.domain} task. \n"
                in_context_input= f"Example 1: {sample1} \n Example 2: {sample2} \n" 
                

                Genetic_prompt = f"""
                                    Your task is to write a sentence about '{class_name}' relation between {args.entity[0]} and {args.entity[1]}. \n 
                                    The two entities should be marked with XML-style tags as <{args.entity[0]}> {args.entity[0]} </{args.entity[0]}> and <{args.entity[1]}> {args.entity[1]} </{args.entity[1]}> respectively in your response.
                                    The sentence should follow the requirements below: \n 
                                        1. The sentence must discuss about the {args.entity[0]} and {args.entity[1]} with the relation {label_def}  \n 
                                        2. The {re.sub('_', ' ', args.attributes[0])}' of the sentence should inherit from Example1;\n 
                                        3. The {re.sub('_', ' ', args.attributes[1])}' of the sentence should inherit from Example2;\n 
                                        4. The {re.sub('_', ' ', args.attributes[2])}' of the sentence should inherit from Example1;\n 
                                        5. The {re.sub('_', ' ', args.attributes[3])}' of the sentence should inherit from Example2;\n 
                                        6. The {re.sub('_', ' ', args.attributes[4])}' of the sentence should inherit from Example1;\n 
                                        7. The {re.sub('_', ' ', args.attributes[5])}' of the sentence should inherit from Example2;\n 
                                        8. The {re.sub('_', ' ', args.attributes[6])}' and entities must be different from the given 2 examples.
                                        """
                Genetic_prompt += f'Relation: {class_name}\n'
                Genetic_prompt += f'Text:'
            elif args.dataset ==  'n2c2-2018':
                
                prompt_input =f"You need to generate synthetic data for {args.domain} task. \n"
                in_context_input= f"Example 1: {sample1} \n Example 2: {sample2} \n" 
                

                Genetic_prompt = f"""
                                    Your task is to write a sentence about '{class_name}' relation between {args.entity[0]} and {args.entity[1]}. \n 
                                    The {args.entity[0]} and {args.entity[1]} should be marked with XML-style tags as <{args.entity[0]}> {args.entity[0]} </{args.entity[0]}>. \n
                                    The '{class_name}' relation means the {args.entity[1]} is the {label_def}\n
                                    The sentence should follow the requirements below: \n 
                                        1. The {re.sub('_', ' ', args.attributes[0])}' of the sentence should inherit from Example1;\n 
                                        2. The {re.sub('_', ' ', args.attributes[1])}' of the sentence should inherit from Example2;\n 
                                        3. The {re.sub('_', ' ', args.attributes[2])}' of the sentence should inherit from Example1;\n 
                                        4. The {re.sub('_', ' ', args.attributes[3])}' of the sentence should inherit from Example2;\n 
                                        5. The {re.sub('_', ' ', args.attributes[4])}' of the sentence should inherit from Example1;\n 
                                        6. The {re.sub('_', ' ', args.attributes[5])}' of the sentence should inherit from Example2;\n 
                                        7. The {re.sub('_', ' ', args.attributes[6])}' and entities must be different from the given 2 examples.
                                        """
                Genetic_prompt += f'Relation: {class_name}\n'
                Genetic_prompt += f'Text:'
                
            elif args.dataset ==  'scierc':
                
                
                prompt_input =f"You need to generate synthetic data for {args.domain} task. \n"
                in_context_input= f"Example 1: {sample1} \n Example 2: {sample2} \n" 
                

                Genetic_prompt = f"""
                                    Your task is to write a sentence about '{class_name}' relation between {args.entity[0]} and {args.entity[1]}. \n 
                                    The '{class_name}' could be one of the relations in [{label_def}]\n
                                    The two entities should be marked with XML-style tags as <{args.entity[0]}> {args.entity[0]} </{args.entity[0]}> and <{args.entity[1]}> {args.entity[1]} </{args.entity[1]}> respectively in your response.
                                    The sentence should follow the requirements below: \n 
                                        1. The {re.sub('_', ' ', args.attributes[0])}' of the sentence should inherit from Example1;\n 
                                        2. The {re.sub('_', ' ', args.attributes[1])}' of the sentence should inherit from Example2;\n 
                                        3. The {re.sub('_', ' ', args.attributes[2])}' of the sentence should inherit from Example1;\n 
                                        4. The {re.sub('_', ' ', args.attributes[3])}' of the sentence should inherit from Example2;\n 
                                        5. The {re.sub('_', ' ', args.attributes[4])}' of the sentence should inherit from Example1;\n 
                                        6. The {re.sub('_', ' ', args.attributes[5])}' and entities must be different from the given 2 examples.
                                        """
                Genetic_prompt += f'Relation: {class_name}\n'
                Genetic_prompt += f'Text:'                
                
                
                
                
            elif args.dataset ==  'semeval2010':
                
                prompt_input =f"You need to generate synthetic data for {args.domain} task. \n"
                in_context_input= f"Example 1: {sample1} \n Example 2: {sample2} \n" 
                

                Genetic_prompt = f"""
                                    Your task is to write a sentence about '{class_name}' relation between 2 entities. \n 
                                    The '{class_name}' relation means {label_def}\n
                                    The two entities should be marked with XML-style tags as <{args.entity[0]}> {args.entity[0]} </{args.entity[0]}>.
                                    The sentence should follow the requirements below: \n 
                                        1. The {re.sub('_', ' ', args.attributes[0])}' of the sentence should inherit from Example1;\n 
                                        2. The {re.sub('_', ' ', args.attributes[1])}' of the sentence should inherit from Example2;\n 
                                        3. The {re.sub('_', ' ', args.attributes[2])}' of the sentence should inherit from Example1;\n 
                                        4. The {re.sub('_', ' ', args.attributes[3])}' of the sentence should inherit from Example2;\n 
                                        5. The {re.sub('_', ' ', args.attributes[4])}' of the sentence should inherit from Example1;\n 
                                        6. The {re.sub('_', ' ', args.attributes[5])}' and entities must be different from the given 2 examples.
                                        """
                Genetic_prompt += f'Relation: {class_name}\n'
                Genetic_prompt += f'Text:'                    
            elif args.dataset ==  'conll04':
                prompt_input =f"You need to generate synthetic data for {args.domain} task. \n"
                in_context_input= f"Example 1: {sample1} \n Example 2: {sample2} \n" 
                

                Genetic_prompt = f"""
                                    Your task is to write a sentence about '{class_name}' relation between 2 entities. \n 
                                    The '{class_name}' relation means {label_def}\n
                                    The two entities should be marked with XML-style tags as <{args.entity[0]}> {args.entity[0]} </{args.entity[0]}>.
                                    The sentence should follow the requirements below: \n 
                                        1. The {re.sub('_', ' ', args.attributes[0])}' of the sentence should inherit from Example1;\n 
                                        2. The {re.sub('_', ' ', args.attributes[1])}' of the sentence should inherit from Example2;\n 
                                        3. The {re.sub('_', ' ', args.attributes[2])}' of the sentence should inherit from Example1;\n 
                                        4. The {re.sub('_', ' ', args.attributes[3])}' of the sentence should inherit from Example2;\n 
                                        5. The {re.sub('_', ' ', args.attributes[4])}' of the sentence should inherit from Example1;\n 
                                        6. The {re.sub('_', ' ', args.attributes[5])}' and entities must be different from the given 2 examples.
                                        """
                Genetic_prompt += f'Relation: {class_name}\n'
                Genetic_prompt += f'Text:'    
            elif args.dataset == 'agnews':
                prompt_input = f"You need to generate synthetic data for {args.domain} task. \n"
                in_context_input = f"Example 1: {sample1} \n Example 2: {sample2} \n" 
                
                Genetic_prompt = f"""
                                    Your task is to write a news article about '{class_name}' category. \n
                                    The article should follow the requirements below: \n 
                                        1. The article must be a news about {label_def} \n 
                                        2. The {re.sub('_', ' ', args.attributes[0])}' of the article should inherit from Example1;\n 
                                        3. The {re.sub('_', ' ', args.attributes[1])}' of the article should inherit from Example2;\n 
                                        4. The {re.sub('_', ' ', args.attributes[2])}' of the article should inherit from Example1;\n 
                                        5. The {re.sub('_', ' ', args.attributes[3])}' must be different from the given 2 examples.\n 
                                        """
                
                Genetic_prompt += f'Category: {class_name}\n'
                Genetic_prompt += f'Text:'


            elif args.dataset == "stackexchange":
                prompt_input = f"You need to generate synthetic data for {args.domain} task. \n"
                in_context_input = f"Example 1: {sample1} \n Example 2: {sample2} \n" 
                
                Genetic_prompt = f"""
                                    Your task is to write a Stack Exchange question about '{class_name}'. \n
                                    The '{class_name}' category means {label_def}\n
                                    The question should follow the requirements below: \n 
                                        1. The {re.sub('_', ' ', args.attributes[0])}' of the question should inherit from Example1;\n 
                                        2. The {re.sub('_', ' ', args.attributes[1])}' of the question should inherit from Example2;\n 
                                        3. The {re.sub('_', ' ', args.attributes[2])}' of the question should inherit from Example1;\n 
                                        4. The {re.sub('_', ' ', args.attributes[3])}' must be different from the given 2 examples.\n 
                                        """
                
                Genetic_prompt += f'Category: {class_name}\n'
                Genetic_prompt += f'Text:'
            elif args.dataset == "sst-2":
                prompt_input = f"You need to generate synthetic data for {args.domain} task. \n"
                in_context_input = f"Example 1: {sample1} \n Example 2: {sample2} \n" 
                
                Genetic_prompt = f"""
                                    Your task is to write a movie review with '{class_name}' sentiment. \n
                                    The review should follow the requirements below: \n 
                                        1. The {re.sub('_', ' ', args.attributes[0])}' of the review should inherit from Example1;\n 
                                        2. The {re.sub('_', ' ', args.attributes[1])}' of the review should inherit from Example2;\n 
                                        3. The {re.sub('_', ' ', args.attributes[2])}' of the review should inherit from Example1;\n 
                                        4. The {re.sub('_', ' ', args.attributes[3])}' must be different from the given 2 examples.\n 
                                        """
                
                Genetic_prompt += f'Sentiment: {class_name}\n'
                Genetic_prompt += f'Text:'
            elif args.dataset == "yelp":
                prompt_input = f"You need to generate synthetic data for {args.domain} task. \n"
                in_context_input = f"Example 1: {sample1} \n Example 2: {sample2} \n" 
                
                Genetic_prompt = f"""
                                    Your task is to write a restaurant review with '{class_name}' sentiment. \n
                                    The review should follow the requirements below: \n 
                                        1. The {re.sub('_', ' ', args.attributes[0])}' of the review should inherit from Example1;\n 
                                        2. The {re.sub('_', ' ', args.attributes[1])}' of the review should inherit from Example2;\n 
                                        3. The {re.sub('_', ' ', args.attributes[2])}' of the review should inherit from Example1;\n 
                                        4. The {re.sub('_', ' ', args.attributes[3])}' must be different from the given 2 examples.\n 
                                        """
                
                Genetic_prompt += f'Sentiment: {class_name}\n'
                Genetic_prompt += f'Text:'

            elif args.dataset == "wikiqa":
                prompt_input = f"You need to generate synthetic data for {args.domain} task. \n"
                in_context_input = f"Example 1: {sample1} \n Example 2: {sample2} \n"
                Genetic_prompt = f"""
                                    Your task is to write a question-answer pair . \n
                                    The question-answer pair should follow the requirements below: \n
                                        1. The answer must be a {label_def} to the question.\n
                                        2. The {re.sub('_', ' ', args.attributes[0])}' of the pair should inherit from Example1;\n
                                        3. The {re.sub('_', ' ', args.attributes[1])}' of the pair should inherit from Example2;\n
                                        4. The {re.sub('_', ' ', args.attributes[2])}' of the pair should inherit from Example1;\n
                                        5. The {re.sub('_', ' ', args.attributes[3])}' must be different from the given 2 examples.\n
                                        """
                Genetic_prompt += f'Output:'

          
            elif args.dataset == "bioasq" :
                pass
            elif args.dataset == "scitldr" :
                prompt_input = f"You need to generate synthetic data for {args.domain} task. \n"
                in_context_input = f"Example 1: {sample1} \n Example 2: {sample2} \n" 
                Genetic_prompt = f"""
                                    Your task is to write computer science paper abstract and it's summary. \n
                                    The abstract and summary should follow the requirements below: \n
                                    1. The length of the summary should inherit from Example1;\n
                                    2. The {re.sub('_', ' ', args.attributes[0])}' of the abstract should inherit from Example1;\n
                                    3. The {re.sub('_', ' ', args.attributes[1])}' of the abstract should inherit from Example2;\n
                                    4. The {re.sub('_', ' ', args.attributes[2])}' of the abstract should inherit from Example1;\n
                                    5. The {re.sub('_', ' ', args.attributes[3])}' must be different from the given 2 examples.\n
                                    """    

                Genetic_prompt += f'Output:'
                
            elif args.dataset == "mediqa" :
                prompt_input = f"You need to generate synthetic data for {args.domain} task. \n"
                in_context_input = f"Example 1: {sample1} \n Example 2: {sample2} \n" 
                
                Genetic_prompt = f"""
                                    Your task is to write a medical question and it's summary. \n
                                    The question and summary should follow the requirements below: \n 
                                        1. The {re.sub('_', ' ', args.attributes[0])}' of the question should inherit from Example1;\n 
                                        2. The {re.sub('_', ' ', args.attributes[1])}' of the question should inherit from Example2;\n 
                                        3. The {re.sub('_', ' ', args.attributes[2])}' of the question should inherit from Example1;\n 
                                        4. The {re.sub('_', ' ', args.attributes[3])}' must be different from the given 2 examples.\n 
                                        5. The length of the summary should inherit from Example1;\n
                                        """
                
                Genetic_prompt += f'Output:'
                

            else:
                raise NotImplementedError

            if attempt == 0 and len(prompt_lst) == 0:
                print(f"Prompt Input: {prompt_input}")

            prompt_lst.append([{"role": "user", "content": prompt_input+in_context_input+Genetic_prompt}])
            
            
            if len(prompt_lst) == args.batch_size: 
                args.attributes = shuffle_attributes(args.attributes)
                try:
                    attempt += 1
                    return_msg = call_api_async(prompt_lst, model, args.temperature, args.max_tokens)                    
                    assert len(return_msg) == args.batch_size 
                    valid = 0
                    tmp = []
                    for msg in return_msg:
                        if "I apologize"  in msg or  "sorry"  in msg or  "Sorry" in msg or  "an AI language model" in msg or "I cannot perform" in msg:
                            continue
                        else:
                            valid += 1
                            example = {"_id": i, "text": msg}
                            #example.update(attr) #add randomly selected attrs to dic
                            #examples.append( example)
                            tmp.append(example)

                    manager.add_samples([x['text'] for x in tmp])

                    sent_cnt += valid 
                    prompt_lst = []

                    
                    print(f"CLass {i}: {class_name}, Attempt: {attempt}, Sent cnt: {sent_cnt}. ")
                    prefix = f"{args.dataset}/{class_name}/train_p{args.top_p}_{i}_{attempt}.jsonl"
                    
                    os.makedirs(f"{args.output_dir}/{args.dataset}/{class_name}", exist_ok= True)
                    f = open(f"{args.output_dir}/{prefix}", 'w')
                    for e in tmp:
                        f.write(json.dumps(e) + "\n")
                    f.close()

                except openai.error.RateLimitError:
                    print("Rate Limit Error! Attempt:", attempt)
                    prompt_lst = []
                    attr_lst = []
                    time.sleep(5)
                    continue
                except  openai.error.APIError:
                    print("API Error! Attempt:", attempt)
                    prompt_lst = []
                    attr_lst = []
                    time.sleep(5)
                    continue
                except openai.error.APIConnectionError:
                    print("APIConnectionError", attempt)
                    prompt_lst = []
                    attr_lst = []
                    time.sleep(5)
                    continue 
                except openai.error.InvalidRequestError:
                    print("InvalidRequestError! Invalid Request:", attempt)
                    prompt_lst = []
                    attr_lst = []
                    continue
                except openai.error.Timeout:
                    print("Timeout Error! Invalid Request:", attempt)
                    prompt_lst = []
                    attr_lst = []
                    continue
            if  sent_cnt >= args.n_sample or attempt > 1000:
                break
       

if __name__ == '__main__':
    main(args)

