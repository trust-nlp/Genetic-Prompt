import openai
import asyncio
from typing import List, Dict, Any
import argparse
import os 
from tqdm import trange, tqdm
import re 
import time
from utils import load_attributes
import numpy as np
import json 
from openai import AsyncOpenAI
import pandas as pd
import random


parser = argparse.ArgumentParser("")
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--temperature", default=1, type=float, help="which seed to use")
parser.add_argument("--top_p", default=0.95, type=float, help="which seed to use")
parser.add_argument("--n_sample", default=100, type=int, help="the number of examples generated for each class")
parser.add_argument("--batch_size", default=20, type=int, help="")

parser.add_argument("--dataset", default='agnews', type=str, help="which model to use")

parser.add_argument("--model_name", default='gpt-3.5-turbo', type=str, help="which model to use")
parser.add_argument("--model_type", default='attrprompt', type=str, help="which model type to use")
parser.add_argument("--max_tokens", default=1024, type=int, help="max output length")
parser.add_argument("--output_dir", default='.', type=str, help="the folder for saving the generated text")

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


    
if args.dataset in ['chemprot']:
    args.input_dir = ''

    args.entity = ['chemical', 'protein']
    args.label_names=['substrate','upregulator', 'downregulator', 'agonist', 'antagonist']
    args.attributes = ["length", "voice","sentence_structure" ,"interaction_verb","modifier","negation" ,"entity_proximity"]
    
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
        #"product_of": "the protein is a product of the reaction on this chemical",
        "substrate": "the chemical is acted upon or modified by the protein, typically an enzyme",
        #"not": "There's no relation between the chemical and the protein from the generated sentence."
    }
    args.domain = 'Protein Chemical Relation extraction'
    
    
elif args.dataset in ['ddi']:
    args.input_dir = ''

    args.entity = ['drug', 'drug']
    args.domain = 'Drug-Drug Interaction extraction'
    args.attributes = ["length", "voice","polarity","interaction_verb","modifier","drug_mentions","entity_proximity"]
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
 





elif args.dataset in ['semeval2010']:
    args.input_dir = ''
    args.attributes = ["length", "voice","Sentence_Structure","Readability","Entity_distance","domain"]
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

    args.attributes = ["length", "voice","Sentence_Structure","Readability","Entity_distance","domain"]
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


elif args.dataset in ['i2b2-2010']:
    args.label_names=["TrAP","TrCP","TrIP","TrNAP","TrWP"]
    args.attributes = ["length", "voice","polarity","interaction_verb","modifier","drug_mentions","entity_proximity"]
    args.label_def = {
        "TrAP": "Treatment Administered for Problem",
        "TrCP": "Treatment Causes Problem",
        "TrIP": "Treatment Improves Problem",
        "TrNAP": "Treatment is Not Administered Because of Problem",
        "TrWP": "Treatment Worsens Problem"}
    args.domain="Treatment-Problem Relation extraction"
    args.entity=["Treatment", "Problem"]
    
    
    
    
elif args.dataset in ['n2c2-2018']:
    args.attributes = ["length", "voice","polarity","interaction_verb","modifier","drug_mentions","entity_proximity"]
    args.label_names=["ADE-Drug","Dosage-Drug","Duration-Drug","Form-Drug","Frequency-Drug","Reason-Drug","Route-Drug","Strength-Drug"]
    args.label_def = {
        "ADE-Drug": "Adverse Drug Event",
        "Dosage-Drug": "Dosage of Drug",
        "Duration-Drug": "Duration of Drug",
        "Form-Drug": "Form of Drug",
        "Frequency-Drug": "Frequency of Drug",
        "Reason-Drug": "Reason for Drug",
        "Route-Drug": "Route of Drug",
        "Strength-Drug": "Strength of Drug"
    }
    args.entity=["Drug", "Attribute"]
    args.domain="Drug-Drug Attribute Relation extraction"

elif args.dataset in ['agnews']:
    args.input_dir = ''
    args.attributes = ["length", "location","style","subtopics"]
    args.label_names=[ "World", "Sports", "Business", "Sci_Tech"]
    
#     World
# Sports
# Business
# Sci_Tech
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
    
elif args.dataset in ["sst-2"]:
    args.input_dir = ''
    args.attributes = ["length", "genre","style","subtopics"]
    args.label_names=["negative","positive"]
    args.label_def = {
        "positive": "Positive sentiment",
        "negative": "Negative sentiment"
    }
    args.domain="Movie Review Sentiment Classification"


elif args.dataset in ["yelp"]:
    args.input_dir = ''
    args.attributes = ["length", "cuisine","style","subtopics"]
    args.label_names=["negative","positive"]
    args.label_def = {
        "positive": "Positive sentiment",
        "negative": "Negative sentiment"
    }
    args.domain="Restaurant Review Sentiment Classification"
    
elif args.dataset in ["wikiqa"]:
    args.input_dir = ''
    args.attributes = ["length", "depth","style","subtopics"]
    args.label_names=["0","1"]
    args.label_def = {
        "0": "wrong answer",
        "1": "correct answer"
    }
    args.domain="wikipedia question answering"
    
elif args.dataset in ["bioasq"]:
    pass
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

def gen_example(attr_dict):
    lengths = {}
    for x in attr_dict:
        lengths[x] = len(attr_dict[x])
    while True:
        return_dict = {}
        for z in lengths:
            
            idx_z = np.random.randint(low = 0, high = lengths[z], dtype = int)
            return_dict[z] = idx_z 
            # lst.append(return_dict)
      
        yield return_dict

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
    
    attr_dict = {}
    for attr_name in args.attributes:
        attr_dict[attr_name] = load_attributes(attr_name = attr_name, model = model, dataset = args.dataset, method = args.model_type, classes = label_names)

    
        
    if args.input_dir.endswith('.jsonl') or args.dataset in ['yelp', 'wikiqa','scitldr']:
        df = pd.read_json(args.input_dir, lines=True)
    elif args.input_dir.endswith('.json'):        
        df = pd.read_json(args.input_dir, lines=False) 
        
    for i, class_name in enumerate(label_names):

        
        if args.dataset == 'wikiqa':
            if i ==0:
                args.n_sample = args.n_sample 
                
            if i ==1:   
                args.n_sample = int(args.n_sample//4)       
        
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
        
        print(i, class_name)
        label_def = args.label_def[class_name]

        print(f"Prompt, Give a synthetic sample of {args.domain} about {re.sub('_', ' ', class_name)} following the requirements below")
        sent_cnt = 0
        attempt = 0
        attr_dict_cls = {}
        for attr in attr_dict:
            if attr in ['subtopics', 'similar', 'brands', 'product_name', 'product_name_filter', "experience", "resource", "scenario", "scenario_filter"]:
                attr_dict_cls[attr] = attr_dict[attr][class_name]
            else:
                attr_dict_cls[attr] = attr_dict[attr]
        prompt_lst = []
        attr_lst = []
        examples = []

        if "similar" in attr_dict:
            similar_label = ",".join(attr_dict["similar"][class_name])
            
            


        for return_dict in gen_example(attr_dict_cls):
            
            if len(texts) >= 2:
                sample1, sample2 = random.sample(texts, 2)
    
                in_context_input = f"Example 1: {sample1} \nExample 2: {sample2} \n"
            prompt_tmp = {x: attr_dict_cls[x][return_dict[x]] for x in return_dict}
            attr_lst.append(return_dict)
            if args.dataset ==  'chemprot':
                        # ["length", "voice","sentence_structure" ,"interaction_verb","modifier","negation" ,"entity_proximity"]
                prompt_input =f"You need to generate synthetic data for the {args.domain} task. \n"

        
                prompt_input += f"""
                                    Your task is to write a sentence about '{class_name}' relation between {args.entity[0]} and {args.entity[1]}. \n
                                    The two entities should be marked with XML-style tags as <{args.entity[0]}> {args.entity[0]} </{args.entity[0]}> and <{args.entity[1]}> {args.entity[1]} </{args.entity[1]}> respectively in your response.
                                    The sentence should follow the requirements below: \n 
                                        1. must discuss about the {args.entity[0]} and {args.entity[1]} with the relation {label_def}  \n
                                        2. should be {prompt_tmp['length']}. \n
                                        3. The sentence should be written in {prompt_tmp['voice']}. \n
                                        4. The sentence structure should be {prompt_tmp['sentence_structure']}. \n
                                        5. The sentence may include the interaction verb {prompt_tmp['interaction_verb']}. \
                                        6. The sentence may include a {prompt_tmp['modifier']} modifier . \n
                                        7. The sentence may include {prompt_tmp['negation']} \n
                                        8. The entity proximity is {prompt_tmp['entity_proximity']}. \n
                                        """
                prompt_input += in_context_input                       
                prompt_input += f'Relation: {class_name}\n'
                prompt_input += f'Text:'      

            elif args.dataset ==  'sst-2':
                prompt_input =f"You need to generate synthetic data for the {args.domain} task. \n"

                prompt_input += f"Write a {re.sub('_', ' ', class_name)} review for a movie, following the requirements below: \n \
                            1. the overall rating should be {re.sub('_', ' ', class_name)};\n \
                            2. the review should discuss about a {prompt_tmp['genre']} movie;  \n \
                            3. the review should focus on '{prompt_tmp['subtopics']}'; \n \
                            4. should be in length between {prompt_tmp['length']} words and {int(prompt_tmp['length']) + 50} words;\n \
                            5. the style of the review should be '{prompt_tmp['style']}'."
                prompt_input += in_context_input
                
                prompt_input += f'Sentiment: {class_name}\n'
                prompt_input += f'Text:'
                
            elif args.dataset ==  'yelp':
                prompt_input =f"You need to generate synthetic data for the {args.domain} task. \n"
                prompt_input += f"Write a {re.sub('_', ' ', class_name)} review for a restaurant, following the requirements below: \n \
                            1. the overall review should be {re.sub('_', ' ', class_name)}';\n \
                            2. should be a '{prompt_tmp['cuisine']}' restaurant';  \n \
                            3. should focus on '{prompt_tmp['subtopics']}'; \n \
                            4. should be in length between {prompt_tmp['length']} words and {int(prompt_tmp['length']) + 50} words;\n \
                            5. the style of the review should be '{prompt_tmp['style']}'."

                prompt_input += in_context_input
                
                prompt_input += f'Sentiment: {class_name}\n'
                prompt_input += f'Text:'


            elif args.dataset == 'agnews':
                prompt_input =f"You need to generate synthetic data for the {args.domain} task. \n"
                prompt_input += f"Give a synthetic sample of news on {re.sub('_', ' ', class_name)} following the requirements below: \n\
                            1. should focus on '{prompt_tmp['subtopics']}';\n \
                            2. should be in length between {prompt_tmp['length']} words and {int(prompt_tmp['length']) + 50} words;\n \
                            3. The writing style of the news should be '{prompt_tmp['style']}';\n \
                            4. The location of the news should be in {prompt_tmp['location']}. \n"
                prompt_input += in_context_input
                prompt_input += f'Category: {class_name}\n'
                prompt_input += f'Text:'
                
            elif args.dataset == 'stackexchange':
                prompt_input =f"You need to generate synthetic data for the {args.domain} task. \n"
                prompt_input += f"Give a synthetic sample of question post in {args.dataset} on {re.sub('_', ' ', class_name)} following the requirements below: \n\
                            1. should focus on the scenario of '{prompt_tmp['scenario']}';\n \
                            2. should be in length between {prompt_tmp['length']} words and {int(prompt_tmp['length']) + 50} words;\n \
                            3. The question should be in {prompt_tmp['depth']};\n \
                            4. The writing style of the question should be '{prompt_tmp['style']}'. \n"
                prompt_input += in_context_input
                prompt_input += f'Category: {class_name}\n'
                prompt_input += f'Text:'


            elif args.dataset ==  'ddi':
                prompt_input =f"You need to generate synthetic data for {args.domain} task. \n"
#    args.attributes = ["length", "voice","polarity","interaction_verb","modifier","drug_mentions","entity_proximity"]


                prompt_input+= f"""
                                    Your task is to write a sentence about '{class_name}' relation between {args.entity[0]} and {args.entity[1]}. \n 
                                    The two {args.entity[0]}s should be marked with XML-style tags as <{args.entity[0]}> {args.entity[0]} </{args.entity[0]}>.
                                    The sentence should follow the requirements below: \n 
                                        1. should discuss about the {args.entity[0]} and {args.entity[1]} with the relation {label_def}  \n 
                                        2. should be {prompt_tmp['length']}. \n
                                        3. The sentence should be written in {prompt_tmp['voice']}. \n
                                        4. The sentence may include the interaction verb {prompt_tmp['interaction_verb']}. \n
                                        5. The sentence may include a {prompt_tmp['modifier']} modifier. \n
                                        6. The entity proximity is {prompt_tmp['entity_proximity']}. \n
                                        7. The drug can be mentioned as {prompt_tmp['drug_mentions']}. \n
                                        8. The polarity of the sentence is {prompt_tmp['polarity']}. \n
                                        """
                prompt_input += in_context_input                            
                prompt_input += f'Relation: {class_name}\n'
                prompt_input += f'Text:'
                
            elif args.dataset ==  'i2b2-2010':
                prompt_input =f"You need to generate synthetic data for {args.domain} task. \n"

    #args.attributes = ["length", "voice","polarity","interaction_verb","modifier","drug_mentions","entity_proximity"]

                prompt_input += f"""
                                    Your task is to write a sentence about '{class_name}' relation between {args.entity[0]} and {args.entity[1]}. \n 
                                    The two entities should be marked with XML-style tags as <{args.entity[0]}> {args.entity[0]} </{args.entity[0]}> and <{args.entity[1]}> {args.entity[1]} </{args.entity[1]}> respectively in your response.
                                    The sentence should follow the requirements below: \n 
                                        1. should discuss about the {args.entity[0]} and {args.entity[1]} with the relation {label_def}  \n 
                                        2. should be {prompt_tmp['length']}. \n
                                        3. The sentence should be written in {prompt_tmp['voice']}. \n
                                        4. The sentence may include the interaction verb {prompt_tmp['interaction_verb']}. \n
                                        5. The sentence may include a {prompt_tmp['modifier']} modifier. \n
                                        6. The entity proximity is {prompt_tmp['entity_proximity']}. \n
                                        7. The drug can be mentioned as {prompt_tmp['drug_mentions']}. \n
                                        8. The polarity of the sentence is {prompt_tmp['polarity']}. \n
                                        """
                prompt_input += f'Relation: {class_name}\n'
                prompt_input += f'Text:'
            elif args.dataset ==  'n2c2-2018':
                
                prompt_input =f"You need to generate synthetic data for {args.domain} task. \n"
    #args.attributes = ["length", "voice","polarity","interaction_verb","modifier","drug_mentions","entity_proximity"]

                prompt_input+= f"""
                                    Your task is to write a sentence about '{class_name}' relation between {args.entity[0]} and {args.entity[1]}. \n 
                                    The {args.entity[0]} and {args.entity[1]} should be marked with XML-style tags as <{args.entity[0]}> {args.entity[0]} </{args.entity[0]}>. \n
                                    The '{class_name}' relation means the {args.entity[1]} is the {label_def}\n
                                    The sentence should follow the requirements below: \n 
                                    1. should be {prompt_tmp['length']}. \n
                                    2. The sentence should be written in {prompt_tmp['voice']}. \n
                                    3. The sentence may include the interaction verb {prompt_tmp['interaction_verb']}. \n
                                    4. The sentence may include a {prompt_tmp['modifier']} modifier. \n
                                    5. The entity proximity is {prompt_tmp['entity_proximity']}. \n
                                    6. The drug can be mentioned as {prompt_tmp['drug_mentions']}. \n
                                    7. The polarity of the sentence is {prompt_tmp['polarity']}. \n
                                        """
                prompt_input+= f'Relation: {class_name}\n'
                prompt_input+= f'Text:'
                
            elif args.dataset ==  'scierc':
                
                
                prompt_input =f"You need to generate synthetic data for {args.domain} task. \n"

    #args.attributes = ["length", "voice","Sentence_Structure","Readability","Entity_distance","research_domain"]

                prompt_input += f"""
                                    Your task is to write a sentence about '{class_name}' relation between {args.entity[0]} and {args.entity[1]}. \n 
                                    The '{class_name}' could be one of the relations in [{label_def}]\n
                                    The two entities should be marked with XML-style tags as <{args.entity[0]}> {args.entity[0]} </{args.entity[0]}> and <{args.entity[1]}> {args.entity[1]} </{args.entity[1]}> respectively in your response.
                                    The sentence should follow the requirements below: \n
                                    1. should be {prompt_tmp['length']}. \n
                                    2. The sentence should be written in {prompt_tmp['voice']}. \n
                                    3. The sentence structure should be {prompt_tmp['Sentence_Structure']}. \n
                                    4. The readability of the sentence should be {prompt_tmp['Readability']}. \n
                                    5. The distance between the two entities should be {prompt_tmp['Entity_distance']}. \n
                                    6. The research domain of the sentence should be {prompt_tmp['research_domain']}. \n
                                        """
                prompt_input += f'Relation: {class_name}\n'
                prompt_input += f'Text:'                
                
                
                
                
            elif args.dataset ==  'semeval2010':
                   #args.attributes = ["length", "voice","Sentence_Structure","Readability","Entity_distance","domain"]
 
                prompt_input =f"You need to generate synthetic data for {args.domain} task. \n"

                prompt_input += f"""
                                    Your task is to write a sentence about '{class_name}' relation between 2 entities. \n 
                                    The '{class_name}' relation means {label_def}\n
                                    The two entities should be marked with XML-style tags as <{args.entity[0]}> {args.entity[0]} </{args.entity[0]}>.
                                    The sentence should follow the requirements below: \n 
                                    1. should be {prompt_tmp['length']}. \n
                                    2. The sentence should be written in {prompt_tmp['voice']}. \n
                                    3. The sentence structure should be {prompt_tmp['Sentence_Structure']}. \n
                                    4. The readability of the sentence should be {prompt_tmp['Readability']}. \n
                                    5. The distance between the two entities should be {prompt_tmp['Entity_distance']}. \n
                                    6. The domain of the sentence should be {prompt_tmp['domain']}. \n
                                        """
                prompt_input += in_context_input    
                prompt_input += f'Relation: {class_name}\n'
                prompt_input += f'Text:'                    
            elif args.dataset ==  'conll04':
                prompt_input =f"You need to generate synthetic data for {args.domain} task. \n"

    #args.attributes = ["length", "voice","Sentence_Structure","Readability","Entity_distance","domain"]

                prompt_input += f"""
                                    Your task is to write a sentence about '{class_name}' relation between 2 entities. \n 
                                    The '{class_name}' relation means {label_def}\n
                                    The two entities should be marked with XML-style tags as <{args.entity[0]}> {args.entity[0]} </{args.entity[0]}>.
                                    The sentence should follow the requirements below: \n 
                                    1. should be {prompt_tmp['length']}. \n
                                    2. The sentence should be written in {prompt_tmp['voice']}. \n
                                    3. The sentence structure should be {prompt_tmp['Sentence_Structure']}. \n
                                    4. The readability of the sentence should be {prompt_tmp['Readability']}. \n
                                    5. The distance between the two entities should be {prompt_tmp['Entity_distance']}. \n
                                    6. The domain of the sentence should be {prompt_tmp['domain']}. \n
                                        """
                prompt_input += in_context_input    
                prompt_input+= f'Relation: {class_name}\n'
                prompt_input += f'Text:'    

            elif args.dataset == "wikiqa":
                #["length", "depth","style","subtopics"]
                prompt_input = f"You need to generate synthetic data for {args.domain} task. \n"
                
                prompt_input += f"""
                                    Your task is to write a question-answer pair . \n
                                    The question-answer pair should follow the requirements below: \n
                                        1. The answer must be a {label_def} to the question.\n
                                        2. should be {prompt_tmp['length']};\n
                                        3. The style of the pair should be {prompt_tmp['style']};\n
                                        4. The depth of the pair should be {prompt_tmp['depth']};\n
                                        5. The subtopics of the pair should be {prompt_tmp['subtopics']}.\n
                                        """
                prompt_input += in_context_input   
                prompt_input += f'Output:'

          
            elif args.dataset == "bioasq" :
                pass
            elif args.dataset == "scitldr" :
                prompt_input = f"You need to generate synthetic data for {args.domain} task. \n"
                prompt_input+= f"""
                                    Your task is to write computer science paper abstract and it's summary. \n
                                    The abstract and summary should follow the requirements below: \n
                                    1. The length of the abstract should be {prompt_tmp['length']};\n
                                    2. The style of the abstract should be {prompt_tmp['style']};\n
                                    3. The depth of the abstract should be {prompt_tmp['depth']};\n
                                    4. The subtopics of the abstract should be {prompt_tmp['subtopics']}.\n
                                    """    
                prompt_input += in_context_input 
                prompt_input+= f'Output:'
                
            elif args.dataset == "mediqa" :
                prompt_input = f"You need to generate synthetic data for {args.domain} task. \n"
  
                
                prompt_input+= f"""
                                    Your task is to write a medical question and it's summary. \n
                                    The question and summary should follow the requirements below: \n 
                                    1. The length of the question should be {prompt_tmp['length']};\n
                                    2. The style of the question should be {prompt_tmp['style']};\n
                                    3. The depth of the question should be {prompt_tmp['depth']};\n
                                    4. The subtopics of the question should be {prompt_tmp['subtopics']}.\n
                                        """
                prompt_input += in_context_input 
                prompt_input += f'Output:'
                
            else:
                raise NotImplementedError

            if attempt == 0 and len(prompt_lst) == 0:
                print(f"Prompt Input: {prompt_input}")

            prompt_lst.append(
                    [{"role": "user", "content": prompt_input}]
            )
            if len(prompt_lst) == args.batch_size:
                try:
                    attempt += 1
                    # return_msg = [
                    #     client.chat.completions.create(
                    #         model=model,
                    #         messages=prompt,
                    #         temperature=args.temperature,
                    #         max_tokens=args.max_tokens,
                    #         top_p=args.top_p,
                    #     ).choices[0].message.content
                    #     for prompt in prompt_lst
                    # ]
                    
                    return_msg = call_api_async(prompt_lst, model, args.temperature, args.max_tokens)        
                    
                    
                                
                    assert len(return_msg) == len(attr_lst)
                    valid = 0
                    tmp = []
                    for (msg, attr) in zip(return_msg, attr_lst):
                        if "I apologize"  in msg or  "sorry"  in msg or  "Sorry" in msg or  "an AI language model" in msg or "I cannot perform" in msg or "To generate" in msg:
                            continue
                        else:
                            valid += 1
                            example = {"_id": i, "text": msg}
                            example.update(attr)
                            examples.append( example)
                            tmp.append(example)
                    sent_cnt += valid 
                    prompt_lst = []
                    attr_lst = []
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
                    time.sleep(15)
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
            if sent_cnt >= args.n_sample or attempt > 200:
                break
       

if __name__ == '__main__':
    main(args)

