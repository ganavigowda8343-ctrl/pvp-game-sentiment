"""
Dataset Synthesis and Generator Module
Recreates the PvP Gaming Chat Sentiment Benchmark based on:
"Trolls, Haters, or Teammates: Classifying Sentiment in the Language of PvP Game Communications"
Stanford CS229 (Spring 2021)

Distribution:
- Class 0 (Neutral/Positive): ~43%
- Class 1 (Negative Attitude): ~31%
- Class 2 (Derogatory / Toxicity): ~26%
"""

import os
import random
import pandas as pd
import numpy as np

# Seed for full reproducibility
random.seed(42)
np.random.seed(42)

CLASS_0_TEMPLATES = [
    "ggwp", "gg", "well played team", "good game all", "gj on the gank",
    "nice ult mid!", "wp bot lane", "thanks for the heal", "ty bro", "thx support",
    "no problem, we got this", "good try everyone, next one", "nice shot!",
    "warding dragon right now", "defending mid tower", "let's group up and push",
    "careful, they are missing", "omw to help", "glhf everyone!", "good luck have fun",
    "b now, buy items", "nice defense!", "great teamfight guys", "commending you after match",
    "super clean combo", "thanks for carrying that fight", "we can still win this, stall late",
    "gg was a close match", "wp enemy mid, very close", "respect for not giving up",
    "nice smite steal!", "heading to baron, ping if needed", "good vision control team",
    "ty for the leash", "our synergy is pretty good", "good job holding top lane",
    "let's contest elder dragon", "good rotation bot", "nice flash play", "gg to both teams",
    "fun game honestly", "thanks for playing with me", "solid teamwork today", "good call on the retreat",
    "happy to play with you guys", "nice peel adc", "safe farming, play safe", "we scale better late"
]

CLASS_1_TEMPLATES = [
    "report for unskilled player is useless", "such a feeder...xD", "totally noob team",
    "go play barbie like a girl", "uninstall the game please", "mid diff honestly",
    "jg diff is insane this game", "our adc has no hands", "why are you always feeding?",
    "ff 15 open mid", "waste of 30 minutes with this team", "worst mid I have ever seen",
    "are you playing with your feet?", "stop inting into their tower", "ez game ez tutorial",
    "selling items, this is unwinnable", "you are totally useless", "bot lane gap is huge",
    "why pick that champion if you cant play?", "stop taking my farm you troll",
    "afk in fountain, good luck 4v5", "another boosted account on my team", "zero map awareness",
    "clown fiesta gameplay", "my team is heavy as rocks", "brain diff", "zero damage dealt",
    "how did you reach this rank?", "unbelievably bad players", "stop talking and play",
    "you threw the entire game", "report this feeder please x9", "literally griefing the match",
    "enjoy the loss feeder", "never play ranked again", "delete game and go back to bots",
    "walking sack of gold for enemy", "blind support no wards at all", "nice feed bro keep it up",
    "our jungler is sleeping in bushes", "just surrender already", "gg team diff go next",
    "npc teammate", "room temp iq play", "waste of oxygen in game", "stop pinging me noob"
]

CLASS_2_TEMPLATES = [
    "shut up fag", "kys idiot", "kill yourself feeder", "hope you get cancer and die",
    "fucking retard uninstall", "stfu bitch", "subhuman trash player", "you are a pathetic piece of shit",
    "drink bleach and die", "hang yourself after this match", "fucking dog go die",
    "faggot ass team", "retarded ape ruining games", "fucking useless retard",
    "die in a fire you cunt", "slit your wrists feeder", "brain dead motherfucker",
    "hope your family gets cancer", "stupid piece of garbage kys", "disgusting animal stop breathing",
    "fucking loser kys already", "eat shit and die bitch", "you deserve to get murdered",
    "waste of human life go end it", "fucking subhuman moron", "stfu you worthless faggot",
    "retarded clown hope you suffer", "piece of shit dog go bark", "fucking die already trash"
]

def generate_dataset(num_samples: int = 4000, output_path: str = "data/pvp_sentiment_dataset.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Target proportions: 43% Class 0, 31% Class 1, 26% Class 2
    n_class_0 = int(num_samples * 0.43)
    n_class_1 = int(num_samples * 0.31)
    n_class_2 = num_samples - n_class_0 - n_class_1
    
    data = []
    
    # Class 0: Neutral / Positive
    for _ in range(n_class_0):
        base = random.choice(CLASS_0_TEMPLATES)
        # Add realistic gaming punctuation/slang variation
        mods = ["", " :)", " <3", "!!", " lol", " wp", " ty", " gg"]
        msg = base + random.choice(mods)
        data.append({"message": msg.strip(), "label": 0, "sentiment": "neutral/positive"})
        
    # Class 1: Negative Attitude
    for _ in range(n_class_1):
        base = random.choice(CLASS_1_TEMPLATES)
        mods = ["", "...", " ???", " smh", " xD", " bro", " trash", " fr"]
        msg = base + random.choice(mods)
        data.append({"message": msg.strip(), "label": 1, "sentiment": "negative attitude"})
        
    # Class 2: Derogatory / Toxic
    for _ in range(n_class_2):
        base = random.choice(CLASS_2_TEMPLATES)
        mods = ["", "!!!", " stfu", " gtfo", " now", " fr"]
        msg = base + random.choice(mods)
        data.append({"message": msg.strip(), "label": 2, "sentiment": "derogatory"})
        
    df = pd.DataFrame(data)
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Generated dataset with {len(df)} samples saved to: {output_path}")
    print("Class Distribution:")
    print(df["sentiment"].value_counts(normalize=True))
    return df

if __name__ == "__main__":
    generate_dataset()
