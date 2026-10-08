"""
Data Preprocessing and Gamer Slang Normalization Module
Project: Trolls, Haters, or Teammates: Classifying Sentiment in the Language of PvP Game Communications
Authors: Ganavi Gowda (PES1UG24CS707), Aadarsh Reddy (PES1UG24CS706)
Institution: PES University
Reference: Stanford CS229 Machine Learning Project (Spring 2021)
"""

import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

# Gamer slang and contraction mappings
SLANG_MAP = {
    "gg": "good game",
    "ggwp": "good game well played",
    "wp": "well played",
    "glhf": "good luck have fun",
    "ty": "thank you",
    "thx": "thanks",
    "gj": "good job",
    "nt": "nice try",
    "mb": "my bad",
    "np": "no problem",
    "omw": "on my way",
    "brb": "be right back",
    "afk": "away from keyboard",
    "ff": "forfeit surrender",
    "diff": "difference gap",
    "jg": "jungler",
    "mid": "middle lane",
    "top": "top lane",
    "bot": "bottom lane",
    "int": "intentional feeding",
    "inting": "intentional feeding dying",
    "feeder": "bad player giving kills",
    "feeding": "repeatedly dying to enemies",
    "noob": "bad inexperienced player",
    "nub": "bad player",
    "ez": "easy game insult",
    "trash": "terrible useless player",
    "plz": "please",
    "pls": "please",
    "u": "you",
    "ur": "your",
    "r": "are",
    "b": "back retreat",
    "def": "defend",
    "push": "attack lane",
}

# The 165 toxic dictionary baseline phrases as modeled in CS229 Section 4.2
TOXIC_DICTIONARY_165 = [
    "noob", "feeder", "feeding", "trash", "useless", "uninstall", "ez", "easy",
    "report", "reported", "diff", "bot", "troll", "trolling", "inting", "afk",
    "worst", "garbage", "delete", "stupid", "idiot", "moron", "loser", "clown",
    "throw", "throwing", "dog", "ape", "monkey", "braindead", "wood", "bronze",
    "iron", "ff15", "open mid", "gg go next", "kys", "kill yourself", "die",
    "get cancer", "fuck", "fucking", "shit", "bitch", "retard", "retarded",
    "fag", "faggot", "autistic", "cancer", "stfu", "shut up", "uninstall game",
    "sell items", "run it down", "grief", "griefer", "griefing", "sell account",
    "boosted", "ebay account", "hardstuck", "peaked", "waste of oxygen", "scum",
    "subhuman", "kill ur self", "hang yourself", "drink bleach", "worthless",
    "pathetic", "blind", "deaf", "mute", "disable", "unskilled", "bad",
    "horrible", "terrible", "stop playing", "leave game", "quit game", "never play",
    "rep mid", "rep top", "rep bot", "rep jg", "x9", "nine x", "nine report",
    "brain issue", "hands missing", "no hands", "go play roblox", "go play barbie",
    "zero damage", "zero impact", "carried", "heavy", "anchored", "dead weight",
    "bot gameplay", "ai gameplay", "npc", "walking ward", "minion", "creep",
    "stop feeding", "why feed", "look at score", "check stats", "refund skin",
    "waste of space", "gross", "disgusting", "pig", "animal", "creature",
    "brainless", "smooth brain", "room temp iq", "low iq", "zero iq", "monkey mode",
    "clown fiesta", "circus", "joke", "fraud", "scammer", "impostor", "traitor",
    "sabotage", "int", "inters", "inter", "feeder team", "team diff", "player diff",
    "gap", "canyon", "chasm", "void", "abyss", "disaster", "catastrophe", "ganked",
    "trolled", "griefed", "ruined", "lost cause", "unwinnable", "hopeless", "doomed"
]

def clean_text(text: str, expand_slang: bool = False) -> str:
    """
    Cleans raw chat text:
    - Lowercase
    - Elongated character reduction (e.g., 'noooob' -> 'noob')
    - Optional slang expansion
    - Stripping irrelevant punctuation while retaining emotes
    """
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()
    
    # Compress elongated characters (3+ repetitions down to 2, e.g. 'sooooo' -> 'soo')
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # Slang replacement if requested
    if expand_slang:
        tokens = text.split()
        tokens = [SLANG_MAP.get(tok, tok) for tok in tokens]
        text = " ".join(tokens)

    # Remove URL links if any
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Clean non-standard symbols but keep basic alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z0-9\s!?:;()<>]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class PvPFeatureExtractor:
    """
    Feature extraction pipeline providing:
    1. TF-IDF vectorization (~2700 dimensions as in CS229 paper)
    2. 165-term toxic dictionary baseline encoding
    """
    def __init__(self, max_features: int = 2700, ngram_range: tuple = (1, 2)):
        self.tfidf = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=True,
            min_df=2
        )
        self.dict_165 = set(TOXIC_DICTIONARY_165)

    def fit_tfidf(self, texts):
        return self.tfidf.fit(texts)

    def transform_tfidf(self, texts):
        return self.tfidf.transform(texts)

    def transform_dict_baseline(self, texts):
        """
        Creates binary indicator vector based on the 165 toxic dictionary terms
        matching Section 4.2 baseline in CS229 report.
        """
        import numpy as np
        features = []
        dict_list = sorted(list(self.dict_165))
        for text in texts:
            cleaned = clean_text(text).split()
            tokens = set(cleaned)
            vec = [1 if term in tokens or term in text else 0 for term in dict_list]
            features.append(vec)
        return np.array(features, dtype=np.float32)
