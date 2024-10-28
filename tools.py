"""
Named entity types in Spacy

PERSON - People, including fictional
NORP - Nationalities or religious or political groups
FACILITY - Buildings, airports, highways, bridges, etc.
ORGANIZATION - Companies, agencies, institutions, etc.
GPE - Countries, cities, states
LOCATION - Non-GPE locations, mountain ranges, bodies of water
PRODUCT - Vehicles, weapons, foods, etc. (Not services)
EVENT - Named hurricanes, battles, wars, sports events, etc.
WORK OF ART - Titles of books, songs, etc.
LAW - Named documents made into laws 
LANGUAGE - Any named language 


"""

import pandas as pd
import numpy as np
from datetime import datetime as dt
from typing import Dict, List
import nltk

nltk.download("vader_lexicon")
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy as sp
import networkx as nx
from itertools import product
from typing import List, Tuple
import nltk
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors as plt_colours


def download_to_parquet(url: str = None, fname: str = None, delim: str = "\t"):
    df = pd.read_csv(url, delimiter=delim)
    df.to_parquet(f"data/{fname}.parquet")




def entity_graph(
    in_df: pd.DataFrame = None,
    text_field: str = "summary",
    nlp_model: str = "en_core_web_sm",
    keep_types: List = ["PERSON"],
    top_prop: float = 0.5,
)-> nx.Graph:
    """
    Construct entity graph of named entities by sequentially running NER on free text field
    By default only PERSON entities
    Edges weighted by proximity of entities to each
    other in article text

    Inputs
        in_df - dataframe with free text to do NER on
        text_field - colun in in_df with free text data
        nlp_model - model to load with Spacy.load()
        keep_types: list of Spacy entity types to accept (see top)
        top_prop: keep this proportion of most commonly occuring node-pairs
    Outputs:
        g -  entity graph
    """
    nlp = sp.load(nlp_model)
    for idx, row in tqdm(in_df.iterrows(), total=in_df.shape[0]):
        text = row[text_field]
        if text:
            text_df = entity_df(text, nlp=nlp, keep_types=keep_types)
            if idx == 0:
                g_df = text_df
            else:
                g_df = pd.concat([g_df, text_df])
    g_df["source"] = g_df["source"].apply(ner_cleanup)
    g_df["target"] = g_df["target"].apply(ner_cleanup)
    g_df = g_df.dropna(subset=["source", "target"])
    g_df = g_df[g_df["source"] != g_df["target"]]
    g_df = g_df.value_counts(normalize=True).reset_index()
    keep_rows = int(round(top_prop * g_df.shape[0]))
    g_df = g_df.head(keep_rows)
    g_df["weight_log"] = np.log(g_df["proportion"])
    weight_range = [g_df["weight_log"].min(), g_df["weight_log"].max()]
    g_df["weight"] = g_df["weight_log"].apply(lambda x: renormalise(x, weight_range))
    g = nx.from_pandas_edgelist(g_df, edge_attr=True)
    return g


def ner_cleanup(text:str = None) -> str:
    """Basic cleanup on NEs - remove problem characters, possessives, initials etc"""
    text = text.replace(":", "").upper()
    text = text.replace(".", "")
    text = text.replace("’", "")
    if text.endswith("'S"):
        text = text.replace("'S", "")
    text_arr = text.split()
    if len(text_arr) < 2:
        text = None
    else:
        text_ls = [x for x in text_arr if len(x) > 1]
        text = " ".join(text_ls)
    return text


def entity_df(text: str = None, nlp: sp.Language = None, keep_types: List[str] = None) -> pd.DataFrame:
    """
    Construct df from named entities in text, with each row being an observed pair
    of NEs in the text

    Inputs
        text: text to mine for NEs
        nlp: Spacy NLP model to use
        keep_types: list of GDelt entity types to accept (see top)
    Outputs:
        df: columns = ["source", "target"]
    """
    doc = nlp(text)
    df = pd.DataFrame(columns=["source", "target"])
    for idx, (src, tgt) in enumerate(product(doc.ents, doc.ents)):
        source = src.text
        target = tgt.text
        if (
            (src.label_ in keep_types)
            and (tgt.label_ in keep_types)
            and (source != target)
            and (src.start_char < tgt.start_char)
            and (source[0].isalpha())
            and (target[0].isalpha())
            and (len(source) < 20)
            and (len(target) < 20)
        ):
            df.at[idx, "source"] = source
            df.at[idx, "target"] = target
    return df


def renormalise(
    n: float = None, range1: List = [10, 5000], range2: List = [10, 0.3]
) -> float:
    """
    Renormalise n from range1 to range2
    """
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]


def plot_single_graph(
    g,
    layout: str = "neato",
    color_attr: bool = False,
    figsize: Tuple[int] = (20, 20),
    save_path: str = None,
) -> None:
    """
    Plot g with nodes coloured by e_type attribute
    """

    fig, ax = plt.subplots(figsize=figsize)
    if layout == "kk":
        pos = nx.kamada_kawai_layout(g, weight="weight")
    elif layout == "spring":
        pos = nx.spring_layout(g, iterations=50, weight="weight")
    elif layout == "neato":
        pos = nx.nx_pydot.graphviz_layout(g, prog="neato")

    if color_attr:
        colors = [g.nodes[node]["color"] for node in list(g.nodes())]
    else:
        colors = "khaki"

    nx.draw(
        g,
        pos=pos,
        with_labels=True,
        ax=ax,
        node_color=colors,
        edge_color="gainsboro",
    )
    fig.set_facecolor("lightblue")
    if save_path:
        fig.savefig(save_path)
