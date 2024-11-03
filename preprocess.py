import pandas as pd
import os
import argparse
from functools import partial

from mol2vec.features import mol2alt_sentence
from gensim.models import Word2Vec
from rdkit.Chem import MolFromSmiles as sm
import numpy as np

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

prose_path = os.path.join(os.getcwd(), "prose")
sys.path.append(prose_path)

from prose.models.multitask import ProSEMT # use the path of ProSE
from prose.alphabets import Uniprot21


def parse_args():
    parser = argparse.ArgumentParser(description='Process some data.')
    parser.add_argument('--raw_data_dir', type=str, default='ic-50-prediction/raw', help='Directory of the data')
    parser.add_argument('--save_processed_dir', type=str, default='ic-50-prediction/processed', help='Directory for the processed data')    
    parser.add_argument('--mol_model_path', type=str, default='pretrained/model_300dim.pkl', help='Path to the pretrained molecule embedder')
    parser.add_argument('--seq_model_path', type=str, default='pretrained/prose_mt_3x1024.sav', help='Path to the pretrained sequence embedder')
    
    return parser.parse_args()

def sentences2vec(sentences, model, unseen=None):
    """Generate vectors for each sentence (list) in a list of sentences. Vector is simply a
    sum of vectors for individual words.
    
    Parameters
    ----------
    sentences : list, array
        List with sentences
    model : word2vec.Word2Vec
        Gensim word2vec model
    unseen : None, str
        Keyword for unseen words. If None, those words are skipped.
        https://stats.stackexchange.com/questions/163005/how-to-set-the-dictionary-for-text-analysis-using-neural-networks/163032#163032

    Returns
    -------
    np.array
    """
    
    keys = set(model.wv.key_to_index)
    vec = []
    
    if unseen:
        unseen_vec = model.wv.get_vector(unseen)

    for sentence in tqdm(sentences, desc="Sentence->Embedding"):
        if unseen:
            vec.append(sum([model.wv.get_vector(y) if y in set(sentence) & keys
                       else unseen_vec for y in sentence]))
        else:
            vec.append(sum([model.wv.get_vector(y) for y in sentence 
                            if y in set(sentence) & keys]))
    return np.array(vec)

ma = partial(mol2alt_sentence, radius=1)
word2vec = None

def mol2embeds(mols: pd.Series, word2vec: Word2Vec) -> pd.DataFrame:
    tqdm.pandas(desc="SMILES->Molecule")
    unique_mols = mols.drop_duplicates()
    sent = unique_mols.progress_apply(sm)
    tqdm.pandas(desc="Molecule->Sentence")
    alt_sent = sent.dropna().progress_apply(ma)
    vec = sentences2vec(alt_sent, word2vec, unseen='UNK')
    vec_df = pd.Series(vec.tolist(), index=alt_sent.index).to_frame(name="mol_embeds")
    return vec_df

def seq2embeds(sequences: pd.Series, prosemt: ProSEMT) -> pd.DataFrame:

    alphabet = Uniprot21()
    vec = []

    unique_sequences = sequences.dropna().drop_duplicates()
    for sequence in tqdm(unique_sequences, desc="Sequence->Embedding"):
        x = sequence.encode().upper()
        x = alphabet.encode(x)
        x = torch.from_numpy(x)
        with torch.no_grad():
            x = x.long().unsqueeze(0)
            z = prosemt.transform(x)
            z = z.squeeze(0)
            z = z.sum(0)
            z = z.cpu().numpy()
        vec.append(z.tolist())
    
    vec_df = pd.Series(vec, index=unique_sequences.index).to_frame(name="seq_embeds")
    return vec_df

def process_data(data_dir: str, sub_dir: str, save_dir: str, word2vec: Word2Vec, prosemt: ProSEMT) -> None:
    save_sub_dir = os.path.join(save_dir, sub_dir)
    os.makedirs(save_sub_dir, exist_ok=True)
    data = pd.read_csv(os.path.join(data_dir, f'{sub_dir}.csv'))
    mol_embeds = mol2embeds(data['Ligand SMILES'], word2vec)

    data = data.iloc[mol_embeds.index]
    data[['Ligand SMILES']].join(mol_embeds).to_csv(os.path.join(save_sub_dir, "mol_embeds.csv"))

    seq_embeds = seq2embeds(data['BindingDB Target Chain Sequence'], prosemt)
    data[['BindingDB Target Chain Sequence']].join(seq_embeds).to_csv(os.path.join(save_sub_dir, "seq_embeds.csv"))

    data.to_csv(os.path.join(save_sub_dir, 'data.csv'))

def main():
    args = parse_args()
    word2vec = Word2Vec.load(args.mol_model_path)
    prosemt = ProSEMT.load_pretrained(args.seq_model_path)

    for sub_dir in ['train', 'val', 'test']:
        process_data(args.raw_data_dir, sub_dir, args.save_processed_dir, word2vec, prosemt)

if __name__ == "__main__":
    main()
