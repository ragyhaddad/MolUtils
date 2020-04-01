#!/bin/bash/python 
# Author: Ragy Haddad <ragy@deepbiometherapeutics.com>
import sys,os
import pandas as pd 
import numpy as np
from .tokenizer import Tokenizer 
from .descriptor import Descriptor 
from rdkit import Chem 

# Dataset functionality for custom batch loading of smiles
class Dataset:
    def __init__(self,input_table,sep='\t',smiles_column='smiles'):
        self.df = pd.read_csv(input_table,sep=sep)
        self.t = Tokenizer() 
        self.d = Descriptor() 
        self.remove_malformed()
        self.smiles_column = smiles_column
        try:
            self.smiles = self.df[smiles_column]
        except:
            print('-- [Error] smiles_column either does not exist or is incorrect')
            exit(1)
        
        # Sanitize Mols to Canonical Smiles if Any are Malformed
        # self.cans = [Chem.MolToSmiles(Chem.MolFromSmiles(smi),True) for smi in self.smiles]
        # self.smiles = pd.DataFrame(self.cans).iloc[:,0] 
    def remove_malformed(self):
        def is_correct(smiles):
            try:
                m = Chem.MolFromSmiles(smiles) 
                return True 
            except:
                return False 
        def is_fp(smiles):
            try:
                fp = self.d.get_morgan_fp(smiles)
                return True 
            except:
                return False
        self.df = self.df[self.df['smiles'].apply(lambda x: is_correct(x))]
        self.df = self.df[self.df['smiles'].apply(lambda x: is_fp(x))]
        self.df.reset_index()


    def format_batch_fp(self,input_df):
        smiles = [self.d.get_morgan_fp_with_descs(smi) for smi in input_df]
        return smiles
    # Format Batch 
    def format_batch(self,input_df,return_labels=False,shifted_targets=True,padding=False,label_encoding=False,return_length=True):
        data = None 
        if shifted_targets and label_encoding == False:
            src = [self.t.smiles_to_2d(r,padding=padding,add_start_token=True) for r in input_df] 
            trg = [self.t.smiles_to_2d(r,padding=padding,add_start_token=False) for r in input_df] 
        if shifted_targets and label_encoding:
            src = [self.t.smiles_to_int(r,padding=padding,add_start_token=True) for r in input_df]
            trg = [self.t.smiles_to_int(r,padding=padding,add_start_token=False) for r in input_df] 
        if return_labels:
            labels = [self.t.smiles_to_int(r,padding=padding,add_start_token=False) for r in input_df] # Labels do not have starting token  
            src,trg,labels = np.array(src),np.array(trg),np.array(labels)
            data = (src,trg,labels) 
        else:
            src,trg = np.array(src),np.array(trg)
            data = (src,trg) 
        if return_length:
            l_src = np.array([len(x) + 2 for x in input_df]) # Add 2 to lengths for sos_token and end_token
            l_trg = np.array([len(x) + 1 for x in input_df]) # Add 1 to lengths for eos_token  
            data = data + (l_src,l_trg)
        return data 
    # Load Batches
    def load_batches(self,batch_size,epoch_shuffle=True,padding=False,return_labels=False,label_encoding=False,shifted_targets=True,return_length=False):
        if epoch_shuffle:
            self.smiles = self.smiles.sample(frac=1)
        i = 0 
        while i < len(self.df.index.values):
            smiles = self.smiles.iloc[i : i + batch_size]
            data = self.format_batch(smiles,padding=padding,return_labels=return_labels,label_encoding=label_encoding,return_length=return_length)  
            i += batch_size 
            yield data 
    
    def load_batches_fp(self,batch_size=4,descs=True,epoch_shuffle=True,return_smiles=False):
        if epoch_shuffle:
            self.df = self.df.sample(frac=1)
        i = 0 
        while i < len(self.df.index.values):
            smiles_str = self.df.iloc[i : i + batch_size]['smiles'].values
            targets = self.df.iloc[i : i + batch_size]['activity'].values
            smiles = self.format_batch_fp(smiles_str) 
            if return_smiles: 
                data = (smiles,targets,smiles_str)
            else:
                data = (smiles,targets)
            i += batch_size 
            yield data 

    def num_smiles(self):
        return len(self.smiles) 
    
    def dataframe_to_fingerprints(self,fingerprint='mogran_descs'):
        x = [self.d.get_morgan_fp_with_descs(x,bit=1024,radius=2) for x in self.df[self.smiles_column]]
        y = self.df['activity'].values 
        x = np.array(x)
        y = np.array(y)
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        return x,y

    





        
    
