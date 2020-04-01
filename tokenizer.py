#!/bin/bash/python 
# Author: Ragy Haddad <ragy@deepbiometherapeutics.com>
import sys,os
import pandas as pd 
import numpy as np 
from .config import Config 
import torch  

# Tokenizer class for encoding and decoding utils 
class Tokenizer:
    def __init__(self):
        self.config = Config() 
        self.char_to_int = self.config.char_to_int 
        self.int_to_char = self.config.int_to_char 
        self.max_len = self.config.max_len   
    # 2D Logits to String
    def encoding_to_string(self,decoder_output_batch):
        int_arr_ = []
        str_out = []
        for m in decoder_output_batch:
            matrix = m 
            max_idx = torch.argmax(matrix).item()
            int_arr_.append(max_idx) 
            str_out.append(self.int_to_char[str(max_idx)])  
        str_out = "".join(str_out)
        return str_out
    # Smiles to 2D 
    def smiles_to_2d(self,smiles_string,padding=False,add_start_token=False,add_end_token=True):
        def encode(smiles_string):
            arr_ = np.zeros((len(smiles_string),self.config.vocab_size),dtype=np.float32)
            for idx,c in enumerate(smiles_string):
                arr_[idx][self.char_to_int[c]] = 1 
            return arr_  
        if padding:
            if add_start_token:
                smiles_string = self.config.sos_token + smiles_string + (self.max_len - len(smiles_string) - 1) * self.config.eos_token
            else:
                smiles_string = smiles_string  + (self.max_len - len(smiles_string)) * self.config.eos_token 
        else:
            if add_start_token:
                smiles_string = self.config.sos_token + smiles_string + self.config.eos_token
            else:
                if add_end_token:
                    smiles_string = smiles_string  + self.config.eos_token 
                else:
                    smiles_string = smiles_string
        return encode(smiles_string) 
    # Smiles to 1D Int
    def smiles_to_int(self,smiles_string,padding=False,add_start_token=False,add_end_token=True):
        def encode(smiles_string):
            arr_ = np.zeros(len(smiles_string),dtype=np.int32) 
            for idx,c in enumerate(smiles_string):
                arr_[idx] = self.char_to_int[c] 
            return arr_ 
        if padding:
            if add_start_token:
                smiles_string = self.config.sos_token + smiles_string + (self.max_len - len(smiles_string) - 1) * self.config.eos_token
            else:
                smiles_string = smiles_string + (self.max_len - len(smiles_string) - 1) * self.config.eos_token  
        else:
            if add_start_token:
                smiles_string = self.config.sos_token + smiles_string + self.config.eos_token
            else:
                if add_end_token:
                    smiles_string = smiles_string + self.config.eos_token   
                else:
                    smiles_string = smiles_string
        return encode(smiles_string) 

    # 1D Int to Smiles
    def int_to_smiles(self,int_arr):
        return  ".".join([self.int_to_char[i] for i in int_arr]) 
    # Change char to int and int to char from default
    def set_char_to_int(self,char_to_int):
        self.char_to_int = char_to_int 
    def set_int_to_char(self,int_to_char):
        self.int_to_char = int_to_char 
    def set_eos_char(self,eos_char):
        self.config.eos_token = eos_token  
    def set_max_len(self,max_len):
        self.config.max_len = max_len 
        self.max_len = max_len  
    # Extract available charset from a dataset
    def extract_charset(self,input_df,save=True,sep='\t',smiles_column='smiles'):
        smiles_df = pd.read_csv(input_df,sep=sep)[smiles_column] # Get Smiles Column
        charset = set("".join(list(smiles_df))+"!E") # i = start , E = end
        char_to_int = dict((c,i) for i,c in enumerate(charset))
        int_to_char = dict((i,c) for i,c in enumerate(charset)) 
        return char_to_int,int_to_char


    









