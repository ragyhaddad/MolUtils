#!/bin/bash 
# Author: Ragy Haddad
import sys,os 
from rdkit import Chem  
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys 
import numpy as np 
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
import math

class Descriptor:
    def __init__(self):
        self.descs = ['HeavyAtomMolWt','NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge', 
        'MaxAbsPartialCharge', 'MinAbsPartialCharge','BertzCT',
        'NumAliphaticCarbocycles', 
        'NumAliphaticHeterocycles', 'NumAliphaticRings','NumAromaticCarbocycles', 
        'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 
        'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 
        'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP','MolMR']
    # Encode Molecule to RDKit ECFP - Default Radius 2 
    def get_morgan_fp(self,smiles,bit=1024,radius=2,string=True):  
        try:
            m = Chem.MolFromSmiles(smiles)
        except:
            return
        if string == False:
            morgan = AllChem.GetMorganFingerprintAsBitVect(m,radius,nBits=bit)
            return morgan
        morgan = AllChem.GetMorganFingerprintAsBitVect(m,radius,nBits=bit).ToBitString()
        bit_array = np.array([int(c) for c in morgan],dtype=np.float32)
        return bit_array  

    # Concatenated 2D fingerprint + Descriptors 
    def get_morgan_fp_with_descs(self,smiles,bit=2048,radius=2,string=True):
        # 2D Fingerprint
        try:
            m = Chem.MolFromSmiles(smiles)
        except:
            return
        if string == False:
            morgan = AllChem.GetMorganFingerprintAsBitVect(m,radius,nBits=bit)
            return morgan
        morgan = AllChem.GetMorganFingerprintAsBitVect(m,radius,nBits=bit).ToBitString()
        bit_array = [int(c) for c in morgan]
        # Descriptors  
        desc = self.get_descs(smiles,return_desc=False)
        
        full_fingerprint = bit_array + desc 
        full_fingerprint = np.array(full_fingerprint)
        return full_fingerprint

    def get_descs(self,smi,return_desc=False):
        descs = self.descs
        mol = Chem.MolFromSmiles(smi)
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(descs)
        c = list(calc.CalcDescriptors(mol))
        for idx in range(len(c)):
            if np.isinf(c[idx]) or math.isnan(c[idx]):
                c[idx] = 0 
        
        return c 
    


