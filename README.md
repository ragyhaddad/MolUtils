# MolUtils
#### All purpose deep learning + cheminformatics Lib

#### Example Usage
``` python
    import sys
    from MolUtils.tokenizer import Tokenizer
    from MolUtils.dataloader import Dataset 
    from MolUtils.descriptor import Descriptor 

    
    t = Tokenizer() 
    d = Descriptor() 

    ## Fingerprinting + Descriptors 
    morgan_fp = d.get_morgan_fp('CCC',bit=1024,radius=2)
    morgan_fp_with_descs = d.get_morgan_fp_with_descs('CCC',bit=1024,radius=2)
    descs = d.get_descs('CC') 

    # Smiles to 2D
    smi = t.smiles_to_2d('CCC',padding=False,add_end_token=False) 

    # Smiles to 1D Int 
    smi = t.smiles_to_int('CCC',padding=False,add_end_token=False)

    # Get Char to Int Dictionary
    char_to_int = t.char_to_int 
    
    # Get Int to Char Dictionary
    int_to_char = t.int_to_char 

    # Setting a new char to int 
    t.set_char_to_int({'C':0,'H':1,'O':2})
    t.set_int_to_char({0:'C',1:"H",2:"O"}) 

    # Get charset and int to char from a dataframe  
    char_to_int,int_to_char = t.extract_charset(sys.argv[1],smiles_column='smiles',sep='\t') 

    # Loading Batches with MolUtils Dataset
    ds = Dataset(sys.argv[1],smiles_column='smiles',sep='\t')

    # Load Batch using the Dataset Class
    for idx,data in enumerate(ds.load_batches(batch_size=2,padding=True)):
        pass
    exit()
 ```
