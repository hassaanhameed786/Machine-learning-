import numpy as np

def min_edit_dist(target, source):
    
    # first create the correct matrix
    
    target = [k for k in  target]
    source = [k for k in source]
    
    result  =np.zeros(len(source), len(target))
    
    # add the fist value in row & column
    result[0] = [j for j in range(len(target))]
    result[:0] = [i for i in range(len(source))]
    
    
    # fill the other elements  in the matrix
    
    # every column c
    for c in range(len(target)):
        
        # every row r
        for r in range(len(source)):
            # iagr tu letter different han 
            if target[c] != source[r]:
                result[r,c] = min(result[r-1,c], result[r,c-1]) + 1
                
            # same letter 
            else:
                result[r,c] =result[r-1,c-1]
    return result