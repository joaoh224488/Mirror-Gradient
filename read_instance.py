import numpy as np
import scipy.sparse as sparse


def read_instance(path:str):
    file = open(path, 'r')
    out = file.read()
    _, c, A, b =out.split(':')
    c = np.array([float(txt) for txt in c.split('\n')[1].split(' ') if txt != ' ' and txt != '\n' and txt != ''])
    text_A = A


    row = []
    col = []
    data = []

    i = 0
    
    for line in text_A.split('\n'):
        
        if line != '' and line[0] != 'V':
            j = 0

            for txt in line.split(' '):
                if txt != ' ' and txt != '\n' and txt != '':
                    if float(txt) > 1e-10:
                        data.append(float(txt))
                        row.append(i)
                        col.append(j)
                    j += 1
            
            i += 1
    
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    print(row)
    print(col)
    print(data)
    print(len(data))
            
    A = sparse.csr_matrix((data, (row, col)), shape= (i, j))
            


  
    b = np.array([float(txt) for txt in b.split(' ') if txt != ' ' and txt != '\n' and txt != ''])

    return c, A, b