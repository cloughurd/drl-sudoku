def print_str_puzzle(p):
    for i in range(0, len(p), 9):
        if i % 27 == 0:
            print('|-------|-------|-------|')
        chunk = p[i:i+9]
        line = f'| {chunk[0]} {chunk[1]} {chunk[2]} |'
        line += f' {chunk[3]} {chunk[4]} {chunk[5]} |'
        line += f' {chunk[6]} {chunk[7]} {chunk[8]} |'
        print(line)            
    print('|-------|-------|-------|')
    
def print_tensor_puzzle(p):
    import torch
    print('|-------|-------|-------|')
    for i in range(9):
        row = '| '
        for j in range(9):
            row += str(torch.argmax(p[i][j]).item()) + ' '
            if (j+1) % 3 == 0:
                row += '| '
        print(row)
        if (i+1) % 3 == 0:
            print('|-------|-------|-------|')
            