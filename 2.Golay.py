import numpy as np

Myword = 'DOGCAT'
debug = True
if debug: print('My word to start with is: ', Myword)
#step 1
def letter_to_ascii(letter):
    ascii_code = ord(letter)
    return '{0:b}'.format(ascii_code).zfill(12)\

#step 2
import numpy as np

H = np.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
])

#print(H)

G = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]
              ])

#print(G)


result = np.matmul(G, H.T)
mod_result = np.mod(result, 2)
print('Check zero matrix in step 2: ')
print(mod_result)


# step 4
def word_to_ascii_codes(word):
    ascii_codes = [letter_to_ascii(letter) for letter in word.upper()]
    return ascii_codes

ascii_codes = word_to_ascii_codes(Myword)

def encode(bits, G):
    v = np.array([list(b) for b in bits], dtype=int)
    v = np.matmul(v, G)
    return np.mod(v, 2)
if debug: print('Convert to 12 bits asc binary codes:', ascii_codes)
v = encode(ascii_codes, G)
if debug:
    print('Encoded codes are 24 bits in v: ')
    print(v)


def create_encoding_map():
    allLetters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    allLetterCodes= word_to_ascii_codes(allLetters)
    V = encode(allLetterCodes, G)

    # Create a dictionary to store the encoding map
    encoding_map = {}

    # Loop through each row of V and its corresponding letter code
    for i, v in enumerate(V):
        # Use the letter code as the key and the row of V as the value in the encoding map
        encoding_map[tuple(v)] = allLetters[i]

    return encoding_map
encoding_map = create_encoding_map()
if debug:
    print('A map of V and letters:')
    print(encoding_map)

def lookup_encoding_map(codes, map):
    # Lookup the encoding map and create the string
    string = ''
    for row in codes:
        string += map[tuple(row)]

    return string


import random

def modify_bits(v, error_num, seed=None):
    if seed is not None:
        random.seed(seed)
    w = []
    error_locations = []
    for code in v:
        # Convert the ASCII code to a list of bits
        bits = [int(b) for b in list(code)]

        # Determine the number of bits to flip randomly
        num_bits_to_flip = error_num

        # Choose random bit positions to flip
        positions_to_flip = random.sample(range(len(bits)), num_bits_to_flip)

        # Flip the chosen bits
        for pos in positions_to_flip:
            bits[pos] = 1 - bits[pos]

        # Store the modified code in the output list
        w.append(bits)

        # Store the error bit positions in the output array
        error_locations.append(positions_to_flip)

    return np.array(w, dtype=int), np.array(error_locations, dtype=int)

# below proved modify_bits works for
# for i in range(3):
#     w, locations = modify_bits(v, i+1)
#     print('*************')
#     print(w, locations)

w, locations = modify_bits(v, 3)
if debug:
    print('*************show modified w, locations, locations are modified bits with position randomly selected based on number of errors***********')
    print('w: ', w)
    print(locations)

A = np.array([[0,1,1,1,1,1,1,1,1,1,1,1],
              [1,1,1,0,1,1,1,0,0,0,1,0],
              [1,1,0,1,1,1,0,0,0,1,0,1],
              [1,0,1,1,1,0,0,0,1,0,1,1],
              [1,1,1,1,0,0,0,1,0,1,1,0],
              [1,1,1,0,0,0,1,0,1,1,0,1],
              [1,1,0,0,0,1,0,1,1,0,1,1],
              [1,0,0,0,1,0,1,1,0,1,1,1],
              [1,0,0,1,0,1,1,0,1,1,1,0],
              [1,0,1,0,1,1,0,1,1,1,0,0],
              [1,1,0,1,1,0,1,1,1,0,0,0],
              [1,0,1,1,0,1,1,1,0,0,0,1]])


def decoder(error_codes):
    # Step 1: Calculate S(w)
    S = np.matmul(error_codes, H.T) %2

    # Step 2: no need, covered by coding in next step
    # Step 3: Assume S(w) != 0
    v_list = []
    for s_row, w_row in zip(S, w):
        w1, w2 = np.split(w_row, 2)

        # Option 1: weight(S(w)) <= 3
        if np.count_nonzero(s_row) <= 3:
            tv = np.concatenate((w1, w2 + s_row)) % 2
            v_list.append(tv)
            continue

        # Option 2: weight(S(w) + e_jA) <= 2
        for j in range(12):
            e_j = np.zeros(12)
            e_j[j] = 1
            e_jA = np.matmul(e_j, A)
            if np.count_nonzero((s_row + e_jA)%2) <= 2:
                tv = np.concatenate((w1 + e_j, w2 + s_row + e_jA)) % 2
                v_list.append(tv)
                break

        # Option 3: weight(S(w)A) <= 3
        if np.count_nonzero(np.matmul(s_row, A)%2) <= 3:
            tv = np.concatenate((w1 + np.matmul(s_row, A), w2)) % 2
            v_list.append(tv)
            continue

        # Option 4: weight(S(w)A + e_jA) <= 2
        for j in range(12):
            e_j = np.zeros(12)
            e_j[j] = 1
            e_jA = np.matmul(e_j, A)
            if np.count_nonzero((np.matmul(s_row, A) + e_jA)%2) <= 2:
                tv = np.concatenate((w1 + np.matmul(s_row, A) + e_jA, w2 + e_j)) % 2
                v_list.append(tv)
                break

    # Combine all the v's into a single array
    tv = v_list

    # If none of the options apply, w is uncorrectable
    if len(tv) == 0:
        print("Error: unable to correct word w")

    return tv



v = decoder(w)
print('v recovered will be: ')
print(v)

print('Letters recovered will be: ')
results = lookup_encoding_map(v, encoding_map)
print(results)

