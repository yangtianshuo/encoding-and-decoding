import numpy as np

Myword = 'CATDOG'
debug = True
if debug: print('My word to start with is: ', Myword)
#step 1
def letter_to_ascii(letter):
    ascii_code = ord(letter)
    return '{0:b}'.format(ascii_code).zfill(11)\

#step 2
H = np.array([
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
])

#print(H)

G = np.array([
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
])

#print(G)


# verify step 3
result = np.matmul(G, H.T)
mod_result = np.mod(result, 2)
#print(mod_result)


# step 4
def word_to_ascii_codes(word):
    ascii_codes = [letter_to_ascii(letter) for letter in word.upper()]
    return ascii_codes

ascii_codes = word_to_ascii_codes(Myword)

def encode(bits, G):
    v = np.array([list(b) for b in bits], dtype=int)
    v = np.matmul(v, G)
    return np.mod(v, 2)
if debug: print('Convert to 11 bits asc binary codes:', ascii_codes)
v = encode(ascii_codes, G)
if debug:
    print('Encoded codes are 15 bits in v: ')
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

def modify_bits(v, error_num):
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

w, locations = modify_bits(v, 1)
if debug:
    print('*************show modified w, locations, locations are modified bits with position randomly selected based on number of errors***********')
    print(w)
    print(locations)


t = np.mod(w@H.T, 2)
#print(t)

def compute_row_sums(arr):
    # Compute the powers of 2 for each column index
    powers_of_2 = 2 ** np.arange(arr.shape[1])

    # Compute the dot product of each row with the powers of 2
    row_sums = np.sum(arr * powers_of_2, axis=1)
    row_sums[row_sums > 0] -= 1

    return row_sums

modifiedIndex = compute_row_sums(t)
if debug: print('Compute and Verify the positions modified: ', modifiedIndex)

def flip_back(bits, indices):
    for i, idx in enumerate(indices):
        bits[i][idx] = 1 - bits[i][idx]
    return bits

bits = flip_back(w, modifiedIndex)
if debug:
    print('Flip back computed bits locations: ')
    print(bits)

print('Letters recovered will be: ')
results = lookup_encoding_map(bits, encoding_map)
print(results)




# # Convert each letter to its ASCII code and format it as a string of 3 digits
# ascii_codes = [f'{letter_to_ascii(letter)}' for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ']
# print(ascii_codes)
# # Join the codes into a single string with spaces between them
# ascii_string = ' '.join(ascii_codes)
# # Print the result
# print(ascii_string)
# print(letter_to_ascii('A')) # Output: 00001000001   (65)
# print(letter_to_ascii('Z')) # Output: 00001011010   (90)
