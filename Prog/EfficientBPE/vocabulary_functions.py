'''
Vocabulary entry dictionary structure

For almost all entries:
key = token text
value = {
        "frequency": Number of occurences of this merge, 
        "order": The order in which this merge is executed,
        "pair": A string tuple of the left and right token of this merge,
        "parent": Parent token string (only defined if mutated),
        "similarity": Similarity score (only defined if mutated)
}

For alphabet symbols:
key = symbol text
value = {
        "frequency": 0,
        "order": 0
}
'''
