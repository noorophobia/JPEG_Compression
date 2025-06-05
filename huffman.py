def huffman_decode(encoded_data, huffman_dict):
    reverse_dict = {v: k for k, v in huffman_dict.items()}
    code = ''
    decoded_data = []
    for bit in encoded_data:
        code += bit
        if code in reverse_dict:
            decoded_data.append(reverse_dict[code])
            code = ''
    return decoded_data

def build_huffman_tree(data):
    freq = defaultdict(int)
    for symbol in data:
        freq[symbol] += 1
    heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    huff_tree = heap[0]
    return {symbol: code for symbol, code in huff_tree[1:]}

def huffman_encode(data, huffman_dict):
    return ''.join(huffman_dict[symbol] for symbol in data)
