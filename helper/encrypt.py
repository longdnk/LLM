def encrypt_password(s: str):
    def imul(a: int, b: int) -> int:
        return (a * b) & 0xFFFFFFFF
    seed = 1
    h1 = 0xdeadbeef ^ seed
    h2 = 0x41c6ce57 ^ seed
    for ch in s:
        ch_code = ord(ch)
        h1 = imul(h1 ^ ch_code, 2654435761)
        h2 = imul(h2 ^ ch_code, 1597334677)

    h1 = imul(h1 ^ (h1 >> 16), 2246822507)
    h1 ^= imul(h2 ^ (h2 >> 13), 3266489909)
    h2 = imul(h2 ^ (h2 >> 16), 2246822507)
    h2 ^= imul(h1 ^ (h1 >> 13), 3266489909)

    item = 4294967296 * (2097151 & h2) + (h1 >> 1)
    item = item if item >= 0 else -item

    return str(item) * 6