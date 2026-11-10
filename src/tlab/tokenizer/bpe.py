"""字节级 BPE: 训练， 编码， 解码"""


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    # 统计每一对相邻 token 出现的次数； 字典按第一次出现的顺序排列
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    # 从左往右扫描， 把每一处 pair 换成新编号 idx
    new_ids = []
    i = 0
    while i < len(ids):
        # 当前位置和下一个位置正好是 pair： 合并成一个， 跳过两个
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        # 否则原样保留， 前进一个
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPETokenizer:
    def __init__(self) -> None:
        # 合并表： (a, b) -> 新编号， 按学到的先后顺序排列
        self.merges: dict[tuple[int, int], int] = {}
        # 词表： 编号 -> 对应的字节串； 0 - 255 是单个字节
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> None:
        assert vocab_size >= 256, "词表至少包含256个字节"
        # 文本 -> UTF-8 字节 -> 整数列表
        ids = list(text.encode("utf-8"))
        for i in range(vocab_size - 256):
            stats = get_stats(ids)
            # 已经没有相邻对可以合并 (文本太短)
            if not stats:
                break
            # 出现次数最多的一对; 并列时 max 取字典里最先出现的
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            # 记录合并规则； 新 token 的字节 = 两个旧 token 的字节拼起来
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        # 反复挑“最早学到”的那条可用规则， 直到没有规则可用
        while len(ids) >= 2:
            stats = get_stats(ids)
            # 编号越小 = 越早学到； 不合并表里的对当作无穷大
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        # 编号 -> 字节串拼接 -> 按 UTF-8 解码； 残缺字节替换成 ？
        data = b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8", errors="replace")
