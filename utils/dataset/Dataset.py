from typing import Callable, List

from torch import Tensor
from torch.utils.data import Dataset


class KfttDataset(Dataset):
    def __init__(
        self,
        path_to_corpus: str,
        max_len: int,
        text_to_id: Callable[[str, int], Tensor],
    ) -> None:
        self.data: List[str] = []
        with open(path_to_corpus, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(line)
        self.max_len = max_len
        self.text_to_id = text_to_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tensor:
        return self.text_to_id(self.data[index], self.max_len)
