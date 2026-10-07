from dataclasses import dataclass
from typing import List, Sequence
@dataclass(frozen=True)
class Request:
    req_id: str
    num_tokens: int
def admit(requests: Sequence[Request], budget: int) -> List[Request]:
    raise NotImplementedError
def remaining_budget(requests, budget):
    raise NotImplementedError
