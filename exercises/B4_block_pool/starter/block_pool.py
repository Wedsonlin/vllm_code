class BlockPoolExhausted(Exception): pass
class BlockPool:
    def __init__(self, num_blocks: int):
        raise NotImplementedError
    def allocate(self): raise NotImplementedError
    def retain(self, block_id): raise NotImplementedError
    def free(self, block_id): raise NotImplementedError
