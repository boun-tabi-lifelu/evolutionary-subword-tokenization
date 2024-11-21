# Symbol node
class Sym:
    def __init__(self, symbol_str: str):
        self.literal = symbol_str
        self.prev = None
        self.next = None
    def __len__(self):
        return len(self.literal)
    def __str__(self):
        return self.literal
    def __repr__(self):
        return self.literal

class SymPair:
    def __init__(self, sym1: str, sym2: str):
        self.left = sym1
        self.right = sym2
        self.count = 0
        self.positions = []
    def add_pos(self, pos: tuple[Sym, Sym]):
        self.positions.append(pos)
        self.count += 1
    def __hash__(self):
        return hash((self.left, self.right))
    def __eq__(self, other):
        return (self.left, self.right) == (other.left, other.right)
    def __lt__(self, other):
        return self.count < other.count
    def __str__(self):
        return f"Pair: ({self.left}, {self.right}), Count: {self.count}"
    def __repr__(self):
        return f"Pair: ({self.left}, {self.right}), Count: {self.count}"


# Doubly Linked List for Sym
class SymList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0
    def append(self, value: Sym):
        if self.head is None:
            self.head = value
            self.tail = value
        else:
            self.tail.next = value
            temp = self.tail
            self.tail = value
            self.tail.prev = temp
        self.length += 1
    def __len__(self):
        return len(self.length)
    def __str__(self):
        res = ""
        cur = self.head
        while cur is not None:
            res += cur.literal
            cur = cur.next
        return res
    def __iter__(self):
        cur = self.head
        while cur is not None:
            yield cur
            cur = cur.next

# Heap map implementation designed to work specifically with SymPairs
class MaxHeapMap:
    def __init__(self):
        self.heap = []
        self.map = {}

    def _swap(self, i: int, j: int):
        self.map[self.heap[i]] = j
        self.map[self.heap[j]] = i
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def push(self, value: SymPair):
        # Add the new value at the end
        self.heap.append(value)
        self.map[value] = len(self.heap) - 1
        # Heapify up to maintain the max heap property
        self._heapify_up(len(self.heap) - 1)

    def pop(self):
        if not self.heap:
            raise IndexError("pop from an empty heap")
        if len(self.heap) == 1:
            del self.map[self.heap[0]]
            return self.heap.pop()
        
        self._swap(0, len(self.heap) - 1)
        max_value = self.heap.pop()
        del self.map[max_value]
        # Heapify down to maintain the max heap property
        self._heapify_down(0)
        return max_value

    def peek(self):
        if not self.heap:
            raise IndexError("peek from an empty heap")
        return self.heap[0]

    def remove_by_value(self, value: SymPair):
        if value not in self.map:
            raise ValueError("value is not in the heap")
        heap_idx = self.map[value]
        self._swap(heap_idx, len(self.heap) - 1)
        internal_val = self.heap.pop()
        del self.map[value]
        if heap_idx < len(self.heap):
            self._heapify_down(heap_idx)
            self._heapify_up(heap_idx)
        return internal_val

    def _heapify_up(self, index: int):
        parent = (index - 1) // 2
        # Keep swapping until the max heap property is restored
        # print(f"Comparing {str(self.heap[index])} , idx: {index} > {str(self.heap[parent])} , idx: {parent} : {self.heap[index] > self.heap[parent]}")
        while index > 0 and self.heap[index] > self.heap[parent]:
            self._swap(index, parent)
            index = parent
            parent = (index - 1) // 2

    def _heapify_down(self, index: int):
        largest = index
        left = 2 * index + 1
        right = 2 * index + 2

        # Check if the left child exists and is greater than the current node
        if left < len(self.heap) and self.heap[left] > self.heap[largest]:
            largest = left
        # Check if the right child exists and is greater than the current largest
        if right < len(self.heap) and self.heap[right] > self.heap[largest]:
            largest = right

        # Swap and continue heapifying if the largest is not the current node
        if largest != index:
            self._swap(index, largest)
            self._heapify_down(largest)

    def __str__(self):
        return str(self.heap)

