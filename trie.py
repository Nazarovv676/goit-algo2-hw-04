"""
Base Trie class for Task 2.
This is a minimal implementation that supports put() and basic trie operations.
"""


class TrieNode:
    """Node in the trie structure."""
    
    def __init__(self):
        self.children: dict[str, 'TrieNode'] = {}
        self.is_end: bool = False
        self.value = None


class Trie:
    """Base trie class for storing words."""
    
    def __init__(self):
        self.root = TrieNode()
        self._word_count = 0
    
    def put(self, word: str, value=None) -> None:
        """
        Insert a word into the trie.
        
        Args:
            word: The word to insert
            value: Optional value to store with the word
        """
        if not isinstance(word, str):
            raise TypeError(f"word must be a string, got {type(word).__name__}")
        
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        if not node.is_end:
            self._word_count += 1
        node.is_end = True
        node.value = value
    
    def _find_node(self, prefix: str):
        """
        Find the node corresponding to a prefix.
        Returns None if prefix doesn't exist.
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    def contains(self, word: str) -> bool:
        """Check if a word exists in the trie."""
        node = self._find_node(word)
        return node is not None and node.is_end
