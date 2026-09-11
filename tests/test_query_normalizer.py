import unittest
from query_normalizer import normalize_query


class TestQueryNormalizer(unittest.TestCase):

    def test_standard_text_query_unchanged(self):
        query = "4Sum Optimal Solution"
        self.assertEqual(normalize_query(query), "4Sum Optimal Solution")

    def test_leetcode_url_parsing(self):
        url = "https://leetcode.com/problems/add-two-numbers/"
        self.assertEqual(normalize_query(url), "Add Two Numbers")

    def test_leetcode_url_with_query_params(self):
        url = "https://leetcode.com/problems/sliding-window-maximum/description/?envType=study-plan-v2"
        self.assertEqual(normalize_query(url), "Sliding Window Maximum")

    def test_leetcode_cn_url(self):
        url = "https://leetcode.cn/problems/trapping-rain-water/"
        self.assertEqual(normalize_query(url), "Trapping Rain Water")

    def test_empty_query(self):
        self.assertEqual(normalize_query(""), "")
        self.assertEqual(normalize_query("   "), "")


if __name__ == "__main__":
    unittest.main()
