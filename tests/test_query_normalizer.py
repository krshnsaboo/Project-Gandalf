import unittest
from query_normalizer import normalize_query, enhance_query


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

    def test_leetcode_acronyms(self):
        url = "https://leetcode.com/problems/lru-cache/"
        self.assertEqual(normalize_query(url), "LRU Cache")

        bst_url = "https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/"
        self.assertEqual(normalize_query(bst_url), "Lowest Common Ancestor Of A Binary Search Tree")

    def test_leetcode_roman_numerals(self):
        url = "https://leetcode.com/problems/course-schedule-ii/"
        self.assertEqual(normalize_query(url), "Course Schedule II")

    def test_enhance_query_expansions(self):
        self.assertIn("Priority Queue", enhance_query("dijkstra with pq"))
        self.assertIn("Dynamic Programming", enhance_query("frog jump dp"))
        self.assertIn("LRU Cache", enhance_query("https://leetcode.com/problems/lru-cache/"))

    def test_enhance_query_question_framing(self):
        enhanced = enhance_query("4Sum Optimal Solution")
        self.assertTrue(enhanced.startswith("Which lecture and timestamp covers"))


if __name__ == "__main__":
    unittest.main()
