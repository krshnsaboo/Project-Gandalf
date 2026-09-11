import unittest
from response_parser import ResponseParser


class TestResponseParser(unittest.TestCase):

    def test_parse_standard_format(self):
        sample = (
            "Lecture: Rotate Matrix/Image by 90 Degrees | Brute - Optimal\n"
            "Timestamp: 01:06\n"
            "Watch: https://www.youtube.com/watch?v=Z0R2u6gd3GU&t=66s"
        )
        results = ResponseParser.parse(sample)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Rotate Matrix/Image by 90 Degrees | Brute - Optimal")
        self.assertEqual(results[0]["timestamp"], "01:06")
        self.assertEqual(results[0]["url"], "https://www.youtube.com/watch?v=Z0R2u6gd3GU&t=66s")

    def test_parse_markdown_bold_format(self):
        sample = (
            "**Lecture:** 4 Sum | Brute - Better - Optimal with Codes\n"
            "**Timestamp:** 14:20\n"
            "**Watch:** https://www.youtube.com/watch?v=eD95WRBhx1n&t=860s"
        )
        results = ResponseParser.parse(sample)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "4 Sum | Brute - Better - Optimal with Codes")
        self.assertEqual(results[0]["timestamp"], "14:20")
        self.assertEqual(results[0]["url"], "https://www.youtube.com/watch?v=eD95WRBhx1n&t=860s")

    def test_parse_multiple_recommendations(self):
        sample = (
            "Lecture: Dijkstra's Algorithm Part 1\n"
            "Timestamp: 05:10\n"
            "Watch: https://www.youtube.com/watch?v=V6H1qAeB-l4&t=310s\n\n"
            "Lecture: Dijkstra's Algorithm Part 2\n"
            "Timestamp: 12:45\n"
            "Watch: https://www.youtube.com/watch?v=3dINs369CEU&t=765s"
        )
        results = ResponseParser.parse(sample)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["timestamp"], "05:10")
        self.assertEqual(results[1]["timestamp"], "12:45")

    def test_empty_or_whitespace_input(self):
        self.assertEqual(ResponseParser.parse(""), [])
        self.assertEqual(ResponseParser.parse("   \n\n  "), [])


if __name__ == "__main__":
    unittest.main()
