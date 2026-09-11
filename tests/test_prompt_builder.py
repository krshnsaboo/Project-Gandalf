import unittest
from prompt_builder import format_timestamp, PromptBuilder


class TestPromptBuilder(unittest.TestCase):

    def test_format_timestamp_minutes(self):
        self.assertEqual(format_timestamp(66), "01:06")
        self.assertEqual(format_timestamp(0), "00:00")
        self.assertEqual(format_timestamp(599), "09:59")

    def test_format_timestamp_hours(self):
        self.assertEqual(format_timestamp(3665), "01:01:05")

    def test_prompt_builder_structure(self):
        contexts = [
            {
                "lecture_title": "2 Sum Problem",
                "start": 392.1,
                "end": 458.0,
                "timestamp_url": "https://www.youtube.com/watch?v=UXDSeD9mN-k&t=392s",
                "text": "This is optimal approach using hashmap."
            }
        ]
        system_prompt, user_prompt = PromptBuilder.build("Two sum using hashmap", contexts)
        self.assertIn("navigation assistant", system_prompt)
        self.assertIn("2 Sum Problem", user_prompt)
        self.assertIn("06:32", user_prompt)
        self.assertIn("https://www.youtube.com/watch?v=UXDSeD9mN-k&t=392s", user_prompt)


if __name__ == "__main__":
    unittest.main()
