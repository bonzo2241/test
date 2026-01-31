import unittest

from app import QUESTIONS


class TestQuestions(unittest.TestCase):
    def test_questions_have_answers(self):
        self.assertGreater(len(QUESTIONS), 0)
        for item in QUESTIONS:
            self.assertIn(item["answer"], item["options"])


if __name__ == "__main__":
    unittest.main()
