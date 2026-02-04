from app import SEED_TASKS


def test_seed_questions_have_valid_answers():
    for task in SEED_TASKS:
        for question in task["questions"]:
            assert question["answer"] in question["options"]
