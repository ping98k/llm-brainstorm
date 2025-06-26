import sys, os, types
from unittest.mock import patch, MagicMock

# Ensure project root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Provide dummy litellm module so import succeeds
fake_litellm = types.ModuleType('litellm')
fake_litellm.completion = MagicMock()
sys.modules.setdefault('litellm', fake_litellm)

import tournament_utils as tu


def make_response(contents):
    class Message:
        def __init__(self, content):
            self.content = content
    class Choice:
        def __init__(self, content):
            self.message = Message(content)
    return MagicMock(choices=[Choice(c) for c in contents])


def test_generate_players():
    resp = make_response([" player1 ", "player2\n"])
    with patch('tournament_utils.completion', return_value=resp) as mock_comp:
        players = tu.generate_players('instr', 2, model='m')
        mock_comp.assert_called_once_with(model='m', messages=[{'role': 'user', 'content': 'instr'}], n=2)
        assert players == ['player1', 'player2']


def test_prompt_score():
    resp = make_response([" {\"score\": [5]} "])
    with patch('tournament_utils.completion', return_value=resp) as mock_comp:
        result = tu.prompt_score('instr', ['c1'], 'block', 'pl', model='m')
        mock_comp.assert_called_once()
        assert result == '{"score": [5]}'


def test_prompt_play():
    resp = make_response([" {\"winner\": \"A\"} "])
    with patch('tournament_utils.completion', return_value=resp) as mock_comp:
        result = tu.prompt_play('instr', 'block', 'A text', 'B text', model='m')
        mock_comp.assert_called_once()
        assert result == '{"winner": "A"}'
