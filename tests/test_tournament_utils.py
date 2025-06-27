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
    resp1 = make_response([" player1 "])
    resp2 = make_response(["player2\n"])
    with patch('tournament_utils.completion', side_effect=[resp1, resp2]) as mock_comp:
        players = tu.generate_players('instr', 2, model='m', api_base='b', api_key='k')
        assert mock_comp.call_count == 2
        for call in mock_comp.call_args_list:
            assert call.kwargs == {'model': 'm', 'messages': [{'role': 'user', 'content': 'instr'}], 'n': 1, 'api_base': 'b', 'api_key': 'k'}
        assert players == ['player1', 'player2']


def test_generate_players_drops_failed():
    with patch('tournament_utils.completion', side_effect=Exception('boom')) as m:
        players = tu.generate_players('instr', 1, model='m')
        assert players == []
        assert m.call_count == 5


def test_prompt_score():
    resp = make_response([" {\"score\": [5]} "])
    with patch('tournament_utils.completion', return_value=resp) as mock_comp:
        result = tu.prompt_score('instr', ['c1'], 'block', 'pl', model='m', api_base='b', api_key='k')
        mock_comp.assert_called_once()
        assert mock_comp.call_args.kwargs['api_base'] == 'b'
        assert mock_comp.call_args.kwargs['api_key'] == 'k'
        assert result == '{"score": [5]}'


def test_prompt_pairwise():
    resp = make_response([" {\"winner\": \"A\"} "])
    with patch('tournament_utils.completion', return_value=resp) as mock_comp:
        result = tu.prompt_pairwise('instr', 'block', 'A text', 'B text', model='m', api_base='b', api_key='k')
        mock_comp.assert_called_once()
        assert mock_comp.call_args.kwargs['api_base'] == 'b'
        assert mock_comp.call_args.kwargs['api_key'] == 'k'
        assert result == '{"winner": "A"}'
