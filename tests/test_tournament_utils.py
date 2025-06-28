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
        players = tu.generate_players('instr', 2, model='m', api_base='b', api_key='k', temperature=0.5)
        mock_comp.assert_called_once_with(model='m', messages=[{'role': 'user', 'content': 'instr'}], n=2, api_base='b', api_key='k', temperature=0.5, chat_template_kwargs={'enable_thinking': False})
        assert players == ['player1', 'player2']


def test_prompt_score():
    resp = make_response(["Final verdict: [5]"])
    with patch('tournament_utils.completion', return_value=resp) as mock_comp:
        result = tu.prompt_score('instr', ['c1'], 'block', 'pl', model='m', api_base='b', api_key='k', temperature=0.2, include_instruction=False)
        mock_comp.assert_called_once()
        assert mock_comp.call_args.kwargs['api_base'] == 'b'
        assert mock_comp.call_args.kwargs['api_key'] == 'k'
        assert mock_comp.call_args.kwargs['temperature'] == 0.2
        assert result == 'Final verdict: [5]'


def test_prompt_pairwise():
    resp = make_response(["Final verdict: A"])
    with patch('tournament_utils.completion', return_value=resp) as mock_comp:
        result = tu.prompt_pairwise('instr', 'block', 'A text', 'B text', model='m', api_base='b', api_key='k', temperature=0.3, include_instruction=False)
        mock_comp.assert_called_once()
        assert mock_comp.call_args.kwargs['api_base'] == 'b'
        assert mock_comp.call_args.kwargs['api_key'] == 'k'
        assert mock_comp.call_args.kwargs['temperature'] == 0.3
        assert result == 'Final verdict: A'


def test_thinking_passed_to_completion():
    resp = make_response(["ok"])
    with patch('tournament_utils.completion', return_value=resp) as mock_comp:
        tu.generate_players('i', 1, thinking=True)
        tu.prompt_score('i', ['c'], 'block', 'p', thinking=True)
        tu.prompt_pairwise('i', 'block', 'a', 'b', thinking=True)
        assert mock_comp.call_count == 3
        for call in mock_comp.call_args_list:
            assert call.kwargs['chat_template_kwargs'] == {'enable_thinking': True}


def test_thinking_disabled_by_default():
    resp = make_response(["ok"])
    with patch('tournament_utils.completion', return_value=resp) as mock_comp:
        tu.generate_players('i', 1)
        tu.prompt_score('i', ['c'], 'block', 'p')
        tu.prompt_pairwise('i', 'block', 'a', 'b')
        assert mock_comp.call_count == 3
        for call in mock_comp.call_args_list:
            assert call.kwargs['chat_template_kwargs'] == {'enable_thinking': False}
