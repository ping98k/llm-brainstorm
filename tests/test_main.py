import sys, os, types, json
from unittest.mock import patch, MagicMock

# Ensure project root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Provide dummy litellm module so import succeeds
fake_litellm = types.ModuleType('litellm')
fake_litellm.completion = MagicMock()
sys.modules.setdefault('litellm', fake_litellm)

# Provide dummy dotenv module
fake_dotenv = types.ModuleType('dotenv')
fake_dotenv.load_dotenv = MagicMock()
sys.modules.setdefault('dotenv', fake_dotenv)

# Dummy gradio module so import succeeds
fake_gradio = types.ModuleType('gradio')
fake_gradio.Interface = MagicMock(return_value=MagicMock(launch=MagicMock()))
fake_gradio.Textbox = MagicMock
fake_gradio.Number = MagicMock
fake_gradio.Checkbox = MagicMock
fake_gradio.Plot = MagicMock
sys.modules.setdefault('gradio', fake_gradio)

# Dummy tqdm module for write method
class FakeTqdmModule(types.ModuleType):
    def __init__(self):
        super().__init__('tqdm')
        self.write = MagicMock()
    def __call__(self, iterable=None, total=None):
        return iterable

fake_tqdm_mod = FakeTqdmModule()
fake_tqdm_mod.tqdm = fake_tqdm_mod
sys.modules.setdefault('tqdm', fake_tqdm_mod)

# Dummy matplotlib module
fake_plt = types.ModuleType('matplotlib.pyplot')
fake_plt.figure = MagicMock(return_value='fig')
fake_plt.hist = MagicMock()
fake_matplotlib = types.ModuleType('matplotlib')
fake_matplotlib.pyplot = fake_plt
sys.modules.setdefault('matplotlib', fake_matplotlib)
sys.modules.setdefault('matplotlib.pyplot', fake_plt)

import main

class DummyFuture:
    def __init__(self, func, *args):
        self._func = func
        self._args = args
    def result(self):
        return self._func(*self._args)

class DummyExecutor:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        pass
    def submit(self, func, *args):
        return DummyFuture(func, *args)
    def map(self, func, iterable):
        for item in iterable:
            yield func(item)

class DummyTqdm:
    def __call__(self, iterable=None, total=None):
        return iterable
    def write(self, msg):
        pass

def test_run_tournament_full_loop():
    dummy_tqdm = DummyTqdm()
    with patch('main.generate_players') as mock_gen, \
         patch('main.prompt_score') as mock_score, \
         patch('main.prompt_pairwise') as mock_pair, \
         patch('main.ThreadPoolExecutor', return_value=DummyExecutor()) as MockExec, \
         patch('main.as_completed', new=lambda futs: futs), \
         patch('main.tqdm', new=dummy_tqdm), \
         patch('main.plt.figure', return_value='fig'), \
         patch('main.plt.hist'):
        mock_gen.return_value = (['p1', 'p2', 'p3', 'p4'], {'prompt_tokens':1,'completion_tokens':1})
        scores = {'p1':3, 'p2':2, 'p3':1, 'p4':0}
        mock_score.side_effect = lambda instr, cl, block, player, **kw: (json.dumps({'score': scores[player]}), {'prompt_tokens':1,'completion_tokens':1})
        mock_pair.side_effect = lambda instr, block, a, b, **kw: (json.dumps({'winner': 'A'}), {'prompt_tokens':1,'completion_tokens':1})

        results = list(main.run_tournament(
            api_base='b',
            api_token='k',
            generate_model='gm',
            score_model='sm',
            pairwise_model='pm',
            generate_temperature=1,
            score_temperature=1,
            pairwise_temperature=1,
            instruction_input='instr',
            criteria_input='c1,c2',
            n_gen=4,
            pool_size=2,
            num_top_picks=1,
            max_workers=1,
            enable_score_filter=True,
            enable_pairwise_filter=True,
            score_with_instruction=True,
            pairwise_with_instruction=True,
            generate_thinking=True,
            score_thinking=True,
            pairwise_thinking=True,
        ))

    process_log, hist_fig, top_picks, usage = results[-1]
    assert 'Done' in process_log
    assert hist_fig == 'fig'
    assert any(p in top_picks for p in {'p1', 'p2'})
    mock_gen.assert_called_once_with('instr', 4, model='gm', api_base='b', api_key='k', temperature=1, thinking=True, return_usage=True)
    assert 'Score completion' in process_log
    assert 'Pairwise completion' in process_log
    assert 'Prompt tokens' in usage
    assert mock_score.call_count == 4
    assert mock_pair.called


def test_run_tournament_pairwise_odd_players():
    dummy_tqdm = DummyTqdm()
    with patch('main.generate_players') as mock_gen, \
         patch('main.prompt_pairwise') as mock_pair, \
         patch('main.ThreadPoolExecutor', return_value=DummyExecutor()) as MockEx, \
         patch('main.as_completed', new=lambda futs: futs), \
         patch('main.tqdm', new=dummy_tqdm), \
         patch('main.plt.figure', return_value='fig'), \
         patch('main.plt.hist'):
        mock_gen.return_value = (['p1', 'p2', 'p3'], {'prompt_tokens':1,'completion_tokens':1})
        mock_pair.side_effect = lambda instr, block, a, b, **kw: (json.dumps({'winner':'A'}), {'prompt_tokens':1,'completion_tokens':1})

        results = list(main.run_tournament(
            api_base='b',
            api_token='k',
            generate_model='gm',
            score_model='sm',
            pairwise_model='pm',
            generate_temperature=1,
            score_temperature=1,
            pairwise_temperature=1,
            instruction_input='instr',
            criteria_input='c1,c2',
            n_gen=3,
            pool_size=3,
            num_top_picks=1,
            max_workers=1,
            enable_score_filter=False,
            enable_pairwise_filter=True,
            score_with_instruction=True,
            pairwise_with_instruction=True,
            generate_thinking=True,
            score_thinking=True,
            pairwise_thinking=True,
        ))

    process_log, fig, top_picks, usage = results[-1]
    assert 'Done' in process_log
    assert any(p in top_picks for p in {'p1', 'p2', 'p3'})
    assert mock_pair.call_count == 3
