import numpy as np
from ncmcm.cognitive_graphs.helpers import shift_pos_by, generate_equidistant_colors, map_names, make_integer_list


def test_shift_pos_by():
    old_pos = {'a': (1.0, 0.0), 'b': (0.0, 1.0)}
    degrees = [0, 45, 90]
    offsets = [0.0, 1.0, 2.0]
    
    expected_poss = {
        (0, 0.0) : {'a': (1.0, 0.0), 'b': (0.0, 1.0)},
        (0, 1.0) : {'a': (2.0, 0.0), 'b': (1.0, 1.0)},
        (0, 2.0) : {'a': (3.0, 0.0), 'b': (2.0, 1.0)},
        (45, 0.0) : {'a': (1.0, 0.0), 'b': (0.0, 1.0)},
        (45, 1.0) : {'a': (1.0 + 1.0 * np.sqrt(2.0) / 2.0, 1.0 * np.sqrt(2.0) / 2.0), 'b': (1.0 * np.sqrt(2.0) / 2.0, 1.0 + 1.0 * np.sqrt(2.0) / 2.0)},
        (45, 2.0) : {'a': (1.0 + 2.0 * np.sqrt(2.0) / 2.0, 2.0 * np.sqrt(2.0) / 2.0), 'b': (2.0 * np.sqrt(2.0) / 2.0, 1.0 + 2.0 * np.sqrt(2.0) / 2.0)},
        (90, 0.0) : {'a': (1.0, 0.0), 'b': (0.0, 1.0)},
        (90, 1.0) : {'a': (1.0, 1.0), 'b': (0.0, 2.0)},
        (90, 2.0) : {'a': (1.0, 2.0), 'b': (0.0, 3.0)},
    }
    
    for degree in degrees:
        for offset in offsets:
            result = shift_pos_by(old_positioning=old_pos, new_positioning={}, degree=degree, offset=offset)
            
            expected_pos = expected_poss[(degree, offset)]
            for key in expected_pos:
                assert np.allclose(result[key], expected_pos[key])
            

def test_generate_equidistant_colors_count():
    colors = generate_equidistant_colors(5)
    assert len(colors) == 5
    

def test_generate_equidistant_colors_range():
    colors = generate_equidistant_colors(8)
    for rgb in colors:
        assert all(0.0 <= c <= 1.0 for c in rgb)
    

def test_generate_equidistant_colors_int_color():
    colors = generate_equidistant_colors(4, color=0)
    assert len(colors) == 4
    
    
def test_map_names():
    states = ['rest', 'move', 'turn']
    assert map_names(states, '1-0') == 'C1:rest'
    assert map_names(states, '2-1') == 'C2:move'
    assert map_names(states, '3-2') == 'C3:turn'
    
    
def test_make_integer_list_values():
    result, translation = make_integer_list(['a', 'b', 'a', 'c'])
    assert result == [0, 1, 0, 2]
    assert list(translation) == ['a', 'b', 'c']
    
    
def test_make_integer_list_unique():
    result, translation = make_integer_list(['x', 'x', 'x'])
    assert result == [0, 0, 0]
    assert len(translation) == 1