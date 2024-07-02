import colorsys
import numpy as np


def shift_pos_by(old_positioning, new_positioning, degree, offset):
    """
        Shift positions in polar coordinates.

        Parameters:
        - old_positioning: Dictionary of node positions.
        - new_positioning: Dictionary of new node positions will be updated
        - degree: Degree to shift positions.
        - offset: Offset distance.

        Returns:
        - new_positioning: Updated dictionary of node positions.
    """
    for node, coords in old_positioning.items():
        new_positioning[node] = (coords[0] + offset * np.cos(np.radians(degree)),
                                 coords[1] + offset * np.sin(np.radians(degree)))
    return new_positioning


def generate_equidistant_colors(n, color=None):
    """
        Generate a list of RGB colors in HSV space with equidistant hues.

        Parameters:
        - n: Number of colors to generate.

        Returns:
        - colors: List of RGB colors.
    """
    colors = []
    if int == type(color):
        color = int(color % 3)
        for i in range(n):
            val = i / n  # value
            rgb = [val, val, val]
            rgb[color] += 2 - np.exp(val)
            colors.append(tuple(rgb))
    else:
        for i in range(n):
            hue = i / n  # hue value
            saturation = 1.0  # fully saturated
            value = 1.0  # full brightness
            rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb_color)
    return colors


def map_names(states, name):
    """
    Used to generate a state-name from a number
    """
    c, b = name.split('-')
    new_name = f'C{c}:{states[int(b)]}'
    return new_name


def make_integer_list(input_list):
    """
        Convert a list of strings to a list of integers and create a translation list.

        Parameters:
        - input_list: List of strings.

        Returns:
        - integer_list: List of integers corresponding to input_list.
        - translation_list: List of unique strings in input_list.
    """
    string_to_int = {}
    integer_list = []

    for s in input_list:
        if s not in string_to_int:
            string_to_int[s] = len(string_to_int)
        integer_list.append(string_to_int[s])

    translation_list = np.asarray(list(string_to_int.keys())).astype(str)

    return integer_list, translation_list
