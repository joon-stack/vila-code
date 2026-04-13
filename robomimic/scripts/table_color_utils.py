"""
Utility functions for modifying table color in MuJoCo environments.
Used for background change experiments.
"""

import re
import json
import os


def load_table_colors(config_path=None):
    """
    Load table color configurations from JSON file.
    
    Args:
        config_path (str): Path to the JSON config file. If None, uses default path.
    
    Returns:
        list: List of color dictionaries with 'name', 'rgb', etc.
    """
    if config_path is None:
        # Default path relative to this file
        config_path = os.path.join(os.path.dirname(__file__), "table_colors_20.json")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config["colors"]


def get_table_color(index, config_path=None):
    """
    Get a specific table color by index.
    
    Args:
        index (int): Color index (0-19)
        config_path (str): Path to the JSON config file.
    
    Returns:
        dict: Color dictionary with 'name', 'rgb', etc.
    """
    colors = load_table_colors(config_path)
    if index < 0 or index >= len(colors):
        raise ValueError(f"Color index {index} out of range. Valid range: 0-{len(colors)-1}")
    return colors[index]


def modify_table_color_in_xml(xml_string, rgb_color):
    """
    Modify the table texture color in a MuJoCo XML string.
    
    This function finds the 'textable' texture element and modifies its rgb1 and rgb2 
    attributes to change the table color. It also modifies 'tex-ceramic' if present.
    
    Args:
        xml_string (str): The MuJoCo XML model string
        rgb_color (list or tuple): RGB color values [r, g, b] in range 0-1
    
    Returns:
        str: Modified XML string with new table color
    """
    r, g, b = rgb_color
    new_rgb_str = f"{r} {g} {b}"
    
    # Pattern to match textable texture with rgb1 and rgb2 attributes
    # Example: <texture name="textable" builtin="flat" height="512" width="512" rgb1="0.5 0.5 0.5" rgb2="0.5 0.5 0.5"/>
    
    # Modify textable rgb1
    xml_string = re.sub(
        r'(<texture[^>]*name="textable"[^>]*rgb1=")[^"]*(")',
        rf'\g<1>{new_rgb_str}\g<2>',
        xml_string
    )
    
    # Modify textable rgb2
    xml_string = re.sub(
        r'(<texture[^>]*name="textable"[^>]*rgb2=")[^"]*(")',
        rf'\g<1>{new_rgb_str}\g<2>',
        xml_string
    )
    
    # Also try matching with different attribute orders
    xml_string = re.sub(
        r'(<texture[^>]*type="cube"[^>]*name="textable"[^>]*rgb1=")[^"]*(")',
        rf'\g<1>{new_rgb_str}\g<2>',
        xml_string
    )
    
    xml_string = re.sub(
        r'(<texture[^>]*type="cube"[^>]*name="textable"[^>]*rgb2=")[^"]*(")',
        rf'\g<1>{new_rgb_str}\g<2>',
        xml_string
    )
    
    return xml_string


def set_table_color_in_sim(env, rgb_color):
    """
    Set table color at runtime by modifying MuJoCo simulation material properties.
    
    This function directly modifies the material RGBA values in the MuJoCo simulation
    without needing to reload the XML. Useful for testing.
    
    Args:
        env: robomimic environment wrapper (EnvRobosuite)
        rgb_color (list or tuple): RGB color values [r, g, b] in range 0-1
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        sim = env.base_env.sim
        r, g, b = rgb_color
        rgba = [r, g, b, 1.0]
        
        modified = False
        
        # Try to find and modify table_mat material
        try:
            mat_id = sim.model.mat_name2id("table_mat")
            sim.model.mat_rgba[mat_id] = rgba
            modified = True
        except:
            pass
        
        # Try to find and modify table_ceramic material
        try:
            mat_id = sim.model.mat_name2id("table_ceramic")
            sim.model.mat_rgba[mat_id] = rgba
            modified = True
        except:
            pass
        
        # Try to modify geom directly if material modification didn't work
        if not modified:
            # Find table-related geoms by name
            for i in range(sim.model.ngeom):
                geom_name = sim.model.geom_id2name(i)
                if geom_name and ('table' in geom_name.lower()):
                    sim.model.geom_rgba[i] = rgba
                    modified = True
        
        return modified
        
    except Exception as e:
        print(f"Warning: Failed to set table color at runtime: {e}")
        return False

# Alias for backward compatibility
set_table_color_runtime = set_table_color_in_sim


def get_available_material_names(env):
    """
    Get list of available material names in the environment.
    Useful for debugging which materials can be modified.
    
    Args:
        env: robomimic environment wrapper (EnvRobosuite)
    
    Returns:
        list: List of material names
    """
    try:
        sim = env.base_env.sim
        materials = []
        for i in range(sim.model.nmat):
            name = sim.model.mat_id2name(i)
            if name:
                materials.append(name)
        return materials
    except Exception as e:
        print(f"Warning: Failed to get material names: {e}")
        return []


def get_available_geom_names(env):
    """
    Get list of available geom names in the environment.
    Useful for debugging which geoms can be modified.
    
    Args:
        env: robomimic environment wrapper (EnvRobosuite)
    
    Returns:
        list: List of geom names
    """
    try:
        sim = env.base_env.sim
        geoms = []
        for i in range(sim.model.ngeom):
            name = sim.model.geom_id2name(i)
            if name:
                geoms.append(name)
        return geoms
    except Exception as e:
        print(f"Warning: Failed to get geom names: {e}")
        return []


if __name__ == "__main__":
    # Test the functions
    print("Testing table color utilities...")
    
    # Test loading colors
    colors = load_table_colors()
    print(f"Loaded {len(colors)} colors:")
    for color in colors:
        print(f"  {color['index']}: {color['name']} - RGB {color['rgb']}")
    
    # Test XML modification
    test_xml = '''
    <mujoco>
        <asset>
            <texture type="cube" name="textable" builtin="flat" rgb1="0.5 0.5 0.5" rgb2="0.5 0.5 0.5" width="512" height="512"/>
        </asset>
    </mujoco>
    '''
    
    modified_xml = modify_table_color_in_xml(test_xml, [0.8, 0.2, 0.2])
    print("\nOriginal XML:")
    print(test_xml)
    print("\nModified XML (red table):")
    print(modified_xml)
