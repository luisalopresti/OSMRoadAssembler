import pandas as pd
import geopandas as gpd
from shapely.geometry import MultiLineString
from shapely.ops import linemerge, unary_union
import re
import ast
import numpy as np


# --------------------------------------------------------
#                   Data Cleaning Functions
# --------------------------------------------------------

def check_datatype(value):
    '''Function that given a column, checks that each element is in the correct datatype and convert it if needed;
    if the value is a list, it check all elements inside the list and convert them into the correct datatype as well.'''
    try:
        value = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        pass 
    
    # if value is a list, evaluate datatype of all its element
    if isinstance(value, list):
        return [check_datatype(v) for v in value]
    # if value should be an int
    elif isinstance(value, str) and value.isdigit():
        return int(value)
    # else, if value should be float (try float conversion unless error)
    elif isinstance(value, str):
        if value.strip().lower() == 'none':
            return float('nan')
        try:
            return float(value)
        except ValueError:
            pass

    return value


def check_multiple_types(column):
    '''Check if columns have multiple data types & which'''
    unique_types = column.apply(type).unique()
    return len(unique_types) > 1, unique_types


def get_max_min(lanes_value):
    '''Get the min and the max value of an attribute;
    when the attribute is an int, min and max value will be the same and equal to that int,
    when it is a list, the function will compute the min and max value in the list.'''
    if isinstance(lanes_value, list):
        return max(lanes_value), min(lanes_value)
    elif isinstance(lanes_value, int):
        return lanes_value, lanes_value
    else:
        return np.nan, np.nan


def clean_highway_value(value):
    '''Remove unclassified values from highway attribute; 
    replace with nan if necessary, or with other values available for the same geometry.'''
    if isinstance(value, list): 
        value = [v for v in value if v != 'unclassified']
        if len(value) == 1:
            return value[0]
        return value 
    
    elif isinstance(value, str): 
        if value == 'unclassified':
            return np.nan
        return value 
    return value


def process_bridge_tunnel(value, true_values):
    '''Replace OSM bridge attribute with a boolean variable stating if there exist a bridge along that road.'''
    if isinstance(value, list):
        return any(v in true_values for v in value)
    elif isinstance(value, str):
        return value.lower() in true_values
    return False


def standardize_name(name, 
                     directions = ['upper', 'lower'],
                     abbreviations = {
                                'st': 'street',
                                'rd': 'road',
                                'ave': 'avenue',
                                'blvd': 'boulevard'
                                }):

    # all to lowercanse
    name = name.lower()

    # remove punctuation
    name = re.sub(r'[^\w\s]', '', name)

    # remove directional prefixes/suffixes
    name = ' '.join(word for word in name.split() if word not in directions)

    # remove most common abbreviations
    words = name.split()
    standardized = [abbreviations.get(word, word) for word in words]
    return ' '.join(standardized)


def remove_words(name, words_to_rm = ['street', 'st', 'road', 'rd', 'square', 'ave', 'avenue', 'drive']):
    # remove road type words
    name = ' '.join(word for word in name.split() if word not in words_to_rm)
    return name



# --------------------------------------------------------
#                   Aggregation Functions
# --------------------------------------------------------

def oneway_boolean_aggregation(x):
    '''
    If multiple bool values are present when aggregating 
    (both True and False for the aggregated segments - meaning some segments are one way and other not),
    return False [aka at least part of the street-block is not oneway], 
    otherwise keep the same boolean value.
    '''
    unique_elements = x.dropna().unique()
    if len(unique_elements) > 1:
        return False
    else:
        return unique_elements[0] if unique_elements.size>0 else None
    

def bridge_tunnel_boolean_aggregation(x):
    '''
    If multiple bool values are present when aggregating 
    (both True and False for the aggregated segments),
    return True [there is a bridge/tunnel along the aggregated road], 
    otherwise keep the same boolean value.
    '''
    unique_elements = x.dropna().unique()
    if len(unique_elements) > 1:
        return True
    else:
        return unique_elements[0] if unique_elements.size>0 else None
    
    
def str_list_aggregation(names):
    '''
    Attribute can be str or list of str. 
    Aggregate as follows:
        - if there is a unique value (either list or str), keep it 
        - if multiple values, put all in a single list (no nested list), with no duplicates
    Final values should be either a str or a list of (unique) str.
    '''
    unique_values = set()  
    result = []

    for name in names:
        if isinstance(name, str) or isinstance(name, int):  # if str or int, add to set
            unique_values.add(name)
        elif isinstance(name, list):  # if list of str, extend the set with the list
            unique_values.update(name)

    result = list(unique_values)

    if len(result) == 1:
        return result[0]
    if len(result) == 0:
        return None
    
    result.sort()
    return result



# --------------------------------------------------------
#          Continuity/Proximity Check & Merge
# --------------------------------------------------------

def are_contiguous(line1, line2):
    '''
    Check if two LineStrings are contiguous (share an endpoint or overlap).
    '''
    return line1.touches(line2) or line1.overlaps(line2)


def merge_contiguous(geometries):
    '''
    Merge contiguous LineStrings into single LineStrings iteratively.
    '''    
    merged = list(geometries)
    changed = True
    while changed:
        new_merged = []
        used = set()
        changed = False
        
        for i, geom1 in enumerate(merged):
            if i in used:
                continue
            group = [geom1]
            for j, geom2 in enumerate(merged):
                if j in used or i == j:
                    continue
                if any(are_contiguous(g, geom2) for g in group):
                    group.append(geom2)
                    used.add(j)
                    changed = True
            # merge the group into a single LineString or keep it as is if 1 element
            if len(group) > 1:
                merged_geom = linemerge(unary_union(group))
                if isinstance(merged_geom, MultiLineString): # if MultiLinestring and not LineString
                    # handle cases where linemerge doesn't fully merge into single LineString
                    merged_geom = linemerge(unary_union(merged_geom))
                new_merged.append(merged_geom)
            else:
                new_merged.append(group[0])
            used.add(i)
        
        merged = new_merged
    
    return merged



def are_contiguous_or_in_proximity(line1, line2, max_distance = 1e-4):
    '''
    Check if two LineStrings or MultiLineStrings are contiguous (share an endpoint, overlap),
    or if their distance is less than a threshold in meters (max_distance).
    '''
    
    return (line1.touches(line2) or 
            line1.overlaps(line2) or 
            line1.distance(line2) < max_distance)


def proximity_merge(geometries, max_distance = 1e-4):
    '''
    Merge LineStrings that are contiguous or in close proximity into single LineStrings iteratively.
    '''    
    merged = list(geometries)
    changed = True
    while changed:
        new_merged = []
        used = set()
        changed = False
        
        for i, geom1 in enumerate(merged):
            if i in used:
                continue
            group = [geom1]
            for j, geom2 in enumerate(merged):
                if j in used or i == j:
                    continue
                if any(are_contiguous_or_in_proximity(g, geom2, max_distance) for g in group):
                    group.append(geom2)
                    used.add(j)
                    changed = True
            # merge the group into a single LineString or keep it as is if 1 element
            if len(group) > 1:
                merged_geom = unary_union(group)
                if isinstance(merged_geom, MultiLineString): # if MultiLinestring and not LineString
                    # handle cases where linemerge doesn't fully merge into single LineString
                    merged_geom = unary_union(merged_geom)
                new_merged.append(merged_geom)
            else:
                new_merged.append(group[0])
            used.add(i)
        
        merged = new_merged
    
    return merged



# --------------------------------------------------------
#           Merge Roads with Multiple Names
# --------------------------------------------------------

def merge_within_set_names(set_names):
    '''Merge geometries within set_names, 
    if they have one or more names in common and are contiguous/distance < 1e-4'''
    merged = True

    while merged:
        merged = False ## no merging done yet
        new_set_names = []
        processed_indices = set()

        for i in range(len(set_names)):
            if i in processed_indices:
                continue

            current_geom = set_names.at[i, 'geometry']
            current_names = set(set_names.at[i, 'standardized_name'])

            for j in range(i + 1, len(set_names)):
                if j in processed_indices:
                    continue

                other_geom = set_names.at[j, 'geometry']
                other_names = set(set_names.at[j, 'standardized_name'])

                # if they share name(s) and meet the proximity condition
                if current_names & other_names and are_contiguous_or_in_proximity(current_geom, other_geom): 
                    current_geom = current_geom.union(other_geom) 
                    current_names.update(other_names)
                    processed_indices.add(j)
                    merged = True

            new_set_names.append({'geometry': current_geom, 'standardized_name': sorted(list(current_names))})
            processed_indices.add(i)

        set_names = gpd.GeoDataFrame(new_set_names)

    return set_names



# --------------------------------------------------------
#               Duplicates & Overlaps Checks
# --------------------------------------------------------

def find_duplicate_or_overlapping_segments(gdf):
    '''
    Identify geometries in a GeoDataFrame that are overlapping or duplicated 
    (i.e., one segment is fully or partially identical to another).
    
    Input:
        - gdf: GeoDataFrame containing LineString or MultiLineString geometries.

    Output:
        - overlapping_pairs (list): list of tuples where each tuple represents two indices of geometries that overlap or are duplicates.
          Returns an empty list if no overlaps/duplicates are found.
    '''
    overlapping_pairs = []
    
    for i, geom in enumerate(gdf.geometry):
        for j, other_geom in gdf.geometry.iloc[i + 1:].items():
            if geom.overlaps(other_geom) or geom.equals(other_geom) or geom.within(other_geom):
                overlapping_pairs.append((i, j))
    
    return overlapping_pairs




# --------------------------------------------------------
#                Others, general utils
# --------------------------------------------------------

def merge_lists(ids_to_aggr):
    '''Takes a list of lists (ids_to_aggr) and 
    iteratively merges any lists that share common elements.'''
    merged = True
    while merged:  
        merged = False
        for i, lst1 in enumerate(ids_to_aggr):
            for j, lst2 in enumerate(ids_to_aggr):
                if i != j and set(lst1) & set(lst2):  
                    ids_to_aggr[i] = list(set(lst1) | set(lst2))  
                    ids_to_aggr.pop(j)
                    merged = True
                    break
            if merged:
                break
    return ids_to_aggr