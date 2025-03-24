import pandas as pd
import geopandas as gpd
import osmnx as ox
from collections import Counter
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union
from utils import *


# --------------------------------------------------------
#          Load data (from osmnx) & Clean
# --------------------------------------------------------

def load_roads_from_placename(place_name, network_type='drive', simplify=True):
    '''From osmnx graph_from_place, extract edges (roads) and build a GeoDataFrame'''
    graph = ox.graph_from_place(place_name, network_type=network_type, simplify=simplify)
    # graph to gdf
    _, edges = ox.graph_to_gdfs(graph)
    edges.reset_index(inplace=True, drop=False)
    return edges

def correct_datatypes(df, columns_tocheck):
    '''Ensure datatypes of the passed columns are correct.'''
    for col in columns_tocheck:
        df[col] = df[col].apply(check_datatype)


def process_maxspeed(value):
    '''
    Apply get_num function to maxspeed column,
    to extract maxspeed number when it is encoded within a string
    (e.g., '30 mph').
    '''
    if isinstance(value, str):
        return get_num(value)
    elif isinstance(value, list):
        return [process_maxspeed(item) for item in value]
    elif pd.isna(value):
        return np.nan
    else:
        return value
    

def process_list_values(edges_df, 
                        tunnel_true_values = ['yes', 'building_passage'], # ref. taginfo.openstreetmap.org/keys/tunnel
                        bridge_true_values = ['yes', 'movable', 'viaduct', 'aqueduct']): # ref. taginfo.openstreetmap.org/keys/bridge
    '''When a geometry have multiple values for an attribute, extract information
    to be able to carry out meaningful aggregations later on.
    
    List-values processing implemented for the following attributes:
    - lanes: decomposed in two column, for min and max lanes along that segment
    - maxspeed: decomposed in two column, for min and max maxspeed along that segment
    - highway: replace unclassified with nan or other existing values for that segment
    - tunnel: becomes a bool stating whether or not there is a tunnel along that street
    - bridge: becomes a bool stating whether or not there is a bridge along that street
    '''
    # lanes
    edges_df['lanes'] = edges_df['lanes'].apply(lambda x: [i for i in x if not isinstance(i, str)] if isinstance(x, list) else float('nan') if isinstance(x, str) else x) ## only numeric are valid 
    edges_df[['max_lanes', 'min_lanes']] = edges_df['lanes'].apply(lambda x: pd.Series(get_max_min(x)))
    edges_df.drop(columns=['lanes'], inplace=True)

    # maxspeed
    ## after correction datatype, should only be numeric: exceptions are string containing number+mph & mapping errors
    ## info: https://taginfo.openstreetmap.org/keys/maxspeed
    ## 1. remove mph & transform to numeric (e.g., instead of 30, the value is '30 mph')
    edges_df['maxspeed'] = edges_df['maxspeed'].apply(process_maxspeed)
    ## 2. remove mapping errors (e.g., in Hamburg, maxspeed assumes values "signals" and "walk")
    edges_df['maxspeed'] = edges_df['maxspeed'].apply(lambda x: [i for i in x if not isinstance(i, str)] if isinstance(x, list) else float('nan') if isinstance(x, str) else x)
    ## apply function to get min and max maxspeed value in that road-block
    edges_df[['max_maxspeed', 'min_maxspeed']] = edges_df['maxspeed'].apply(lambda x: pd.Series(get_max_min(x)))
    edges_df.drop(columns=['maxspeed'], inplace=True)

    # highway
    edges_df['highway'] = edges_df['highway'].apply(clean_highway_value)

    # tunnel
    edges_df['tunnel'] = edges_df['tunnel'].apply(process_bridge_tunnel, args=(tunnel_true_values,))

    # bridge
    edges_df['bridge'] = edges_df['bridge'].apply(process_bridge_tunnel, args=(bridge_true_values,))

    


# --------------------------------------------------------
#                   Common Aggregations
# --------------------------------------------------------

def aggregate_segments_info(gdf, grp_by_col = 'geometry'):
    '''Function to aggregate information coming from different segments composing the same road,
    so that the final road representation will not lose important attributes.'''
    aggregation_dict = {'osmid': str_list_aggregation,
                        'min_lanes': 'min', 'max_lanes': 'max', 
                        'min_maxspeed' : 'min', 'max_maxspeed': 'max', 
                        'oneway': oneway_boolean_aggregation,
                        'name': str_list_aggregation,
                        'highway': str_list_aggregation,
                        'tunnel': bridge_tunnel_boolean_aggregation, 
                        'bridge': bridge_tunnel_boolean_aggregation}
    
    if 'standardized_name' in gdf.columns:
        aggregation_dict['standardized_name'] = str_list_aggregation
    
    return gdf.groupby(grp_by_col).agg(aggregation_dict) 
    

# --------------------------------------------------------
#                   Process Roundabout
# --------------------------------------------------------

def extract_roundabouts(edges_df, junction_values = ['roundabout', 'circular']):
    '''
    Extract roundabouts from edges dataframe; 
    in OSM, roundabouts are marked with the attribute _junction_, using values _roundabout_ or _circular_
    '''
    # get df with roundabouts only
    identified_junctions = edges_df[edges_df['junction'].isin(junction_values)].reset_index(drop=True)
    # remove roundabouts from main dataset
    edges_df = edges_df[~edges_df['junction'].isin(junction_values)].reset_index(drop=True)
    return identified_junctions, edges_df

def continous_roundabout(junctions_gdf):
    '''Build a continuous representation of each roundabout based on their building linestrings.'''
    geometries = junctions_gdf.geometry.tolist()
    merged_geometries = merge_contiguous(geometries)
    roundabout_gdf = gpd.GeoDataFrame(geometry=merged_geometries, crs=junctions_gdf.crs)
    return roundabout_gdf


def augment_roundabouts(roundabout_gdf, junctions_gdf, edges_df):
    '''Add to the complete junction representation the information of their composing segments (e.g., name, maxspeed, etc.)'''
    # spatial joint of roundabout dataframe with full dataset, to preserve other attributes
    # each segments is assigned to the belonging roundabout
    roundabout_gdf['roundabout_full_geom'] = roundabout_gdf.geometry
    complete_junctions = gpd.sjoin(junctions_gdf, roundabout_gdf, how='left', predicate='within')
    # aggregate attributes
    grp_complete_junctions = aggregate_segments_info(complete_junctions, grp_by_col = 'roundabout_full_geom')
    grp_complete_junctions['junction'] = 'roundabout'
    grp_complete_junctions.reset_index(inplace=True)
    grp_complete_junctions.rename(columns={'roundabout_full_geom':'geometry'}, inplace=True)
    # add is_closed attribute (bool) to determine whether segments are missing to complete the roundabout representation
    grp_complete_junctions['is_closed'] = grp_complete_junctions.geometry.apply(lambda x: x.is_ring if isinstance(x, LineString) else False)
    # add roundabout to overall street network data (edges_df)
    edges_df = pd.concat([edges_df, grp_complete_junctions], ignore_index=True)
    return edges_df


# --------------------------------------------------------
#           Build Continuous Road Representations
# --------------------------------------------------------

def process_road_names(edges_df, 
                        directions = ['upper', 'lower'],
                        abbreviations = {
                                'st': 'street',
                                'rd': 'road',
                                'ave': 'avenue',
                                'blvd': 'boulevard'
                                },
                        words_to_rm = ['street', 'st', 'road', 'rd', 
                                       'square', 'ave', 'avenue', 'drive']):
    
    '''Standardize road names & remove void words (modify edges_df inplace).'''

    edges_df['standardized_name'] = None

    for i in range(len(edges_df)):
        if isinstance(edges_df.at[i, 'name'], list):
            # standardize names & remove type words in lists of names
            std_names = [remove_words(standardize_name(name, directions, abbreviations), words_to_rm) for name in edges_df.at[i, 'name']]
            std_names = list(set(std_names))
            if len(std_names) == 1:
                std_names = std_names[0]
            else:
                std_names.sort()
            edges_df.at[i, 'standardized_name'] = std_names

        elif isinstance(edges_df.at[i, 'name'], str):
            # apply normalization to string names
            edges_df.at[i, 'standardized_name'] = standardize_name(edges_df.at[i, 'name'], directions, abbreviations)
            edges_df.at[i, 'standardized_name'] = remove_words(edges_df.at[i, 'standardized_name'], words_to_rm)

        else:
            pass


def merge_equal_name_sets(multiple_names_edges, verbose = True):
    '''
    Get geometries having multiple standardized names: 
    merge those that have all names in common.
    '''
    # group by unique set of values
    multiple_names_edges.loc[:, 'standardized_name'] = multiple_names_edges['standardized_name'].apply(frozenset)

    if verbose:
        print('Unique sets:', len(Counter(multiple_names_edges.standardized_name).keys()))
        print('Total occurrences:', len(multiple_names_edges))
        
    grouped = multiple_names_edges.groupby('standardized_name')

    ## linemerge geometries having same set of names (all names in the set are the same)
    # perform linemerge for geometries within each group (if multiple element)
    merged_rows = []
    for name, group in grouped:
        if len(group) > 1:
            ## NOTE: did not check contiguity, in order to merge also multilanes of same road (parallel)
            ## TODO: may add proximity check if deemed necessary, but will slow performances
            merged_geom = linemerge(unary_union(group.geometry))
            merged_rows.append({
                'standardized_name': name,
                'geometry': merged_geom
            })
        else:
            merged_rows.append({
                'standardized_name': group.standardized_name.item(),
                'geometry': group.geometry.item()
            })

    linemerged_sets = gpd.GeoDataFrame(merged_rows, geometry='geometry', crs=multiple_names_edges.crs)
    linemerged_sets.loc[:, 'standardized_name'] = linemerged_sets['standardized_name'].apply(list).apply(sorted)
    
    # add other attributes 
    linemerged_sets['full_geom'] = linemerged_sets.geometry
    complete_linemerged_sets = gpd.sjoin(multiple_names_edges, linemerged_sets, how='left', predicate='within')
    complete_linemerged_sets.rename(columns={'standardized_name_right':'standardized_name'}, inplace=True)

    # aggregate
    grp_complete_linemerged_sets = aggregate_segments_info(complete_linemerged_sets, 'full_geom')
    grp_complete_linemerged_sets.reset_index(inplace=True)
    grp_complete_linemerged_sets.rename(columns={'full_geom':'geometry'}, inplace=True)
    grp_complete_linemerged_sets = gpd.GeoDataFrame(grp_complete_linemerged_sets, geometry='geometry', crs=multiple_names_edges.crs)

    return grp_complete_linemerged_sets 


def merge_segments_with_multiple_names(lst_names_df, verbose = True):
    lst_merged_geometries = merge_within_set_names(lst_names_df)

    if verbose:
        print(' ----------- RESULT PROCESSING LIST NAMES: ----------- ')
        print(f'Reduced from {len(lst_names_df)} to {len(lst_merged_geometries)} geometries.')

    lst_merged_geometries = gpd.GeoDataFrame(lst_merged_geometries, geometry='geometry', crs = lst_names_df.crs)

    # ADD OTHER ATTRIBUTES
    lst_merged_geometries['full_geom'] = lst_merged_geometries.geometry
    lst_merged_geometries = gpd.sjoin(lst_names_df, lst_merged_geometries, how='left', predicate='within')
    lst_merged_geometries.rename(columns={'standardized_name_right':'standardized_name'}, inplace=True)

    # aggregate 
    grp_lst_merged_geometries = aggregate_segments_info(lst_merged_geometries, grp_by_col = 'full_geom')
    grp_lst_merged_geometries.reset_index(inplace=True)
    grp_lst_merged_geometries.rename(columns={'full_geom':'geometry'}, inplace=True)
    grp_lst_merged_geometries = gpd.GeoDataFrame(grp_lst_merged_geometries, geometry='geometry', crs=lst_names_df.crs)
    return grp_lst_merged_geometries


def merge_segments_with_single_name(str_names_df, verbose = True):
    grp = str_names_df.groupby('standardized_name')
    merged_geometries = []
    for name, group in grp:
        # if multiple segments with the same name (in the same grp)
        if len(group) > 1:
            # if contiguous or in close proximity (ideally to include different lanes of the same road in the merging)
            # maximum distance set to 1e-4 meters (we look for an infinitesimely small number, 
            # as empirically the distance is 0.0 meters, but without any touching point between the lanes)
            geometries = group.geometry.to_list()
            merged_geom = proximity_merge(geometries) 
            merged_geometries.extend(merged_geom)
        else:
            merged_geometries.append(group.geometry.item())

    str_merged_geometries = gpd.GeoDataFrame(merged_geometries, columns=['geometry'], geometry='geometry', crs=str_names_df.crs)

    # ADD OTHER ATTRIBUTES
    str_merged_geometries['full_geom'] = str_merged_geometries.geometry
    complete_str_merged_geometries = gpd.sjoin(str_names_df, str_merged_geometries, how='left', predicate='within')

    if len(complete_str_merged_geometries) != len(str_names_df):
        raise ValueError('Each original segment should be assigned to a unique block of LineStrings!')

    if verbose:
        print(' ----------- RESULT PROCESSING STRING NAMES: ----------- ')
        print('Original number of segments (with unique name - as string):', len(str_names_df))
        print('Final number of MultiLineStrings (geometries blocks) obtained:', complete_str_merged_geometries.full_geom.nunique())

    # aggregate 
    grp_complete_str_merged_geometries = aggregate_segments_info(complete_str_merged_geometries, grp_by_col = 'full_geom')
    grp_complete_str_merged_geometries.reset_index(inplace=True)
    grp_complete_str_merged_geometries.rename(columns={'full_geom':'geometry'}, inplace=True)
    grp_complete_str_merged_geometries = gpd.GeoDataFrame(grp_complete_str_merged_geometries, geometry='geometry', crs=str_names_df.crs)
    return grp_complete_str_merged_geometries



# --------------------------------------------------------
#        Merge Roads with Single & Multiple Names
#                     (Recursive)
# --------------------------------------------------------

# For each geometry having at least one name in common, check if they satify are_contiguous_or_in_proximity requirements;
# If yes, add thier ids are in the same list for merging.
# An id may be present in multiple lists: this means that those lists should be merged into a single list and all geometries should be merged together.
# This is done via the `merge_lists` function and it is an essential step to avoid overlapping and geometries belonging to multiple blocks.

def add_geoms_IDs(edges_df):
    edges_df['ID_geom'] = range(len(edges_df)) ## once df exploded, repeating geoms will have the same id
    return edges_df


def geoms_id_to_merge(edges_df, max_distance = 0.0001, crs_metric='EPSG:2157'):

    exploded_df = edges_df.copy()
    exploded_df = exploded_df.set_index('ID_geom').explode('standardized_name') # each name in list-name in diff lines, thus repeating geoms for as many times as their number of names
    exploded_df.sort_values(by=['standardized_name', 'geometry'], inplace=True)
    exploded_df.to_crs(crs_metric, inplace=True)

    merged_ids_df = pd.DataFrame([], columns=['IDs'])

    for std_name in exploded_df.standardized_name.unique():
        std_name_df = exploded_df[exploded_df.standardized_name == std_name]

        tpl_ids = [[id] for id in std_name_df.index] # list of ids
        merged_ids = [] # list ids of geoms to merge

        while tpl_ids:
            current_ids = tpl_ids.pop(0)

            # evaluate geoms in same group for merging
            to_merge = [current_ids]
            remaining_ids = []

            for other_ids in tpl_ids:
                if are_contiguous_or_in_proximity(std_name_df.loc[current_ids[0], 'geometry'], 
                                                std_name_df.loc[other_ids[0], 'geometry'],
                                                max_distance=max_distance): ## NOTE increasing max distance (in meters)
                    to_merge.append(other_ids)
                else:
                    remaining_ids.append(other_ids)

            # merge ids in a single list 
            merged_group = [item for sublist in to_merge for item in sublist]  
            merged_ids.append(merged_group)
            
            # update tpl_ids
            tpl_ids = remaining_ids

        # add ids of geometries to the final DataFrame; geometries ids in the same list must be merged
        for ids in merged_ids:
            merged_ids_df = pd.concat([merged_ids_df, pd.DataFrame([[ids]], columns=['IDs'])], ignore_index=True)

    
    # get only ids of geometries that need to be merged (more than one id per list)
    merged_ids_df['num_geoms'] = merged_ids_df.IDs.apply(len)
    return merged_ids_df[merged_ids_df['num_geoms']>1].IDs.to_list()


def IDs_merging(edges_df, ids_to_merge, crs_latlon='EPSG:4326'):
    ## apply merging according to IDs
    aggr_geoms_gdf = edges_df.copy()
    for list_ids in ids_to_merge:
        # extract geoms to merge
        geoms_to_merge = aggr_geoms_gdf[aggr_geoms_gdf.ID_geom.isin(list_ids)].reset_index(drop=True)
        geoms_to_merge = gpd.GeoDataFrame(geoms_to_merge, geometry='geometry', crs=crs_latlon)

        # remove unmerged geoms from df
        aggr_geoms_gdf = aggr_geoms_gdf[~aggr_geoms_gdf.ID_geom.isin(list_ids)]

        # merge geometries
        new_geoms = proximity_merge(geoms_to_merge.geometry)

        # merge geometries
        new_geoms_df = gpd.GeoDataFrame(new_geoms, columns=['geometry'], geometry='geometry', crs=crs_latlon) 

        # spatial join + group by to add the other features/columns to the geometry
        new_geoms_df['full_geom'] = new_geoms_df.geometry
        new_geoms_plus_feats = gpd.sjoin(geoms_to_merge, new_geoms_df, how='left', predicate='within')
        grp_new_geoms_plus_feats = aggregate_segments_info(new_geoms_plus_feats, grp_by_col = 'full_geom')

        grp_new_geoms_plus_feats.reset_index(inplace=True)
        grp_new_geoms_plus_feats.rename(columns={'full_geom':'geometry'}, inplace=True)

        # add merged geoms to df
        aggr_geoms_gdf = pd.concat([grp_new_geoms_plus_feats, aggr_geoms_gdf], ignore_index=True)

    # to gdf
    aggr_geoms_gdf = gpd.GeoDataFrame(aggr_geoms_gdf, geometry='geometry', crs=crs_latlon)

    return aggr_geoms_gdf


def apply_IDs_merging(edges_df, max_distance = 20, crs_latlon='EPSG:4326', crs_metric='EPSG:2157'):
    edges_df = add_geoms_IDs(edges_df)
    ids_to_merge = merge_lists(geoms_id_to_merge(edges_df, max_distance, crs_metric))
    edges_df_merge = IDs_merging(edges_df, ids_to_merge, crs_latlon)
    return edges_df_merge


def recursive_IDs_merging(edges_df, max_distance = 20, crs_latlon='EPSG:4326', crs_metric='EPSG:2157', i = 0, i_max = 10):
    edges_df_merge = apply_IDs_merging(edges_df, max_distance, crs_latlon)

    # stop when no more changes occur
    # if all(edges_df_merge.geom_equals_exact(edges_df, tolerance=0.0001)): # doesnt work cause applies row-wise comparison (different order means not equals)
    # if edges_df_merge.sort_values(by='geometry').geometry.tolist() == edges_df.sort_values(by='geometry').geometry.tolist(): # this works as well
    if len(edges_df_merge) == len(edges_df):
        return edges_df_merge

    # recursive case: apply function until no changes or max iteration reached
    i+=1
    return recursive_IDs_merging(edges_df_merge, max_distance, crs_latlon, crs_metric, i, i_max)



# --------------------------------------------------------
#          Function Encapsulating all Operations
#             for Continuous Representations
# --------------------------------------------------------

def merge_segments(edges_df, 
                   crs_metric = 'EPSG:2157', crs_latlon = 'EPSG:4326', 
                   max_distance = 20, i = 0, i_max = 100,
                   verbose = True):
    
    # project to metric system for proximity/contiguity operations
    edges_df = edges_df.to_crs(crs_metric)

    ## divide 3 cases, based on standardized_name datatype
    str_names = edges_df[edges_df.standardized_name.apply(isinstance, args=(str,))].reset_index(drop=True)
    lst_names = edges_df[edges_df.standardized_name.apply(isinstance, args=(list,))].reset_index(drop=True)
    # missing_names = edges_df[edges_df.standardized_name.isna()].reset_index(drop=True)

    # --------------------------------------------------------
    #                   Process LIST names
    # --------------------------------------------------------
    ## name as instance of list
    # there are multiple names associated to the same road
    # Merge geometries with multiple name with one another 
    # if they have at least one name in common and they are contiguous or within a 1e-4 distance:
    lst_names = merge_equal_name_sets(lst_names, verbose) # all names are the same --> merge
    grp_lst_merged_geometries = merge_segments_with_multiple_names(lst_names, verbose) # at least one name in common + proximity --> merge

    # --------------------------------------------------------
    #                  Process STRING names
    # --------------------------------------------------------    
    ## name as instance of string
    # the road has a unique name
    # merge geometries with same name and satisfying contiguity (or within a 1e-4 distance) criterion:
    grp_complete_str_merged_geometries = merge_segments_with_single_name(str_names, verbose)

    # --------------------------------------------------------
    #                  Process NAN names
    # --------------------------------------------------------
    ## missing name
    # minor or internal roads
    # missing_names dataframe --> drop


    # --------------------------------------------------------
    #                      MERGE RESULTS 
    # --------------------------------------------------------

    # concatenate
    final_edges = pd.concat([grp_complete_str_merged_geometries, grp_lst_merged_geometries], ignore_index=True)
    # to GeoDataFrame (with crs_metric)
    final_edges = gpd.GeoDataFrame(final_edges, geometry='geometry', crs=crs_metric)
    # change crs
    final_edges.to_crs(crs_latlon, inplace=True)


    # --------------------------------------------------------
    #         MERGE LIST & STR NAMES WITH ONE ANOTHER
    #                     (RECURSIVELY) 
    # --------------------------------------------------------

    final_edges_opt = recursive_IDs_merging(final_edges, max_distance, crs_latlon, crs_metric, i, i_max) 

    if verbose:
        print(f'Reduction in number of geometries after recursion: from {len(final_edges)} to {len(final_edges_opt)}')
        print(f'Overall reduction: from {len(edges_df)} in the original dataset, to {len(final_edges_opt)}.')

    # update geometry length
    final_edges_opt['length'] = final_edges_opt.geometry.to_crs(crs_metric).length

    return final_edges_opt


