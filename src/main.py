from process_roads import *
import argparse

def main(place_name, simplify = False, tolerance = 100.0, crs_latlon = 'EPSG:4326', crs_metric = 'EPSG:2157'):
    # --------------------------------------------------------
    #                        Load data 
    # --------------------------------------------------------

    # get city road network as gdf
    edges = load_roads_from_placename(place_name = place_name,  network_type='drive', simplify=True)
    # check datatypes
    correct_datatypes(edges, ['osmid', 'oneway', 'lanes', 'name', 'highway', 'maxspeed', 'tunnel', 'bridge', 'width', 'junction', 'est_width'])
    # process attributes for meaningful aggregation
    process_list_values(edges)


    # --------------------------------------------------------
    #                   Process Roundabouts
    # --------------------------------------------------------

    # extract roundabout segments
    identified_junctions, edges = extract_roundabouts(edges)
    # get junctions building geometries and compose the roundabouts
    roundabout_gdf = continous_roundabout(identified_junctions)
    # augment roundabout with other information from OSM (e.g., name, maxspeed, etc.) linked to the identified junction segments
    # and augment overall edges dataset
    edges = augment_roundabouts(roundabout_gdf, identified_junctions, edges)


    # --------------------------------------------------------
    #           Build Continuous Road Representations
    # --------------------------------------------------------

    process_road_names(edges)
    final_edges = merge_segments(edges, crs_metric = crs_metric, crs_latlon = crs_latlon, verbose = True)


    # --------------------------------------------------------
    #           Apply Douglas-Peucker algorithm to
    #               final geometries (optional)
    # --------------------------------------------------------
    if simplify:
        final_edges['simplified_geometry'] = final_edges['geometry'].to_crs(crs_metric).simplify(tolerance, preserve_topology=True).to_crs(crs_latlon)


    # --------------------------------------------------------
    #                  Save results as parquet
    # --------------------------------------------------------    
    # convert columns containing lists to datatype supported by parquet
    for col in ['standardized_name', 'name', 'highway', 'osmid']:
        final_edges[col] = final_edges[col].astype(str)
    # save data as parquet
    final_edges.to_parquet(f'../res/{place_name}.parquet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process road network data for a given city.')
    parser.add_argument('--place_name', type=str, default='Dublin, Ireland', help='City name for road network extraction.')
    # parser.add_argument('--simplify', type=bool, default=True, help='Whether to apply Douglas-Peucker simplification.')
    # parser.add_argument('--tolerance', type=float, default=100.0, help='Tolerance for Douglas-Peucker simplification in meters.')

    args = parser.parse_args()
    main(args.place_name, args.tolerance)