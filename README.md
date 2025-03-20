## OSM Road Assembler

### Introduction
The OpenStreetMap (OSM) street network is rich in information, but this level of detail can lead to significant fragmentation, with road segments sometimes as short as just a few meters. In the OSM graph, nodes are placed at every junction and intersection, breaking the road geometry into separate segments. Additionally, fragmentation can be further exacerbated by variations in one or more attributes along a route, such as changes in the maximum speed limit.

This excessive segmentation makes it difficult to conduct meaningful analyses of road characteristics. To address this, the network must be reorganized into a more coherent and unified representation. Such a representation can greatly benefit transport studies, traffic flow modeling, air quality assessments, and urban planning by providing a more realistic view of the road network structure.


### Data
The data for this project are retrieved from OpenStreetMap using the Python library *osmnx*.


### Project Structure
- `src/main.py`: main script that integrates all processing steps to generate the final network representation.
- `src/process_roads.py`: contains functions for processing and reconstructing the OSM street network.
- `src/utils.py`: provides general utility functions for network manipulation.
- `notebooks/example.ipynb`: a jupyter notebook showcasing examples and case studies demonstrating the proposed methodology.


### Main Steps

**Step 1: Roundabout Simplification**  
Roundabouts are often represented by multiple small segments, as each incoming road causes fragmentation of the geometry. We identified and reconstructed roundabouts into a unified geometry using the OSM *junction* attribute, which can be labeled as *roundabout* or *circular*. The reconstruction was performed using *shapely*'s *linemerge* and *unary union* operations, ensuring a coherent and complete representation of intersections.

**Step 2: Road Aggregation**  
Road segments were aggregated based on the OSM *name* attribute and their spatial contiguity or proximity. We standardized road names to ensure consistency and then grouped segments with the same standardized name. These groups were iteratively evaluated for spatial proximity, and contiguous segments were merged into a single geometry via unary union. A minimal tolerance distance was applied to merge separate lanes of the same road into a unified geometry.

**Step 3: Handling Roads with Alternative Names**  
Cases where road segments had multiple *name* attributes were treated similarly, with all alternative names considered during the aggregation process, ensuring a complete and accurate representation of each road.

**Step 4: Removal of Missing Names**  
Segments without an OSM *name* attribute were discarded. These segments are typically shorter and located in less critical areas of the network, such as internal residential zones. The absence of the *name* attribute may suggest lower relevance for the analysis.  

**Additional Road Attributes**  
Additional road attributes, including road type, number of lanes, speed limit, and the presence of tunnels and bridges, were preserved through careful aggregation, with each attribute handled according to its specific characteristics.


### Related Pubblication
This code is associated with the research paper titled _"Road network simplification to support air quality analysis"_, which has been accepted for the proceedings of [GISRUK 2025](https://gisruk.github.io/).


### Acknowledgment
This work has emanated from research conducted with the financial support of Taighde Éireann – Research Ireland under Grant number 18/CRT/6049. For the purpose of Open Access, the author has applied a CC BY public copyright licence.