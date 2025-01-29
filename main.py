import time #importing time module
import folium #for plotting the map
import base64 #for encoding the image to base64
import matplotlib #for plotting the graph
matplotlib.use('Agg') #setting the backend of matplotlib
from flask import render_template,Flask,request,jsonify #for rendering the html page
from io import BytesIO #for creating a byte stream
import geopandas as gpd #for reading the shapefile
import matplotlib.pyplot as plt #for plotting the graph
import warnings #for ignoring the warnings
warnings.filterwarnings("ignore")
from data import fine_tune_model #importing the fine tune model function
import networkx as nx #for creating the graph
from matplotlib.colors import ListedColormap #for color map



#Function from flask app
application = Flask(__name__, template_folder='page')

#Route operation for different pages in project
@application.route('/')
def dashboardPage():
    #returning the index page, rendering the index.html
    return render_template('index.html')

@application.route('/info')
def infoPage():
    #returning the about page, rendering the about.html
    return render_template('info.html')

#Function to get color map
def get_color_map(label):
    color_map = {
        0: 'darkpurple',
        1: 'indigo',
        2: 'blue',
        3: 'cyan',
        4: 'green',
        5: 'yellowgreen',
        6: 'yellow',
        7: 'lightyellow',
    }
    return color_map.get(label, 'gray')

#Function to get color map
def color_code():
    color = [
        '#440154', # dark purple
        '#482777', # indigo
        '#3F4A8A', # blue
        '#2A788E', # cyan
        '#21918C', # green
        '#27AD81', # yellowgreen
        '#7AD151', # yellow
        '#FDE725', # light yellow
    ]
    return ListedColormap(color)

#Function to plot the graph accoording to given algorithm
def plot_kmeans(dataset,clustering_labels):
    #reading the shapefile
    shapefile = '../src/ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp'
    states=gpd.read_file(shapefile)
    # print(states.info())
    #filtering the states of USA
    states= states[states['admin']=='United States of America']
    # Graph object to store the nodes and edges
    G=nx.Graph()
    #Building a libary of labels for each cluster formation, using the origin airport as the key
    lib_label ={row[1]['Origin_airport']: label for row, label in zip(dataset.iterrows(), clustering_labels)}

    #Adding nodes to the graph   the origin airport is in the lib_label
    for _,r in dataset.iterrows():
        #checking if the origin airport is in the lib_label
        origin = r['Origin_airport']
        #if the origin airport is in the lib_label, then add the node to the graph
        if origin in lib_label:
            #adding the node to the graph, with the position and label
            G.add_node(r['Origin_airport'], pos=(r['Org_airport_long'], r['Org_airport_lat']), label=lib_label[r['Origin_airport']])
    
    #Adding edges to the graph the origin airport is in the lib_label
    for _,r in dataset.iterrows():
        #checking if the origin airport is in the lib_label
        origin = r['Origin_airport']
        #if the origin airport is in the lib_label, then add the edge to the graph
        if origin in lib_label:
            #adding the edge to the graph, with the weight as the number of flights
            G.add_edge(r['Origin_airport'], r['Destination_airport'],  weight=r['Flights'])

    positions = nx.get_node_attributes(G, 'pos') #getting the positions of the nodes
    nodePositions = [node for node in G.nodes() if node in positions] #getting the nodes with positions
    colorPositions = [G.nodes[node].get('label',0) for node in nodePositions] #getting the labels of the nodes
    
    #Plotting Folium map
    map_centre= [39.8283, -98.5795] #center of USA
    m=folium.Map(location=map_centre, zoom_start=5, height=750, weight=1280 ) #creating a map
    #adding title to the map
    title='''<div style="position: fixed; width: 100%; height: 50px;font-size:24px; top:10px;left:50px; font-weight:bold; z-index:9999;
                        text-align: center;"> Kmeans Clustering on USA Flights</div>'''
    m.get_root().html.add_child(folium.Element(title))
    
    #adding markers to the map
    for node in nodePositions:
        label= G.nodes[node]['label']
        pos = G.nodes[node]['pos']
        #getting the information of the node
        node_info= dataset[dataset['Origin_airport']==node].iloc[0]
        text=f'''<b>Cluster: {label}</b><br>
        Origin: {node_info['Origin_airport']}<br>
        Destination: {node_info['Destination_airport']}<br>
        Passengers: {node_info['Passengers']}<br>
        Seats: {node_info['Seats']}<br>
        Flights: {node_info['Flights']}<br>
        Distance: {node_info['Distance']}'''

        folium.Marker(
            location=[pos[1],pos[0]],  #location of the marker
            popup=text,
            icon=folium.Icon(color=get_color_map(label), icon='info-sign')).add_to(m)
        
    map_html=m._repr_html_()


    #Plotting cluster graph
    color=color_code()
    figure1, ax1 = plt.subplots(figsize=(15, 10)) #creating a figure
    states.boundary.plot(ax=ax1,edgecolor='black', linewidth=2,)
    #Plotting the nodes on the graph, with the color of the nodes as the label
    scatterPlot= nx.draw_networkx_nodes(G, positions, cmap=color, nodelist=nodePositions, node_color=colorPositions, node_size=50, ax=ax1)
    cbar = plt.colorbar(scatterPlot, ax=ax1, orientation='horizontal', shrink=0.5, pad=0.01) #adding color bar
    cbar.set_label('Cluster tags')
    plt.figtext(0.5, 1, 'KMeans Clustering', va='top',fontsize=12, ha='center')
    
    ax1.set_aspect('equal','box') #setting the aspect of the graph
    plt.tight_layout() #setting the layout of the graph
    ax1.axis('off') #removing the axis of the graph

    #For additional visualization information of the data this has been used for entire project as a common logic. 
    #Hence, it is not specific to K-means so commented out after getting output from it.
    # figure,ax3 = plt.subplots(figsize=(15,10)) #creating a figure 
    # ax3.set_xlabel('Year')
    # ax3.set_ylabel('Number of Passengers', color = 'green')
    # ax3.plot(dataset['CalendarYear'],dataset['Passengers'],color='green', label='TotalPassengers')
    # ax3.tick_params(axis='y',labelcolor='green')
    # ax4=ax3.twinx()
    # ax3.set_ylabel('Number of Flights', color = 'blue')
    # ax3.plot(dataset['CalendarYear'],dataset['Passengers'],color='blue', label='TotalFlights')
    # ax3.tick_params(axis='y',labelcolor='blue')
    # plt.title('Variation of Flights and Passengers from 1990 to 2009')
    # figure.legend(loc='upper left')
    # plt.show()

    image= BytesIO() #creating a byte stream
    figure1.savefig(image, format='png', bbox_inches='tight',pad_inches=0) #saving the figure to the byte stream
    image.seek(0) #setting the pointer to the start of the stream
    plot1 = base64.b64encode(image.getvalue()).decode() #encoding the image to base64
    plt.close(figure1)

    return plot1, map_html

def plot_standard(dataset,clustering_labels):
    #reading the shapefile
    shapefile = 'C:/Users/Aanand/Desktop/COMP702/src/ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp'
    states=gpd.read_file(shapefile)
    # print(states.info())
    #filtering the states of USA
    states= states[states['admin']=="United States of America"]
    # states = states[states.is_valid]
    #creating a graph object to store the nodes and edges
    G=nx.Graph()
    #Building a libary of labels for each cluster formation, using the origin airport as the key
    lib_label ={row[1]['Origin_airport']: label for row, label in zip(dataset.iterrows(), clustering_labels)}

    #Adding nodes to the graph   the origin airport is in the lib_label
    for _,r in dataset.iterrows():
        #checking if the origin airport is in the lib_label
        origin = r['Origin_airport']
        #if the origin airport is in the lib_label, then add the node to the graph
        if origin in lib_label:
            #adding the node to the graph, with the position and label
            G.add_node(r['Origin_airport'], pos=(r['Org_airport_long'], r['Org_airport_lat']), label=lib_label[r['Origin_airport']])
    
    #Adding edges to the graph the origin airport is in the lib_label
    for _,r in dataset.iterrows():
        #checking if the origin airport is in the lib_label
        origin = r['Origin_airport']
        #if the origin airport is in the lib_label, then add the edge to the graph
        if origin in lib_label:
            #adding the edge to the graph, with the weight as the number of flights
            G.add_edge(r['Origin_airport'], r['Destination_airport'],  weight=r['Flights'])

    positions = nx.get_node_attributes(G, 'pos') #getting the positions of the nodes
    nodePositions = [node for node in G.nodes() if node in positions] #getting the nodes with positions
    colorPositions = [G.nodes[node].get('label',0) for node in nodePositions]  #getting the labels of the nodes

    #Plotting Folium map
    map_centre= [39.8283, -98.5795] #center of USA
    m=folium.Map(location=map_centre, zoom_start=5, height=750, weight=1280 ) #creating a map
    #adding title to the map
    title='''<div style="position: fixed; width: 100%; height: 50px;font-size:24px; top:10px;left:50px; font-weight:bold; z-index:9999;
                        text-align: center;"> Standard Clustering on USA Flights</div>'''
    m.get_root().html.add_child(folium.Element(title))
    
    #adding markers to the map
    for node in nodePositions:
        label= G.nodes[node]['label']
        pos = G.nodes[node]['pos']
        #getting the information of the node
        node_info= dataset[dataset['Origin_airport']==node].iloc[0]
        text=f'''<b>Cluster: {label}</b><br>
        Origin: {node_info['Origin_airport']}<br>
        Destination: {node_info['Destination_airport']}<br>
        Passengers: {node_info['Passengers']}<br>
        Seats: {node_info['Seats']}<br>
        Flights: {node_info['Flights']}<br>
        Distance: {node_info['Distance']}'''

        folium.Marker(
            location=[pos[1],pos[0]], #location of the marker
            popup=text,
            icon=folium.Icon(color=get_color_map(label), icon='info-sign')).add_to(m)
        
    map_html=m._repr_html_()

    #Plotting cluster graph
    color=color_code()
    figure, ax = plt.subplots(figsize=(15, 10)) #creating a figure
    border_color = 'black'

    states.boundary.plot(ax=ax, linewidth=2, edgecolor=border_color, facecolor='none') #plotting the boundary of the states
    #Plotting the nodes on the graph, with the color of the nodes as the label
    scatterPlot= nx.draw_networkx_nodes(G, positions, nodelist=nodePositions, node_color=colorPositions, node_size=50, cmap=color, ax=ax)
    cbar = plt.colorbar(scatterPlot, ax=ax, orientation='horizontal' ) #adding color bar
    cbar.set_label('Cluster Labels')
    plt.figtext(0.6, 0.07, 'Standardized Clustering', ha='center', va='top', fontsize=12)

    
    ax.set_aspect('equal','box')  #setting the aspect of the graph
    plt.tight_layout() #setting the layout of the graph
    ax.axis('off') #removing the axis of the graph

    image= BytesIO() #creating a byte stream
    figure.savefig(image, format="png", bbox_inches='tight',pad_inches=0) #saving the figure to the byte stream
    image.seek(0) #setting the pointer to the start of the stream
    plot = base64.b64encode(image.getvalue()).decode() #encoding the image to base64
    plt.close(figure)

    return plot, map_html

def plot_normalized(dataset,clustering_labels):
    #reading the shapefile
    shapefile = 'C:/Users/Aanand/Desktop/COMP702/src/ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp'
    states=gpd.read_file(shapefile)
    # print(states.info())
    states= states[states['admin']=="United States of America"]
    # states = states[states.is_valid]
    #creating a graph object to store the nodes and edges
    G=nx.Graph()
    #Building a libary of labels for each cluster formation, using the origin airport as the key
    lib_label ={row[1]['Origin_airport']: label for row, label in zip(dataset.iterrows(), clustering_labels)}

    #Adding nodes to the graph   the origin airport is in the lib_label
    for _,r in dataset.iterrows():
        #checking if the origin airport is in the lib_label
        origin = r['Origin_airport']
        #if the origin airport is in the lib_label, then add the node to the graph
        if origin in lib_label:
            #adding the node to the graph, with the position and label
            G.add_node(r['Origin_airport'], pos=(r['Org_airport_long'], r['Org_airport_lat']), label=lib_label[r['Origin_airport']])
    
    #Adding edges to the graph the origin airport is in the lib_label
    for _,r in dataset.iterrows():
        #checking if the origin airport is in the lib_label
        origin = r['Origin_airport']
        #if the origin airport is in the lib_label, then add the edge to the graph
        if origin in lib_label:
            #adding the edge to the graph, with the weight as the number of flights
            G.add_edge(r['Origin_airport'], r['Destination_airport'],  weight=r['Flights'])

    positions = nx.get_node_attributes(G, 'pos') #getting the positions of the nodes
    nodePositions = [node for node in G.nodes() if node in positions] #getting the nodes with positions
    colorPositions = [G.nodes[node].get('label',0) for node in nodePositions] #getting the labels of the nodes

    #Plotting Folium map which can help to visualize the data
    map_centre= [39.8283, -98.5795] #center of USA
    m=folium.Map(location=map_centre, zoom_start=5, height=750, weight=1280 ) #creating a map
    #adding title to the map
    title='''<div style="position: fixed; width: 100%; height: 50px;font-size:24px; top:10px;left:50px; font-weight:bold; z-index:9999;
                        text-align: center;"> Normalized Clustering on USA Flights</div>'''
    m.get_root().html.add_child(folium.Element(title))
    
    #Introduction of markers to the map
    for node in nodePositions: #iterating through the nodes
        label= G.nodes[node]['label'] #getting the label of the node
        pos = G.nodes[node]['pos'] #getting the position of the node
        node_info= dataset[dataset['Origin_airport']==node].iloc[0] #getting the information of the node
        #creating a text to display on the marker
        text=f'''<b>Cluster: {label}</b><br> 
        Origin: {node_info['Origin_airport']}<br>Destination: {node_info['Destination_airport']}<br>
        Passengers: {node_info['Passengers']}<br>Seats: {node_info['Seats']}<br>
        Flights: {node_info['Flights']}<br>Distance: {node_info['Distance']}'''
        #adding marker to the map
        folium.Marker(
            location=[pos[1],pos[0]], #location of the marker
            popup=text,
            icon=folium.Icon(color=get_color_map(label), icon='info-sign')).add_to(m) #color and icon of the marker
    #returning the html of the map   
    map_html=m._repr_html_()

    #Plotting cluster graph
    color=color_code()
    figure, ax = plt.subplots(figsize=(15, 10))
    border_color = 'black'
    
    states.boundary.plot(ax=ax, linewidth=2, edgecolor=border_color, facecolor='none')
    #Plotting the nodes on the graph, with the color of the nodes as the label
    scatterPlot= nx.draw_networkx_nodes(G, positions, nodelist=nodePositions, node_color=colorPositions, node_size=50, cmap=color, ax=ax)
    cbar = plt.colorbar(scatterPlot, ax=ax, orientation='horizontal' ) #adding color bar
    cbar.set_label('Cluster Labels') #setting the label of the color bar
    plt.figtext(0.6, 0.07, 'Normalized Clustering', ha='center', va='top', fontsize=12)
    
    
    ax.set_aspect('equal','box')  #setting the aspect of the graph
    plt.tight_layout() #setting the layout of the graph
    ax.axis('off')  #removing the axis of the graph

    image= BytesIO()  
    figure.savefig(image, format="png", bbox_inches='tight',pad_inches=0) #saving the figure to the byte stream
    image.seek(0) #setting the pointer to the start of the stream
    plot = base64.b64encode(image.getvalue()).decode() #encoding the image to base64
    plt.close(figure)

    return plot, map_html

def plot_dbscan(dataset,clustering_labels):
    #reading the shapefile
    shapefile = 'C:/Users/Aanand/Desktop/COMP702/src/ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp'
    states=gpd.read_file(shapefile)
    # print(states.info())
    #filtering the states of USA
    states= states[states['admin']=="United States of America"]
    # states = states[states.is_valid]
    #creating a graph object to store the nodes and edges
    G=nx.Graph()
    #Building a libary of labels for each cluster formation, using the origin airport as the key
    lib_label ={row[1]['Origin_airport']: label for row, label in zip(dataset.iterrows(), clustering_labels)}

    #Adding nodes to the graph   the origin airport is in the lib_label
    for _,r in dataset.iterrows():
        #checking if the origin airport is in the lib_label
        origin = r['Origin_airport']
        #if the origin airport is in the lib_label, then add the node to the graph
        if origin in lib_label:
            #adding the node to the graph, with the position and label
            G.add_node(r['Origin_airport'], pos=(r['Org_airport_long'], r['Org_airport_lat']), label=lib_label[r['Origin_airport']])
    
    #Adding edges to the graph the origin airport is in the lib_label
    for _,r in dataset.iterrows():
        #checking if the origin airport is in the lib_label
        origin = r['Origin_airport']
        #if the origin airport is in the lib_label, then add the edge to the graph
        if origin in lib_label:
            #adding the edge to the graph, with the weight as the number of flights
            G.add_edge(r['Origin_airport'], r['Destination_airport'],  weight=r['Flights'])

    positions = nx.get_node_attributes(G, 'pos') #getting the positions of the nodes
    nodePositions = [node for node in G.nodes() if node in positions] #getting the nodes with positions
    colorPositions = [G.nodes[node].get('label',0) for node in nodePositions] #getting the labels of the nodes

    #Plotting Folium map
    map_centre= [39.8283, -98.5795] #center of USA
    m=folium.Map(location=map_centre, zoom_start=5, height=750, weight=1280 ) #creating a map
    #adding title to the map
    title='''<div style="position: fixed; width: 100%; height: 50px;font-size:24px; top:10px;left:50px; font-weight:bold; z-index:9999;
                        text-align: center;"> DBscan Clustering on USA Flights</div>'''
    m.get_root().html.add_child(folium.Element(title))
    
    #adding markers to the map
    for node in nodePositions:
        label= G.nodes[node]['label']
        pos = G.nodes[node]['pos']
        #getting the information of the node
        node_info= dataset[dataset['Origin_airport']==node].iloc[0]
        text=f'''<b>Cluster: {label}</b><br>
        Origin: {node_info['Origin_airport']}<br>
        Destination: {node_info['Destination_airport']}<br>
        Passengers: {node_info['Passengers']}<br>
        Seats: {node_info['Seats']}<br>
        Flights: {node_info['Flights']}<br>
        Distance: {node_info['Distance']}'''

        folium.Marker(
            location=[pos[1],pos[0]],  #location of the marker
            popup=text,
            icon=folium.Icon(color=get_color_map(label), icon='info-sign')).add_to(m)
        
    map_html=m._repr_html_()

    #Plotting cluster graph
    color=color_code()
    figure, ax = plt.subplots(figsize=(15, 10))
    border_color = 'black'
    
    states.boundary.plot(ax=ax, linewidth=2, edgecolor=border_color, facecolor='none')
    #Plotting the nodes on the graph, with the color of the nodes as the label
    scatterPlot= nx.draw_networkx_nodes(G, positions, nodelist=nodePositions, node_color=colorPositions, node_size=50, cmap='viridis', ax=ax)
    cbar = plt.colorbar(scatterPlot, ax=ax, orientation='horizontal' ) #adding color bar
    cbar.set_label('Cluster Labels')
    plt.figtext(0.6, 0.07, 'DBScan Clustering', ha='center', va='top', fontsize=12)
        
    ax.set_aspect('equal','box')   #setting the aspect of the graph
    plt.tight_layout() #setting the layout of the graph
    ax.axis('off') #removing the axis of the graph

    image= BytesIO() #creating a byte stream
    figure.savefig(image, format="png", bbox_inches='tight',pad_inches=0) #saving the figure to the byte stream
    image.seek(0) #setting the pointer to the start of the stream
    plot = base64.b64encode(image.getvalue()).decode() #encoding the image to base64
    plt.close(figure)

    return plot, map_html

def plot_kernel(dataset,clustering_labels):
    #reading the shapefile
    shapefile = 'C:/Users/Aanand/Desktop/COMP702/src/ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp'
    states=gpd.read_file(shapefile)
    # print(states.info())
    states= states[states['admin']=="United States of America"]
    # states = states[states.is_valid]
    G=nx.Graph()
    #Building a libary of labels for each cluster formation using the origin airport as the key
    lib_label ={row[1]['Origin_airport']: label for row, label in zip(dataset.iterrows(), clustering_labels)}
    #checking the origin airport is in the lib_label
    crop= dataset[dataset['Origin_airport'].isin(lib_label.keys())]

    #Adding nodes to the graph   the origin airport is in the lib_label
    for _,r in crop.iterrows():
        #checking if the origin airport is in the lib_label
        origin = r['Origin_airport']
        #if the origin airport is in the lib_label, then add the node to the graph
        if origin in lib_label:
            #adding the node to the graph, with the position and label
            G.add_node(r['Origin_airport'], pos=(r['Org_airport_long'], r['Org_airport_lat']), label=lib_label[r['Origin_airport']])
    #Adding edges to the graph the origin airport is in the lib_label
    for _,r in crop.iterrows():
        #checking if the origin airport is in the lib_label
        origin = r['Origin_airport']
        #if the origin airport is in the lib_label, then add the edge to the graph
        if origin in lib_label:
            #adding the edge to the graph, with the weight as the number of flights
            G.add_edge(r['Origin_airport'], r['Destination_airport'],  weight=r['Flights'])

    positions = nx.get_node_attributes(G, 'pos') #getting the positions of the nodes
    nodePositions = [node for node in G.nodes() if node in positions] #getting the nodes with positions
    colorPositions = [G.nodes[node].get('label',0) for node in nodePositions] #getting the labels of the nodes

    #Plotting Folium map
    map_centre= [39.8283, -98.5795] #center of USA
    m=folium.Map(location=map_centre, zoom_start=5, height=750, weight=1280 )
    #adding title to the map
    title='''<div style="position: fixed; width: 100%; height: 50px;font-size:24px; top:10px;left:50px; font-weight:bold; z-index:9999;
                        text-align: center;"> Kernel Clustering on USA Flights</div>'''
    m.get_root().html.add_child(folium.Element(title))
    
    #adding markers to the map
    for node in nodePositions:
        label= G.nodes[node]['label']
        pos = G.nodes[node]['pos']
        #getting the information of the node
        node_info= dataset[dataset['Origin_airport']==node].iloc[0]
        text=f'''<b>Cluster: {label}</b><br>
        Origin: {node_info['Origin_airport']}<br>
        Destination: {node_info['Destination_airport']}<br>
        Passengers: {node_info['Passengers']}<br>
        Seats: {node_info['Seats']}<br>
        Flights: {node_info['Flights']}<br>
        Distance: {node_info['Distance']}'''

        folium.Marker(
            location=[pos[1],pos[0]], #location of the marker
            popup=text,
            icon=folium.Icon(color=get_color_map(label), icon='info-sign')).add_to(m)
        
    map_html=m._repr_html_()

    #Plotting cluster graph
    color=color_code()
    figure, ax = plt.subplots(figsize=(15, 10))
    border_color = 'black'

    states.boundary.plot(ax=ax, linewidth=2, edgecolor=border_color, facecolor='none')
    #Plotting the nodes on the graph, with the color of the nodes as the label
    scatterPlot= nx.draw_networkx_nodes(G, positions, nodelist=nodePositions, node_color=colorPositions, node_size=50, cmap=color, ax=ax)
    # nx.draw_networkx_edges(G, positions, nodelist=nodePositions, node_color=colorPositions, alpha=0.5, ax=ax)
    cbar = plt.colorbar(scatterPlot, ax=ax, orientation='horizontal') #adding color bar
    cbar.set_label('Cluster Labels') #setting the label of the color bar
    plt.figtext(0.6, 0.07, 'Kernel Clustering', ha='center', va='top', fontsize=12)
    
    
    ax.set_aspect('equal','box') #setting the aspect of the graph
    plt.tight_layout()  #setting the layout of the graph
    ax.axis('off') #removing the axis of the graph

    image= BytesIO() #creating a byte stream
    figure.savefig(image, format="png", bbox_inches='tight',pad_inches=0) #saving the figure to the byte stream
    image.seek(0) #setting the pointer to the start of the stream
    plot = base64.b64encode(image.getvalue()).decode() #encoding the image to base64
    plt.close(figure)
    return plot, map_html

def plot_polynomial(dataset,clustering_labels):
    #reading the shapefile
    shapefile = 'C:/Users/Aanand/Desktop/COMP702/src/ne_110m_admin_1_states_provinces/ne_110m_admin_1_states_provinces.shp'
    states=gpd.read_file(shapefile)
    # print(states.info())
    states= states[states['admin']=="United States of America"]
    # states = states[states.is_valid]
    #creating a graph object to store the nodes and edges
    G=nx.Graph()
    #Building a libary of labels for each cluster formation using the origin airport as the key
    lib_label ={row[1]['Origin_airport']: label for row, label in zip(dataset.iterrows(), clustering_labels)}
    dataset[dataset['Origin_airport'].isin(lib_label.keys())]

    #Adding nodes to the graph the origin airport is in the lib_label
    for _,r in dataset.iterrows():
        origin = r['Origin_airport'] #getting the origin airport
        if origin in lib_label: #checking if the origin airport is in the lib_label
            #adding the node to the graph, with the position and label
            G.add_node(r['Origin_airport'], pos=(r['Org_airport_long'], r['Org_airport_lat']), label=lib_label[r['Origin_airport']])    
    
    #Adding edges to the graph the origin airport is in the lib_label
    for _,r in dataset.iterrows():
        origin = r['Origin_airport'] #getting the origin airport
        if origin in lib_label: #checking if the origin airport is in the lib_label
            #adding the edge to the graph, with the weight as the number of flights
            G.add_edge(r['Origin_airport'], r['Destination_airport'],  weight=r['Flights'])

    positions = nx.get_node_attributes(G, 'pos') #getting the positions of the nodes
    nodePositions = [node for node in G.nodes() if node in positions] #getting the nodes with positions
    colorPositions = [G.nodes[node].get('label',0) for node in nodePositions] #getting the labels of the nodes

    #Plotting Folium map
    map_centre= [39.8283, -98.5795] #center of USA
    m=folium.Map(location=map_centre, zoom_start=5, height=750, weight=1280 ) #creating a map
    #adding title to the map
    title='''<div style="position: fixed; width: 100%; height: 50px;font-size:24px; top:10px;left:50px; font-weight:bold; z-index:9999;
                        text-align: center;"> Polynomial Clustering on USA Flights</div>'''
    m.get_root().html.add_child(folium.Element(title))
    
    #adding markers to the map
    for node in nodePositions:
        label= G.nodes[node]['label']
        pos = G.nodes[node]['pos']
        #getting the information of the node
        node_info= dataset[dataset['Origin_airport']==node].iloc[0]
        text=f'''<b>Cluster: {label}</b><br>
        Origin: {node_info['Origin_airport']}<br>
        Destination: {node_info['Destination_airport']}<br>
        Passengers: {node_info['Passengers']}<br>
        Seats: {node_info['Seats']}<br>
        Flights: {node_info['Flights']}<br>
        Distance: {node_info['Distance']}'''

        folium.Marker(
            location=[pos[1],pos[0]],  #location of the marker
            popup=text,
            icon=folium.Icon(color=get_color_map(label), icon='info-sign')).add_to(m)
        
    map_html=m._repr_html_()

    #Plotting cluster graph
    color=color_code()
    figure, ax = plt.subplots(figsize=(15, 10))
    border_color = 'black'
    
    states.boundary.plot(ax=ax, linewidth=2, edgecolor=border_color, facecolor='none')
    #Plotting the nodes on the graph, with the color of the nodes as the label
    scatterPlot= nx.draw_networkx_nodes(G, positions, nodelist=nodePositions, node_color=colorPositions, node_size=50, cmap=color, ax=ax)
    cbar = plt.colorbar(scatterPlot, ax=ax, orientation='horizontal' ) #adding color bar
    cbar.set_label('Cluster Labels')
    plt.figtext(0.6, 0.07, 'Polynomial Clustering', ha='center', va='bottom', fontsize=12)
    
    
    ax.set_aspect('equal','box') #setting the aspect of the graph
    plt.tight_layout() #setting the layout of the graph
    ax.axis('off') #removing the axis of the graph

    image= BytesIO() #creating a byte stream
    figure.savefig(image, format="png", bbox_inches='tight',pad_inches=0) #saving the figure to the byte stream
    image.seek(0) #setting the pointer to the start of the stream
    plot = base64.b64encode(image.getvalue()).decode() #encoding the image to base64
    plt.close(figure)
    
    return plot, map_html



#A dictionary to call specific plotting function of algorithm
PLOT_DICTIONARY = {
    'kmeans': plot_kmeans,'dbscan': plot_dbscan,'standard': plot_standard,
    'normalized': plot_normalized,'kernel': plot_kernel,'polynomial': plot_polynomial,
}

@application.route('/updateGraph',methods=['POST'])
def updateGraph():
    
    #getting the json data from the request
    dataSet=request.json
    season=dataSet.get("season")
    zone = dataSet.get("zone")
    calendaryear=dataSet.get("calendaryear")
    algorithm=dataSet.get("algorithm")
    #printing the data
    print(zone,calendaryear,season,algorithm)

    #execution time
    start=time.time()
    #calling the fine tune model function from model.py
    data,result_pca,clustering_labels= fine_tune_model('../src/data/flights.csv',zone,calendaryear,season,algorithm)
    #calling the plotting function from PLOT_DICTIONARY
    function_for_plot = PLOT_DICTIONARY.get(algorithm)
    # print(function_for_plot)

    #plotting the graph
    Graph, Map = function_for_plot(data,clustering_labels)
    #returning the json response
    response={'Graph_data': f"data:image/png;base64,{Graph}",
               'Map_data': Map,
            }
    
    #execution time
    end= time.time()
    print(f"Execution time: {end-start}")
    return jsonify(response)

if __name__ == '__main__':
    #running the application
    application.run(port= 5000,debug=True)

