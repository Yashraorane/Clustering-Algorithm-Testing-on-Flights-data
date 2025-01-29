// function to toggle the read more content
$(document).ready(function() {
    $("#readMoreBtn").click(function() { //when the read more button is clicked
        $("#readMoreContent").slideToggle(); //toggle the read more content
    });
});

//Selecting the image and displaying it in the overlay
document.getElementById('imagecluster').addEventListener('click', function() {
    const img = this; //selecting the image
    if(img.style.transform === 'scale(1.5)') { //if the image is zoomed in
        img.style.transform = 'scale(1)'; //zoom out the image
        img.style.cursor = 'zoom-in'; //change the cursor to zoom in
    } else {
        img.style.transform = 'scale(1.5)'; //zoom in the image
        img.style.cursor = 'zoom-out'; //change the cursor to zoom out
    }});


function updateGraph() {
    //getting the values of the selected algorithm, year, season
    const algorithm = document.getElementById('algorithm').value;
    const calendaryear = document.getElementById('calendaryear').value;
    const season = document.getElementById('season').value;
    const zone = document.getElementById('region').value;
    const image = document.getElementById('imagecluster');
    const edaimg = document.getElementById('eda1-img');
    const edamap = document.getElementById('eda1-map');
    image.src = '../../static/images/YlWC.gif';  //downloaded from https://gifer.com/en/gifs/loader
    edaimg.src = '../../static/images/YlWC.gif';  //downloaded from https://gifer.com/en/gifs/loader

    edamap.innerHTML = '';

    //fetching the data from the server
    fetch('/updateGraph', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({   //converts the data into JSON format
            algorithm: algorithm,
            calendaryear: calendaryear,
            season: season,
            zone: zone
    })
    }).then(response => response.json()).then(data => { 
        image.src = data.Graph_data; //display the image
        edaimg.style.display = 'none';
        edamap.innerHTML = data.Map_data; //display the map
        edamap.style.display = 'block';
    });
}

