var audioManager = new AudioManager();
var metronome = new Metronome(audioManager);
let sliderInstance = null;
// metronome.play();

function send_request_get_response(url, return_function){
  var xhttp = new XMLHttpRequest();
  xhttp.open("GET", url, true);
  xhttp.setRequestHeader("Content-Type", "application/json");
  xhttp.send();
  xhttp.onreadystatechange = function() {
      if (this.readyState == 4 && this.status == 200) {
         var jsonObj = JSON.parse(xhttp.response);
         return_function( jsonObj );
      }
  };
}

function replaceAllSpaces(a) {
  var replaced_a = a.replaceAll(" ", "_");

  if (replaced_a.charAt(0) == "_") {
    replaced_a = replaced_a.substring(1);
  }
  if (replaced_a.charAt(replaced_a.length - 1) == "_") {
    replaced_a = replaced_a.substring(0, replaced_a.length - 1);
  }
  return replaced_a
}

// Request for metadata

function apply_songsmetadata_list(resp){
  composer_of_composition = Object.values(resp)[0];
  form_of_composition = Object.values(resp)[1];
  genre_of_composition = Object.values(resp)[2];
  harmonic_style_of_composition = Object.values(resp)[3];
  nameslist =  Object.values(resp)[4];
  tonality_of_composition = Object.values(resp)[5];
  year_of_composition = Object.values(resp)[6];
}
send_request_get_response( location.href + 'songsmetadata', apply_songsmetadata_list );

// function apply_centroid_infos(resp){
//   centroid = Object.values(resp)[0];
// }
// send_request_get_response( location.href + 'centroid_data', apply_centroid_infos );

// Request for data coordinates & call for plotly function

function mask_visualization_data(response) {
  
    initialise_plotly_chart(response, nameslist, year_of_composition, genre_of_composition, tonality_of_composition, composer_of_composition, form_of_composition, harmonic_style_of_composition);

}
function run_mask_visualization_data(){
  send_request_get_response(location.href + 'mask_visualization_data', mask_visualization_data);
}
run_mask_visualization_data()


// Main plotly code
function initialise_plotly_chart(response, nameslist, years, genre, tonality, composer, form, harmonic_style) {
  
    console.log(response)
    nameslistNoUnderbar = [];

    for (i=0; i<nameslist.length; i++) {
        nameslistNoUnderbar.push(nameslist[i].replaceAll("_", " "));
    }

    function RGBAToHex(r, g, b, a) {
      r = Math.floor(r * 255);
      g = Math.floor(g * 255);
      b = Math.floor(b * 255);
      a = Math.floor(a * 255);
    
      var rHex = ("0" + r.toString(16)).slice(-2);
      var gHex = ("0" + g.toString(16)).slice(-2);
      var bHex = ("0" + b.toString(16)).slice(-2);
      var aHex = ("0" + a.toString(16)).slice(-2);
    
      return "#" + rHex + gHex + bHex + aHex;
    }
    
    x_axis_post_encoding = [];
    y_axis_post_encoding = [];
    colors_post_encoding = [];

    x_axis_form = [];
    y_axis_form = [];
    colors_form = [];

    x_axis_style = [];
    y_axis_style = [];
    colors_style = [];

    // x_axis_style_centroid = [];
    // y_axis_style_centroid = [];
    // colors_style_centroid = [];

    x_axis_year = [];
    y_axis_year = [];
    colors_year = [];

    x_axis_tonality = [];
    y_axis_tonality = [];
    colors_tonality = [];

    x_axis_composer = [];
    y_axis_composer = [];
    colors_composer = [];

    x_axis_genre = [];
    y_axis_genre = [];
    colors_genre = [];
    
    symbol = [];
    if (typeof(response.form) != "undefined") {
      
      for (i=0; i<response.form.coordinates.length; i++) {
        x_axis_form.push(response.form.coordinates[i][0]);
        y_axis_form.push(response.form.coordinates[i][1]);
        colors_form.push(RGBAToHex(response.form.colors[i][0],response.form.colors[i][1],response.form.colors[i][2],response.form.colors[i][3]));
      }

      for (i=0; i<response.style.coordinates.length; i++) {
        x_axis_style.push(response.style.coordinates[i][0]);
        y_axis_style.push(response.style.coordinates[i][1]);
        colors_style.push(RGBAToHex(response.style.colors[i][0],response.style.colors[i][1],response.style.colors[i][2],response.style.colors[i][3]));
      }

      for (i=0; i<response.post_encoding.coordinates.length; i++) {
        x_axis_post_encoding.push(response.post_encoding.coordinates[i][0]);
        y_axis_post_encoding.push(response.post_encoding.coordinates[i][1]);
        colors_post_encoding.push(RGBAToHex(response.post_encoding.colors[i][0],response.post_encoding.colors[i][1],response.post_encoding.colors[i][2],response.post_encoding.colors[i][3]));
      }

      // for (i=0; i<centroid.coordinates.length; i++) {
      //   x_axis_style_centroid.push(centroid.coordinates[i][0]);
      //   y_axis_style_centroid.push(centroid.coordinates[i][1]);
      //   colors_style_centroid.push(RGBAToHex(centroid.colors[i][0],centroid.colors[i][1],centroid.colors[i][2],centroid.colors[i][3]));
      // }
      
      for (i=0; i<response.year.coordinates.length; i++) {
        x_axis_year.push(response.year.coordinates[i][0]);
        y_axis_year.push(response.year.coordinates[i][1]);
        colors_year.push(RGBAToHex(response.year.colors[i][0],response.year.colors[i][1],response.year.colors[i][2],response.year.colors[i][3]));
        
      }

      for (i=0; i<response.tonality.coordinates.length; i++) {
        x_axis_tonality.push(response.tonality.coordinates[i][0]);
        y_axis_tonality.push(response.tonality.coordinates[i][1]);
        colors_tonality.push(RGBAToHex(response.tonality.colors[i][0],response.tonality.colors[i][1],response.tonality.colors[i][2],response.tonality.colors[i][3]));
        
      }

      for (i=0; i<response.composer.coordinates.length; i++) {
        x_axis_composer.push(response.composer.coordinates[i][0]);
        y_axis_composer.push(response.composer.coordinates[i][1]);
        colors_composer.push(RGBAToHex(response.composer.colors[i][0],response.composer.colors[i][1],response.composer.colors[i][2],response.composer.colors[i][3]));
        
      }

      for (i=0; i<response.genre.coordinates.length; i++) {
        x_axis_genre.push(response.genre.coordinates[i][0]);
        y_axis_genre.push(response.genre.coordinates[i][1]);
        colors_genre.push(RGBAToHex(response.genre.colors[i][0],response.genre.colors[i][1],response.genre.colors[i][2],response.genre.colors[i][3]));
        
      }
      //console.log(x_axis_style_centroid,y_axis_style_centroid,colors_style_centroid);
    } 
   
    customdata = [];
    for (i=0; i<Object.keys(nameslist).length; i++) {
      customdata.push("Genre style: "+genre[i]+"<br>Tonality: "+tonality[i]+"<br>Composer: "+composer[i]+"</br>"+"Form: "+form[i]+"</br>"+"Harmonic Style: "+harmonic_style[i]+"</br>"+"Year: "+parseInt(years[i])+"</br>");
    }

    var myPlot = d3.csv('https://raw.githubusercontent.com/plotly/datasets/master/3d-scatter.csv', function(err, rows) {


    var defaultOpacity = Array(x_axis_form.length).fill(0.8);
    //console.log('defaultOpacity:', defaultOpacity);

    var myHovertemplate = '<br>%{text}</br>' + '<b>%{customdata}</b>' + "<extra></extra>";
    var hoverTemplateArray = Array(1048).fill(
      myHovertemplate
    );
    
    //console.log(hoverTemplateArray);

  // Create initial trace
  var trace1 = {
    x: x_axis_post_encoding,
    y: y_axis_post_encoding,
    mode: 'markers',
    marker: {
      opacity: defaultOpacity,
      size: 4,
      color: colors_post_encoding,
    },
    hovertemplate: hoverTemplateArray,
    customdata: customdata,
    text: nameslistNoUnderbar,
    type: 'scatter',
    name: "Songs' data"
  };

  // centroidTrace = {
  //   x: x_axis_style_centroid,
  //   y: y_axis_style_centroid,
  //   mode: 'markers',
  //   marker: {
  //     color: colors_style_centroid,
  //     size: 10,
  //     symbol: 'cross'
  //   },
  //   customdata: customdata,
  //   type: 'scatter',
  //   name: "Centroids"
  // };

  // Create initial data array
  var initialData = [trace1];

  // Create layout

  // TODO: Function that defines dynamically min & max 
  var min_x = -70;
  var max_x = 70;
  var min_y = -70;
  var max_y = 70;
  var total_x_range = max_x - min_x;
  var total_y_range = max_y - min_y;
  var x_dimensions = [min_x, max_x];
  var y_dimensions = [min_y, max_y];


  var layout = {
    //plot_bgcolor:"#FFF6F4",
    scene: {
      aspectmode: 'manual',
      aspectratio: { x: 1, y: 1 },
      xaxis: {
        autorange: true,
        showgrid: true,
        zeroline: true,
        showticklabels: false,
      },
      yaxis: {
        autorange: true,        
        showgrid: true,
        zeroline: true,
        showticklabels: false
      }
    },
    xaxis: {
      range: x_dimensions,
      ticks: '',
      showticklabels: false,
      showgrid: true,
      zeroline: true,
      showline: false
    },
    yaxis: {
      range: y_dimensions,
      ticks: '',
      showticklabels: false,
      showgrid: true,
      zeroline: true,
      showline: false  
    }, 
    showlegend: false
  };
  
  // Create dropdown menu options
  var dropdownOptions = [
    { text: 'Post Encoding', value: 'post_encoding_selector' },
    { text: 'Form Style', value: 'form_selector' },
    { text: 'Harmonic Style', value: 'harmony_selector' },
    { text: 'Year of Composition', value: 'year_selector' },
    { text: 'Tonality', value: 'tonality_selector' },
    { text: 'Composer', value: 'composer_selector' },
    { text: 'Genre Style', value: 'genre_selector' }
    // Add more options as needed
  ];

  // Create dropdown menu
  var dropdown = document.createElement('select');
  dropdown.id = 'traceDropdown';
  dropdownOptions.forEach(function(option) {
    var dropdownOption = document.createElement('option');
    dropdownOption.value = option.value;
    dropdownOption.text = option.text;
    dropdown.appendChild(dropdownOption);
  });

  // Append dropdown menu to the document
  var dropdownContainer = document.getElementById('dropdownContainer');
  dropdownContainer.appendChild(dropdown);

  // Create dropdown menu options
  var dropdownOptions_color_schemes = [
    { text: 'No colors', value: 'post_encoding_color_selector' },
    { text: 'Form', value: 'form_color_selector' },
    { text: 'Harmony', value: 'harmony_color_selector' },
    { text: 'Year', value: 'year_color_selector' },
    { text: 'Tonality', value: 'tonality_color_selector' },
    { text: 'Composer', value: 'composer_color_selector' },
    { text: 'Genre', value: 'genre_color_selector' }
    // Add more options as needed
  ];

  // Create dropdown menu
  var dropdown_color_schemes = document.createElement('select');
  dropdown_color_schemes.id = 'traceDropdown-color_schemes';
  dropdownOptions_color_schemes.forEach(function(option) {
    var dropdownOption_color_schemes = document.createElement('option');
    dropdownOption_color_schemes.value = option.value;
    dropdownOption_color_schemes.text = option.text;
    dropdown_color_schemes.appendChild(dropdownOption_color_schemes);
  });

  // Append dropdown menu to the document
  var dropdownContainer_color_schemes = document.getElementById('dropdownContainer-color_schemes');
  dropdownContainer_color_schemes.appendChild(dropdown_color_schemes);
  
  // Function to update the trace
  function updateTrace(selectedTrace) {
    var newTrace;
    document.getElementById('traceDropdown-color_schemes').value = "form_color_selector"

    // Define the properties of the selected trace based on the dropdown value
    if (selectedTrace === 'post_encoding_selector') {

      document.getElementById('traceDropdown-color_schemes').value = "post_encoding_color_selector"
      
      newTrace = {
        x: x_axis_post_encoding,
        y: y_axis_post_encoding,
        mode: 'markers',
        marker: {
          color: colors_post_encoding,
        },
        hovertemplate: '<br>%{text}</br>' +
          '<b>%{customdata}</b>' +
          "<extra></extra>",
        customdata: customdata,
        text: nameslistNoUnderbar,
        type: 'scatter'
      };
    } else if (selectedTrace === 'form_selector') {

      document.getElementById('traceDropdown-color_schemes').value = "form_color_selector"
      
      newTrace = {
        x: x_axis_form,
        y: y_axis_form,
        mode: 'markers',
        marker: {
          color: colors_form,
        },
        hovertemplate: '<br>%{text}</br>' +
          '<b>%{customdata}</b>' +
          "<extra></extra>",
        customdata: customdata,
        text: nameslistNoUnderbar,
        type: 'scatter'
      };
    } else if (selectedTrace === 'harmony_selector') {

      document.getElementById('traceDropdown-color_schemes').value = "harmony_color_selector"

      // Define the properties of the second trace
      newTrace = {
        x: x_axis_style,
        y: y_axis_style,
        mode: 'markers',
        marker: {
          color: colors_style,
        },
        hovertemplate: '<br>%{text}</br>' +
          '<b>%{customdata}</b>' +
          "<extra></extra>",
        customdata: customdata,
        text: nameslistNoUnderbar,
        type: 'scatter'
      };

      // centroidTrace = {
      //   x: x_axis_style_centroid,
      //   y: y_axis_style_centroid,
      //   mode: 'markers',
      //   marker: {
      //     color: colors_style_centroid,
      //     size: 30
      //   },
      //   hovertemplate: '<br>%{text}</br>' +
      //     '<b>%{customdata}</b>' +
      //     "<extra></extra>",
      //   customdata: customdata,
      //   text: nameslistNoUnderbar,
      //   type: 'scatter'
      // };
      
    } else if (selectedTrace === 'year_selector') {

      document.getElementById('traceDropdown-color_schemes').value = "year_color_selector"
      // Define the properties of the third trace
      newTrace = {
        x: x_axis_year,
        y: y_axis_year,
        mode: 'markers',
        marker: {
          color: colors_year,
        },
        hovertemplate: '<br>%{text}</br>' +
          '<b>%{customdata}</b>' +
          "<extra></extra>",
        customdata: customdata,
        text: nameslistNoUnderbar,
        type: 'scatter'
      };
    } else if (selectedTrace === 'tonality_selector') {

      document.getElementById('traceDropdown-color_schemes').value = "tonality_color_selector"
      // Define the properties of the third trace
      newTrace = {
        x: x_axis_tonality,
        y: y_axis_tonality,
        mode: 'markers',
        marker: {
          color: colors_tonality,
        },
        hovertemplate: '<br>%{text}</br>' +
          '<b>%{customdata}</b>' +
          "<extra></extra>",
        customdata: customdata,
        text: nameslistNoUnderbar,
        type: 'scatter'
      };
    } else if (selectedTrace === 'composer_selector') {

      document.getElementById('traceDropdown-color_schemes').value = "composer_color_selector"
      // Define the properties of the third trace
      newTrace = {
        x: x_axis_composer,
        y: y_axis_composer,
        mode: 'markers',
        marker: {
          color: colors_composer,
        },
        hovertemplate: '<br>%{text}</br>' +
          '<b>%{customdata}</b>' +
          "<extra></extra>",
        customdata: customdata,
        text: nameslistNoUnderbar,
        type: 'scatter'
      };
    } else if (selectedTrace === 'genre_selector') {

      document.getElementById('traceDropdown-color_schemes').value = "genre_color_selector"
      // Define the properties of the third trace
      newTrace = {
        x: x_axis_genre,
        y: y_axis_genre,
        mode: 'markers',
        marker: {
          color: colors_genre,
        },
        hovertemplate: '<br>%{text}</br>' +
          '<b>%{customdata}</b>' +
          "<extra></extra>",
        customdata: customdata,
        text: nameslistNoUnderbar,
        type: 'scatter'
      };
    }

    // Update the plot with the new trace
    Plotly.animate('myDiv', {
      data: [newTrace]
    }, {
      transition: {
        duration: 500,
        easing: 'linear'
      },
      frame: {
        duration: 500,
        redraw: true
      }
    });
  }

  // Dropdown change event handler
  dropdown.addEventListener('change', function() {
    var selectedTrace = dropdown.value;
    updateTrace(selectedTrace);
    //updatedropdownColorSchemes(selectedTrace);
  });


  // Function to update the trace
  function updatedropdownColorSchemes(selectedColorScheme) {
    sizes = initialiseSizes();
    //Plotly.restyle('myDiv', update);
    if (selectedColorScheme == "post_encoding_color_selector") {
      Plotly.restyle(myDiv, {marker: {color: colors_post_encoding}});
    } else if (selectedColorScheme == "form_color_selector") {
      Plotly.restyle(myDiv, {marker: {color: colors_form}});
    } else if (selectedColorScheme == "harmony_color_selector") {
      Plotly.restyle(myDiv, {marker: {color: colors_style}});
    }	else if (selectedColorScheme == "year_color_selector") {
      Plotly.restyle(myDiv, {marker: {color: colors_year}});
    }	else if (selectedColorScheme == "tonality_color_selector") {
      Plotly.restyle(myDiv, {marker: {color: colors_tonality}});
    } else if (selectedColorScheme == "composer_color_selector") {
      Plotly.restyle(myDiv, {marker: {color: colors_composer}});
    } else if (selectedColorScheme == "genre_color_selector") {
      Plotly.restyle(myDiv, {marker: {color: colors_genre}});
    }
  }
    

  // Dropdown change event handler
  dropdown_color_schemes.addEventListener('change', function() {
    var selectedColorScheme = dropdown_color_schemes.value;
    //updateTrace(selectedTrace);
    updateTrace_opacity(which_ui="dropdown", 0,0,selectedColorScheme)
    //updatedropdownColorSchemes(selectedColorScheme);
  });


  // Create initial plot
  Plotly.newPlot('myDiv', initialData, layout).then(gd => {
    gd.on('plotly_click', function(data) {
      console.log(layout);
      for (var i = 0; i < data.points.length; i++) {
        var selectedsong = data.points[i].text;
        Swal.fire({
          toast: true,
          position: 'top-end',
          showConfirmButton: false,
          icon: 'success',
          html: '<p>Selected song: <br>' + selectedsong + '</p>',
          width: '20rem',
          height: '1rem',
          timer: 1500
        });
        addEvent(selectedsong, r, h);
      }
    });
  });

  // var myPlot = document.getElementById('myDiv');
  // var myPlot = document.getElementById('myDiv');
  // myPlot.on('plotly_hover', function(data){
  //   data.points.forEach(function(point) {
  //     var markerOpacity = point.data.marker.opacity[point.pointNumber];
  //     console.log(markerOpacity,point)
  //     if (markerOpacity === 0) {
  //       console.log("BEFORE:",point.data.hovertemplate)
  //       point.data.hovertemplate = '';
  //       console.log("AFTER:",point.data.hovertemplate)
  //     }
  //   });
  // });


  var graphDiv = document.getElementById('myDiv');

  year = years;
  
  min_year = Math.min(...year);
  max_year = Math.max(...year);

  var year_sorted = year.slice().sort();

  var valuesSlider = document.getElementById('values-slider');

  var year_sorted = [...new Set(year_sorted)];

  var stringArray = year_sorted;

  var valuesForSlider = stringArray.map(function(num) {
    return parseInt(num, 10);
  });
  //console.log(" valuesForSlider:", valuesForSlider); 

  var format = {
    to: function(value) {
      
      return valuesForSlider[Math.round(value)];
    },
    from: function (value) {
      return valuesForSlider.indexOf(Number(value));
    }
  };

  noUiSlider.create(valuesSlider, {
    start: [0, 5],
    // A linear range from 0 to 15 (16 values)

    range: { 
      'min': [  0 ],
      '10%': [  5 ],
      '20%': [  ((max_year - min_year)/2) - 39 ],
      '30%': [  ((max_year - min_year)/2) - 29 ],
      '40%': [  ((max_year - min_year)/2) - 19 ],
      '50%': [  ((max_year - min_year)/2) - 9 ],
      '60%': [  ((max_year - min_year)/2) + 1 ],
      '70%': [  ((max_year - min_year)/2) + 11 ],
      '80%': [  ((max_year - min_year)/2) + 21 ],
      '90%': [  ((max_year - min_year)/2) + 27 ],
      'max': [ ((max_year - min_year)/2) + 32 ]
    },
    // steps of 1,
    connect: true,
    step: 10,
    tooltips: true,
    format: format,
    pips: { mode: 'steps', format: format , density: 10, values: [1902,1999]},
    
  });

  // The display values can be used to control the slider
  valuesSlider.noUiSlider.set([1902, 1999]);



  function updateTrace_opacity(which_ui,selected_range_index, hoverTemplateArray, selectedColorScheme) {
    console.log(which_ui)
    sizes = initialiseSizes();
    if (which_ui == "slider") {
      

      if (document.getElementById("first-song").value != "" && document.getElementById("first-song").value != "None" ) {
        if (document.getElementById("first-song").value != "None") {
          var firstSongValue = replaceAllSpaces(document.getElementById("first-song").value);
          firstSongIndex = find_index_of_name(firstSongValue);
          sizes[firstSongIndex] = 25;
          //console.log(sizes, firstSongValue, firstSongIndex)
        }
      }
      if (document.getElementById("traceDropdown").value == "post_encoding_selector") {
        Plotly.restyle(myDiv, {hovertemplate: [hoverTemplateArray], marker: {size: sizes,color: colors_post_encoding, opacity:selected_range_index, line: {width: 0}},xaxis:{range:[-70,70]},yaxis:{range:[-70,70]},autorange:false,type: 'scatter'});
      } else if (document.getElementById("traceDropdown").value == "form_selector") {
        Plotly.restyle(myDiv, {hovertemplate: [hoverTemplateArray], marker: {size: sizes,color: colors_form, opacity:selected_range_index, line: {width: 0}},xaxis:{range:[-70,70]},yaxis:{range:[-70,70]},autorange:false,type: 'scatter'});
      } else if (document.getElementById("traceDropdown").value == "harmony_selector") {
        Plotly.restyle(myDiv, {hovertemplate: [hoverTemplateArray], marker: {size: sizes,color: colors_style, opacity:selected_range_index, line: {width: 0}},xaxis:{range:[-70,70]},yaxis:{range:[-70,70]},autorange:false,type: 'scatter'});
      } else if (document.getElementById("traceDropdown").value == "year_selector") {
        Plotly.restyle(myDiv, {hovertemplate: [hoverTemplateArray], marker: {size: sizes,color: colors_year, opacity:selected_range_index, line: {width: 0}},xaxis:{range:[-70,70]},yaxis:{range:[-70,70]},autorange:false,type: 'scatter'});
      }	else if (document.getElementById("traceDropdown").value == "tonality_selector") {
        Plotly.restyle(myDiv, {hovertemplate: [hoverTemplateArray], marker: {size: sizes,color: colors_tonality, opacity:selected_range_index, line: {width: 0}},xaxis:{range:[-70,70]},yaxis:{range:[-70,70]},autorange:false,type: 'scatter'});
      }	else if (document.getElementById("traceDropdown").value == "composer_selector") {
        Plotly.restyle(myDiv, {hovertemplate: [hoverTemplateArray], marker: {size: sizes,color: colors_composer, opacity:selected_range_index, line: {width: 0}},xaxis:{range:[-70,70]},yaxis:{range:[-70,70]},autorange:false,type: 'scatter'});
      }	else if (document.getElementById("traceDropdown").value == "genre_selector") {
        Plotly.restyle(myDiv, {hovertemplate: [hoverTemplateArray], marker: {size: sizes,color: colors_genre, opacity:selected_range_index, line: {width: 0}},xaxis:{range:[-70,70]},yaxis:{range:[-70,70]},autorange:false,type: 'scatter'});
      }	
    } else {
      //Plotly.restyle('myDiv', update);
      if (selectedColorScheme == "post_encoding_color_selector") {
        Plotly.restyle(myDiv, {marker: {color: colors_post_encoding}});
      } else if (selectedColorScheme == "form_color_selector") {
        Plotly.restyle(myDiv, {marker: {color: colors_form}});
      } else if (selectedColorScheme == "harmony_color_selector") {
        Plotly.restyle(myDiv, {marker: {color: colors_style}});
      }	else if (selectedColorScheme == "year_color_selector") {
        Plotly.restyle(myDiv, {marker: {color: colors_year}});
      }	else if (selectedColorScheme == "tonality_color_selector") {
        Plotly.restyle(myDiv, {marker: {color: colors_tonality}});
      } else if (selectedColorScheme == "composer_color_selector") {
        Plotly.restyle(myDiv, {marker: {color: colors_composer}});
      } else if (selectedColorScheme == "genre_color_selector") {
        Plotly.restyle(myDiv, {marker: {color: colors_genre}});
      }
      var valuesSlider = document.getElementById('values-slider');
			valuesSlider.noUiSlider.set([1902, 1999]);	
    }
    	
  }

  valuesSlider.noUiSlider.on('change', function( values, handle) {
    var values_array = years; // Your unsorted array of values
    var minValue = values[0]; // Minimum value for the range
    var maxValue = values[1]; // Maximum value for the range
    // Create an array of objects with values and original indices
    // Find the indices of values that meet the range requirement
    var selected_range_index = [];
    //console.log(values_array);
    for (i = 0; i < values_array.length; i++) {
      //console.log(parseInt(values_array[i]), parseInt(minValue))
      if (parseInt(values_array[i]) >= parseInt(minValue) && parseInt(values_array[i]) <= parseInt(maxValue)) {
        
        selected_range_index.push(i)
        //console.log(dropdown.value)
        
        //console.log(values,i);
      }
    }
    // Map the keyList to create a new array with values of 0.2 at specified keys
    var resultArray = defaultOpacity.map((value, index) => {
      if (!selected_range_index.includes(index)) {
        return 0;
      }
      return value;
    });

    var resultHoverTemplateArray = hoverTemplateArray.map((value, index) => {
      if (!selected_range_index.includes(index)) {
        return '';
      }
      return value;
    });
    //console.log(resultHoverTemplateArray);

    //console.log(resultArray);
    updateTrace_opacity(which_ui="slider", resultArray, resultHoverTemplateArray, 0)
    

    //console.log(selected_range_index); // Array of indices that meet the range requirement

    //console.log(values,handle, year)
  
  });

  // Zoom feature
  graphDiv.on('plotly_relayout',
      function(eventdata){
        JSON.stringify(eventdata);
        console.log(eventdata['xaxis.range[0]'], eventdata['xaxis.range[1]'])
        sizes = initialiseSizes();
        if (eventdata['xaxis.range[0]'] == -70 || eventdata['xaxis.range[1]'] == 70)  {
          w = 4
          //console.log(w)
        } else {
          var zoomedXAxisLength = eventdata['xaxis.range[1]'] - eventdata['xaxis.range[0]']; // Replace with the actual zoomed x-axis length
          var zoomedYAxisLength = eventdata['yaxis.range[1]'] - eventdata['yaxis.range[0]']; // Replace with the actual zoomed y-axis length
          // Define the initial marker size
          var initialMarkerSize = 4;
          const scalingFactor = 6; // Adjust this value to control the rate of marker size change

          w = initialMarkerSize + (1 - (zoomedXAxisLength + zoomedYAxisLength)/(total_x_range + total_y_range)) * scalingFactor;

          sizes.fill(w);

          
        }

        if (document.getElementById("first-song").value != "" && document.getElementById("first-song").value != "None" ) {
          if (document.getElementById("first-song").value != "None") {
            var firstSongValue = replaceAllSpaces(document.getElementById("first-song").value);
            firstSongIndex = find_index_of_name(firstSongValue);
            sizes[firstSongIndex] = 25;
            //console.log("SIZES:",sizes)
          }
        }
        
        
        if (document.getElementById('traceDropdown-color_schemes').value == "post_encoding_color_selector") {
          color_based_on_color_selector = colors_post_encoding
        } else if (document.getElementById('traceDropdown-color_schemes').value == "form_color_selector") {
          color_based_on_color_selector = colors_form
        } else if (document.getElementById('traceDropdown-color_schemes').value == "harmony_color_selector") {
          color_based_on_color_selector = colors_style
        } else if (document.getElementById('traceDropdown-color_schemes').value == "year_color_selector") {
          color_based_on_color_selector = colors_year
        } else if (document.getElementById('traceDropdown-color_schemes').value == "tonality_encoding_color_selector") {
          color_based_on_color_selector = colors_tonality
        } else if (document.getElementById('traceDropdown-color_schemes').value == "composer_encoding_color_selector") {
          color_based_on_color_selector = colors_composer
        } else if (document.getElementById('traceDropdown-color_schemes').value == "genre_encoding_color_selector") {
          color_based_on_color_selector = colors_genre
        }

        
        console.log(w)
        var update = {
          
          'marker.size': sizes
        };
        //Plotly.restyle('myDiv', update);
        if (document.getElementById("traceDropdown").value == "post_encoding_selector") {
          Plotly.restyle(myDiv, {marker: {size: sizes,color: color_based_on_color_selector, line: {width: 0}}});
        } else if (document.getElementById("traceDropdown").value == "form_selector") {
          Plotly.restyle(myDiv, {marker: {size: sizes,color: color_based_on_color_selector, line: {width: 0}}});
        } else if (document.getElementById("traceDropdown").value == "harmony_selector") {
          Plotly.restyle(myDiv, {marker: {size: sizes,color: color_based_on_color_selector, line: {width: 0}}});
        } else if (document.getElementById("traceDropdown").value == "year_selector") {
          Plotly.restyle(myDiv, {marker: {size: sizes,color: color_based_on_color_selector, line: {width: 0}}});
        }	else if (document.getElementById("traceDropdown").value == "tonality_selector") {
          Plotly.restyle(myDiv, {marker: {size: sizes,color: color_based_on_color_selector, line: {width: 0}}});
        }	else if (document.getElementById("traceDropdown").value == "composer_selector") {
          Plotly.restyle(myDiv, {marker: {size: sizes,color: color_based_on_color_selector, line: {width: 0}}});
        } else if (document.getElementById("traceDropdown").value == "genre_selector") {
          Plotly.restyle(myDiv, {marker: {size: sizes,color: color_based_on_color_selector, line: {width: 0}}});
        }
          
      });

});

}

window.addEventListener("error", (event) => {
  //log.textContent = `${log.textContent}${event.type}: ${event.message}\n`;
  console.log("message",event);
});
  
  // For other errors, you can choose to log or handle them as needed
  //console.error('An error occurred:', error);

