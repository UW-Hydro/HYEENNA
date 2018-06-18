function chord_plot(matrix, names, colors) {
    var w = 700;
    var h = 700;
	var opacity = 0.9;

    var margin = {top: 20,
                  bottom: 0,
                  left: 0,
                  right: 0};

    var width = w - margin.left - margin.right;
    var height = h - margin.top - margin.bottom;

    var svg = d3.select("#diagram")
                .append("svg")
                .attr("id", "chart")
                .attr("width", w)
                .attr("height", h);

	var wrapper = svg.append("g").attr("transform", "translate(" + width / 2 + "," + height / 2 + ")");
    var outerRadius = Math.min(width, height) * 0.5 -55;
    var innerRadius = outerRadius - 30;
    var chordGenerator = d3.chord()
      .padAngle(0.05)
      .sortSubgroups(d3.descending)
      .sortChords(d3.descending);
    var chord = chordGenerator(matrix);
    var arcs = d3.arc()
      .innerRadius(innerRadius)
      .outerRadius(outerRadius);

    var ribbon = d3.ribbon().radius(250);

    var opacities = d3.scaleOrdinal()
      .domain(d3.range(names.length))
      .range(opacity);

    var color = d3.scaleOrdinal()
    .domain(d3.range(names.length))
    .range(colors);

    // creating the fill gradient
    function getGradID(d){ return "linkGrad-" + d.source.index + "-" + d.target.index; }


    var grads = svg.append("defs")
      .selectAll("linearGradient")
      .data(chord)
      .enter()
      .append("linearGradient")
      .attr("id", getGradID)
      .attr("gradientUnits", "userSpaceOnUse")
      .attr("x1", function(d, i){ return innerRadius * Math.cos((d.source.endAngle-d.source.startAngle) / 2 + d.source.startAngle - Math.PI/2); })
      .attr("y1", function(d, i){ return innerRadius * Math.sin((d.source.endAngle-d.source.startAngle) / 2 + d.source.startAngle - Math.PI/2); })
      .attr("x2", function(d,i){ return innerRadius * Math.cos((d.target.endAngle-d.target.startAngle) / 2 + d.target.startAngle - Math.PI/2); })
      .attr("y2", function(d,i){ return innerRadius * Math.sin((d.target.endAngle-d.target.startAngle) / 2 + d.target.startAngle - Math.PI/2); });

      // set the starting color (at 0%)

    grads.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", function(d){ return color(d.target.index)});

        //set the ending color (at 100%)
    grads.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", function(d){ return color(d.source.index)});

    // making the ribbons
    d3.select("g")
      .selectAll("path")
      .data(chord)
      .enter()
      .append("path")
      .attr("class", function(d) {
		// The first chord allows us to select all of them.
		// The second chord allows us to select each individual one.
        return "chord chord-" + d.source.index + " chord-" + d.target.index
      })
      .style("fill", function(d){ return "url(#" + getGradID(d) + ")"; })
      .attr("d", ribbon)
      // .style("stroke", function(d){ return d3.rgb(color(d.target.index)).darker(); })
      .style("opacity", function(d){ return 0.6 });

    // making the arcs
    var g = wrapper.selectAll("g")
      .data(chord.groups)
      .enter()
      .append("g")
      .attr("class", "group");


    g.append("path")
      .style("fill", function(d){ return color(d.index)})
      // .style("stroke", function(d){ return d3.rgb(color(d.index)).darker(); })
      .attr("d", arcs);
      //.style("opacity", 1)

    /// adding labels
    g.append("text")
      .each(function(d){ d.angle = (d.startAngle + d.endAngle) / 2; })
      .attr("dy", ".35em")
      .attr("class", "titles")
      .attr("text-anchor", "middle" )//function(d) { return d.angle > Math.PI ? "end" : null; })
      .attr("transform", function(d) {
        return "rotate(" + (d.angle * 180 / Math.PI - 90) + ")"
        + "translate(" + (outerRadius + 15) + ")"
        + "rotate(" + (270)  + ")"
        + (d.angle < Math.PI/2 ? "rotate(180)" : "")
        + (d.angle > Math.PI *1.25 ? "rotate(180)" : "");
      })
      .text(function(d,i){ return names[i]; })
      .style("font-size", "18px");
}
