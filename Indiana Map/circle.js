let canvasI = document.getElementById('myCanvas');
let ctx = canvasI.getContext('2d');
let image = new Image(700,900);
image.onload = function(){
  ctx.drawImage(image,0,0,700,900);
}

function draw() {
  canvasI.addEventListener("click", (event) => 
  { 
    draw_circles(event.offsetX, event.offsetY, 'RED');
    mouseMoved(event.offsetX, event.offsetY);
    console.log("LONG and LAT");
    console.log (ytoLat(event.offsetY), xtoLon(event.offsetX));
  });
}
image.src = 'imageIndiana.jpg';

function draw_circles(X,Y, color) {
  var x = X;
  var y = Y;
  var R = 9;
  ctx.globalAlpha = 0.7;
  ctx.beginPath();
  ctx.arc(x, y, R, 0, 2 * Math.PI, false); 
  ctx.lineWidth = 1;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.fill();
  ctx.stroke();
}
 
function mouseMoved(x,y){
  alert("You are at " + "LONGITUDE : " + ytoLat(y) + "," + " LATITUDE : " + xtoLon(x)+ "!");
}

/*
 *  NE canvasI (650, 9) 
 *  NE Point (41.760, -84.806)
 *   
 *  SW canvasI (45, 892)
 *  SW Point (37.798, -88.093)
 *
 *  LATITUDE: input y -> output latitude
 *    y = screen corrdinates; lat = latitude
 *
 *    y = 9, lat = 41.760;
 *    y = 892, lat = 37.798;
 * 
 *    slope = (41.760 - 37.789) / (9 - 892)
 *    y = -0.0044869762174405x + 41.800382785957
 * 
 *  Longitude: input x -> output longitute
 *    y = screen corrdinates; long = longitude
 *
 *    y = 650, lat = -84.772;
 *    y = 45, lat = -88.093;
 * 
 *    slope = (-84.772 - (-88.093)) / (650 - 45)
 *    y = 0.0054892561983471x – 88.340016528926
 */


function ytoLat(y){
  res = -0.0044869762174405 * y + 41.800382785957;
  return (res.toFixed(4));
}

function xtoLon(x){
  res = (0.0054330578512397 * x) - 88.337487603306;
  return (res.toFixed(4));
}



