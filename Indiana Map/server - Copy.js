const http = require('http')
const port = 4099
const fs = require('fs');
const {MongoClient} = require('mongodb')

const getinfo = async (res,coords) => {
    const uri = "mongodb://app:ziggy@139.102.55.156:27017/geo"
    const client = new MongoClient(uri)
    let near = {near:{'type':'Point','coordinates':coords},
                spherical:true,
                distanceField:'calcDistance',
                distanceMultiplier:1/1609.0}
    console.log(near['near'])
    let stage1 = {$geoNear:near}
    let stages = [stage1]

    try {
        await client.connect()
        const result = await client.db("geo").collection("airports").aggregate(stages).toArray()
        res.writeHead(200, {'Content-Type': 'text/plain',})
 /*
  *     send the header
  *     send the new data
  *     send the trailer
  */
        data = result[0].name + ', ' + result[0].city + ', ' + result[0].state
        res.end(data)
    } catch(e) {
        console.error(e)
    } finally {
        await client.close()
    }
}

function sendimage(res, path, type)
{
    try {
        const data = fs.readFileSync(path)
        res.writeHead(200, {'Context-Type': `image/${type}`})
        res.end(data)
    } catch(err) {
        res.writeHead(500, {'Context-Type': 'text/plain'})
        res.end('500 - Internal Error')
    }
}

function getFile(path)
{
    try {
        const data = fs.readFileSync(path, 'utf8')
        return data
    } catch(err) {
        console.error(err)
        return ""
    }
}

const server = http.createServer((req,res) => {
    let path, lon, lat

    console.log(req.url)
    if(req.url.includes('?')){
        path = req.url.replace(/\?.*/,'')
        const coords = req.url.replace(/.*\?/,'')
        const arr = coords.split('&')
        const londata = arr[0].split('=')
        const latdata = arr[1].split('=')
        lat = parseFloat(latdata[1])
        lon = parseFloat(londata[1])
        console.log('coords: ' + lon + ',' + lat)
    } else {
        path = req.url
    }
    console.log('path: ' + path)

    switch(path){
        case '/':
            res.writeHead(200, {'Content-Type': 'text/html',})
            data = getFile("mao.html")
            res.end(data)
            break
        case '/finder':
            getinfo(res, new Array(lon,lat))
            break
        case '/about':
            res.writeHead(200, {'Content-Type': 'text/plain',})
            res.end('About stuff')
            break
        case '/circle.js':
            res.writeHead(200, {'Content-Type': 'application/json',})
            data.getFile('circle.js')
            res.end('Help stuff')
            break
        case '/imageIndiana.png':
            sendimage(res, "imgaeIndiana.png","png")
            break
        case '/mapcss.css':
            res.writeHead(200, {'Content-Type': 'text/css',})
            data = getFile("map.css")
            res.end(data)
            break
        default:
            res.writeHead(200, {'Content-Type': 'text/plain',})
            res.end('Not found:' + req.url)
            break
    }
})

server.listen(port, () => console.log(`Running on port ${port}.`))

