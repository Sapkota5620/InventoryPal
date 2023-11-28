const http = require('http')
const port = 4099
const fs = require('fs')

function getFile(path, type) {

    try {
        const data = fs.readFileSync(path, 'utf8')
        console.log(data);
        return data;
    } catch (err) {
        console.error(err)
        return "";
    }

}

function sendimage(res, path, type)
{
    try {
        const data = fs.readFileSync(path)
        console.log('1')
        res.writeHead(200, {'Context-Type': `image/${type}`})
        console.log('1')
        res.end(data)
        console.log('reached')
    } catch(err) {
        res.writeHead(500, {'Context-Type': 'text/plain'})
        res.end('500 - Internal Error')
    }
}


const server = http.createServer((req, res) => {
    console.log(req.url)
    const path = req.url.replace(/.*\?/,'')
    res.writeHead(200, {'Content-Type': 'text/html',})
    data = getFile("map.html")
    res.end(data)

})

server.listen(port, () => console.log(`Running on port ${port}.`))
