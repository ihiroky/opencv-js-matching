const fs = require('fs')

fs.copyFileSync('src/html/index.html', 'dist/index.html')
fs.copyFileSync('src/html/target.png', 'dist/target.png')
fs.copyFileSync('src/html/whole.png', 'dist/whole.png')
fs.copyFileSync('opencv/build_wsam/bin/opencv.js', 'dist/opencv.js')
