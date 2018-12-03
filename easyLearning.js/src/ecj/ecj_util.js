(function(exports) {
    var fs = require('fs'),
        PNG = require('pngjs').PNG,
        JPEG = require('jpeg-js');

    var Volume, act;
    if ( exports.Volume !== undefined ) {
      Volume = exports.Volume;
    } else {
      Volume = require("./ecj_volume.js").Volume;
    }

    var util = {};

    util.readJpeg = function(filePath) {
      var jpegData = fs.readFileSync(filePath);
      var jpegObject = JPEG.decode(jpegData);
      return jpegObject;
    };

    util.readRawPNG = function(filePath, fn) {
      fs.createReadStream(filePath)
        .pipe(new PNG({
            filterType: 4
        }))
        .on('parsed', function() {
            fn(this.data, this.width, this.height);
        });
    };

    util.readGrayPNG = function(filePath, buffer, fn) {
      fs.createReadStream(filePath)
        .pipe(new PNG({
            filterType: 4
        }))
        .on('parsed', function() {

            if ( buffer === undefined ) {
              buffer = new Float64Array(this.width * this.height);
            }

            src = 0;
            dst = 0;
            var grad = 0;
            for (var i = 0; i < this.width * this.height; i++) {
              grad = this.data[dst+0] * 0.299 + this.data[dst+1] * 0.587 + this.data[dst+2] * 0.114;
              grad = (grad - 128)/128;
              buffer[src++] = grad;
              dst+=4;
            }
            fn(buffer, this.width, this.height);
        });
    };

    util.readPNG = function(filePath, buffer, fn) {
      fs.createReadStream(filePath)
        .pipe(new PNG({
            filterType: 4
        }))
        .on('parsed', function() {

            if ( buffer === undefined ) {
              buffer = new Float64Array(this.width * this.height * 3);
            }

            src = 0;
            dst = 0;
            for ( i = 0; i < this.width * this.height; i++) {
              buffer[src++] = (this.data[dst++] - 128)/128;  // R,G,B
              buffer[src++] = (this.data[dst++] - 128)/128;
              buffer[src++] = (this.data[dst++] - 128)/128;
              dst++;
            }
            fn(buffer, this.width, this.height);
        });
    };

    util.writePNG = function(filePath, d, width, height) {
      var x, y , i, maxV, minV;

      maxV = d[0];
      minV = d[0];
      for(i = 0; i < width*height; i++) {
        if ( d[i] > maxV) {
          maxV = d[i];
        }
        if ( d[i] < minV) {
          minV = d[i];
        }
      }

      var png = new PNG({
        filterType: -1,
        'width': width,
        'height': height
      });

      var n = 0;
      for (y = 0; y < png.height; y++) {
        for (x = 0; x < png.width; x++) {
          var idx = (png.width * y + x) << 2;
          var v = Math.floor( (d[n] - minV) / (maxV - minV) * 255 );

          png.data[idx] = v;
          png.data[idx+1] = v;
          png.data[idx+2] = v;
          png.data[idx+3] = 255;
          n++;
        }
      }

      var dst = fs.createWriteStream(filePath);
      png.pack().pipe(dst);

    };

    util.expandFeatures = function(samples, pow) {
        var i, j, k, l;
        var allMap = {};
        var sx = samples[0].length;
        var combin = new Array(sx);

        for(i = 2; i <= pow; i++) {
            for(j = 0; j < sx; j++) {
                combin[j] = 0;
            }
            util._buildComb(allMap, combin, i);
        }

        for(i in allMap) {
            combin = JSON.parse("[" + i + "]");

            for(j = 0; j < samples.length; j++) {
                v = 1.0;
                for(k = 0; k < combin.length; k++) {
                    for (l = 0; l < combin[k]; l++) {
                        v = v * samples[j][k];
                    }
                }
                samples[j].push(v);
            }
        }
    };

    util._buildComb = function(allMap, combin, l) {
        var i;

        if( l == 0) {
            allMap[ combin.toString() ] = 1;
            return;
        }
        for(i = 0; i < combin.length; i++) {
            combin[i] += 1;
            util._buildComb(allMap, combin, l-1);
            combin[i] -= 1;
        }
    };

    exports.util = util;
})( (typeof module != 'undefined' && module.exports) || ecj );
