(function(exports) {
  "use strict";

  var native = undefined;
  if ( typeof window === 'undefined' ) {
    //native = require('../build/Release/nativeConvnet');
  }

  var heapMemory = undefined;
  var heapSize = 0;

  var begin = function() {
    heapMemory = undefined;
    heapSize = 0;
  };

  var end = function() {
    // 分配实际内存
    heapMemory = new Float64Array(heapSize);
    console.log("分配内存: " + heapMemory.length + " " + heapSize);

    if ( native !== undefined) {
      native.initVolume(heapMemory, heapSize);
    }
  };

  var save = function(fileName) {
    var heapData = JSON.stringify(heapMemory);
    var fs = require('fs');
    fs.writeFileSync(fileName, heapData);
  };

  var load = function(heapData) {
    var i;
    for (i = 0; i < heapSize; i++) {
      heapMemory[i] = heapData[i];
    }
  };


  var Volume = function(sx, sy, depth, expand) {
    this.sx = sx;
    this.sy = sy;
    this.depth = depth;
    this.size = sx * sy * depth;
    this._offset = heapSize;

    if ( expand === undefined ) {
      this.size = sx * sy * depth;
    } else {
      this.size = sx * sy * depth + expand;
    }
    heapSize = heapSize + this.size;

  };


  var gaussion = function() {
    var u = 2*Math.random()-1;
    var v = 2*Math.random()-1;
    var r = u*u + v*v;
    if ( r >= 1)
      return gaussion();
    var c = Math.sqrt(-2*Math.log(r)/r);
    return c*u;
  };

  Volume.prototype = {
    maxValue: function() {
      var i;
      var max = this.$(0);
      for(i = 1; i < this.size; i++) {
        if ( this.$(i) > max ) {
          max = this.$(i);
        }
      }
      return max;
    },
 
    maxIndex: function() {
      var i;
      var max = this.$(0);
      var maxIndex = 0;
      for(i = 1; i < this.size; i++) {
        if ( this.$(i) > max ) {
          max = this.$(i);
          maxIndex = i;
        }
      }
      return maxIndex;
    },
    
    randnorm: function(mu, std) {
      var i;
      for(i = 0; i < this.size; i++) {
        heapMemory[i + this._offset] = mu + gaussion() * std;
      }
    },
    
    random: function(min,max) {
      var i;
      for(i = 0; i < this.size; i++) {
        heapMemory[i + this._offset] = min + Math.random() * (max-min);
      }
    },
    reset: function(v) {
        var i;
        if ( v === undefined) {
            v = 0.0;
        }
        for(i = 0; i < this.size; i++) {
          heapMemory[i + this._offset] = 0.0;
        }
    },

    $: function(i) {
      return heapMemory[i + this._offset];
    },

    $_: function(i, v) {
      /*
      if ( isNaN(v)) {
        throw new Error().stack;
      }
      */
      heapMemory[i + this._offset] = v;
    },

    $$: function(x, y, d) {
        var ix=((this.sx * y)+x)*this.depth + d + this._offset;
        return heapMemory[ix];
    },

    $$_: function(x, y, d, v) {
        var ix=((this.sx * y)+x)*this.depth + d + this._offset;
        heapMemory[ix] = v;
    },
  };

  exports.vol = {};
  exports.vol.begin = begin;
  exports.vol.end = end;
  exports.vol.save = save;
  exports.vol.load = load;
  exports.vol.heapMemory = heapMemory;
  exports.vol.heapSize = heapSize;

  exports.Volume = Volume;
})((typeof module != 'undefined' && module.exports) || ecj )
