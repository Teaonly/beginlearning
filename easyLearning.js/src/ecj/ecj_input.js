(function(exports) {
  "use strict";

  var Volume;
  if ( exports.Volume !== undefined ) {
    Volume = exports.Volume;
  } else {
    Volume = require("./ecj_volume.js").Volume;
  }

  var InputLayer = function(opt) {
    this.sx = opt.sx;
    this.sy = opt.sy;
    this.depth = opt.depth;

    this._data = new Volume( this.sx, this.sy, this.depth);
  };

  InputLayer.prototype = {
    forward: function(sampleData) {
      var i;
      for(i = 0; i < this._data.size; i++) {
        this._data.$_(i, sampleData[i]);
      }
      return this._data;
    },
  };

  exports.InputLayer = InputLayer;
})((typeof module != 'undefined' && module.exports) || ecj )
