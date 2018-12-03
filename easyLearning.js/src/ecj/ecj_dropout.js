(function(exports) {
  "use strict";

  var Volume;
  if ( exports.Volume !== undefined ) {
    Volume = exports.Volume;
  } else {
    Volume = require("./ecj_volume.js").Volume;
  }

  var DropoutLayer = function(opt) {
    this.sx = opt.sx;
    this.sy = opt.sy;
    this.depth = opt.depth;
    this.dropProb = opt.dropProb;
    this.type = "dropout";

    if ( this.dropProb === undefined || this.dropProb > 1) {
      throw new Error().stack;
    }

    this.out = new Volume( this.sx, this.sy, this.depth);
    this.din = new Volume( this.sx, this.sy, this.depth);

    this.w = [];
    this.dw = [];
  };

  DropoutLayer.prototype = {
    forward: function(data, isTrain) {
      var i;
      if ( isTrain ) {
        this.out.reset();
        this.din.reset();
        for ( i = 0; i < this.out.size; i++) {
          if ( Math.random() > this.dropProb ) {
            this.out.$_(i, data.$(i)/this.dropProb);
            this.din.$_(i, 1/this.dropProb);
          }
        }
        return this.out;
      } else {
        return data;        // just by pass
      }
    },

    backward: function(data) {
      var i;
      for ( i = 0; i < this.out.size; i++) {
        this.din.$_(i, this.din.$(i) * data.$(i));
      }
      return this.din;
    },

  };

  exports.DropoutLayer = DropoutLayer;
})((typeof module != 'undefined' && module.exports) || ecj )
