(function(exports) {
  "use strict";

  var Volume, act;
  if ( exports.Volume !== undefined ) {
    Volume = exports.Volume;
  } else {
    Volume = require("./ecj_volume.js").Volume;
  }

  var SoftmaxLayer = function(opt) {
    this.type = "softmax";
    this.inputSize = opt.inputSize;

    this.out = new Volume(1, 1, this.inputSize);
    this.din = this.out;
  };

  SoftmaxLayer.prototype = {
    forward: function(data) {
      var i;
      var sum = 0.0;

      var maxv = data.$(0);
      for(i = 1; i < this.inputSize; i++) {
        if ( data.$(i) > maxv ) {
          maxv = data.$(i);
        }
      }

      for(i = 0; i < this.inputSize; i++) {
        this.out.$_(i, Math.exp(data.$(i) - maxv) + 0.0000000001);
        sum += this.out.$(i);
      }

      for (i = 0; i < this.inputSize; i++) {
        this.out.$_(i, this.out.$(i) / sum);
      }
      return this.out;
    },

    loss: function(y) {
      var i;
      return -1 * Math.log( this.out.$(y[0]) + 0.0000000001 );
    },

    backward: function(y) {
      var i;
      this.din.$_(y[0], this.din.$(y[0]) - 1);
      return this.din;
    },
  };

  exports.SoftmaxLayer = SoftmaxLayer;

})((typeof module != 'undefined' && module.exports) || ecj )
