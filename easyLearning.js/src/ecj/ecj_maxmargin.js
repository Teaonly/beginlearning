(function(exports) {
  "use strict";

  var Volume, act;
  if ( exports.Volume !== undefined ) {
    Volume = exports.Volume;
  } else {
    Volume = require("./ecj_volume.js").Volume;
  }

  var MaxMarginLayer = function(opt) {
    this.inputSize = opt.inputSize;
    this.out = new Volume(1, 1, this.inputSize);
    this.din = new Volume(1, 1, this.inputSize);
    this.margin = opt.margin | 1.0;
    this.type = "maxmargin";
  };

  MaxMarginLayer.prototype = {
    forward: function(data) {
      var i;
      for(i = 0; i < this.inputSize; i++) {
        this.out.$_(i, data.$(i) );
      }
      return this.out;
    },

    loss: function(y) {
      var i, sum;
      sum = 0.0;
      for(i = 0; i < this.inputSize; i++) {
        if ( i != y[0] ) {
          if ( this.out.$(i) - this.out.$(y[0]) + this.margin > 0) {
            sum += this.out.$(i) - this.out.$(y[0]) + this.margin;
          }
        }
      }
      return sum;
    },

    backward: function(y) {
      var i;

      this.din.$_(y[0],0);
      for(i = 0; i < this.inputSize; i++) {
        if ( i != y[0] ) {
          if ( this.out.$(i) - this.out.$(y[0]) + this.margin > 0) {
            this.din.$_(i,1);
            this.din.$_(y[0], this.din.$(y[0]) - this.margin);
          }
        }
      }
      return this.din;
    },
  };

  var HingelossLayer = function(opt) {
    // 只能处理二分类问题
    if (opt.inputSize !== 1) {
      throw new Error().stack;
    }
    this.inputSize = opt.inputSize;
    this.out = new Volume(1, 1, this.inputSize);
    this.din = new Volume(1, 1, this.inputSize);
    this.type = "hinge";
  };

  HingelossLayer.prototype = {
    forward: function(data) {
      var i;
      for(i = 0; i < this.inputSize; i++) {
        this.out.$_(i, data.$(i) );
      }
      return this.out;
    },

    loss: function(y) {
      var value = 1 - y*this.out.$(0);
      if ( value > 0) {
        return value;
      } else {
        return 0;
      }
    },

    backward: function(y) {
      var value = 1 - y*this.out.$(0);
      if ( value > 0) {
        this.din.$_(0, -1*y);
      } else {
        this.din.$_(0, 0);
      }
      return this.din;
    },

  };

  exports.MaxMarginLayer = MaxMarginLayer;
  exports.HingelossLayer = HingelossLayer;

})((typeof module != 'undefined' && module.exports) || ecj )
