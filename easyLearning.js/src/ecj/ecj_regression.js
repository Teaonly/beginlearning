(function(exports) {
  "use strict";

  var Volume, act;
  if ( exports.Volume !== undefined ) {
    Volume = exports.Volume;
  } else {
    Volume = require("./ecj_volume.js").Volume;
  }

  var L1RegressionLayer = function(opt) {
    this.inputSize = opt.inputSize;
    this.out = null;
    this.dz = new Volume(1, 1, this.inputSize);
    this.type = "l1regression";
  };

  L1RegressionLayer.prototype = {
    forward: function(data) {
      this.out = data;
      return this.out;
    },

    loss: function(y) {
      var i, sum;
      sum = 0.0;
      for(i = 0; i < this.inputSize; i++) {
        sum = sum + Math.abs(this.out.$(i) - y[i]);
      }
      return sum;
    },

    backward: function(y) {
      var i;
      for(i = 0; i < this.inputSize; i++) {
        if ( this.out.$(i) > y[i]) {
          this.dz.$_(i, 1);
        } else {
          this.dz.$_(i, -1);
        }
      }
      return this.dz;
    },
  };


  var L2RegressionLayer = function(opt) {
    this.inputSize = opt.inputSize;
    this.out = null;
    this.dz = new Volume(1, 1, this.inputSize);
    this.type = "l2regression";
  };

  L2RegressionLayer.prototype = {

    forward: function(data) {
      this.out = data;
      return this.out;
    },

    loss: function(y) {
      var i, sum;
      sum = 0.0;
      for(i = 0; i < this.inputSize; i++) {
        sum = sum + (this.out.$(i) - y[i]) * ( this.out.$(i) - y[i]);
      }
      return 0.5 * sum;
    },

    backward: function(y) {
      var i;
      for(i = 0; i < this.inputSize; i++) {
        this.dz.$_(i, (this.out.$(i) - y[i]));
      }
      return this.dz;
    },
  };

  var LogisticRegressionLayer = function(opt) {
    if ( opt.inputSize !== 1) {
      throw new Error().stack;
    }

    this.inputSize = opt.inputSize;
    this.out = new Volume(1, 1, this.inputSize);
    this.dz = new Volume(1, 1, this.inputSize);

    this.type = "logistic";
  };

  LogisticRegressionLayer.prototype = {
    forward: function(data) {
      this.out.$_(0, 1 / (1+ Math.exp(-1*data.$(0))) );
      return this.out;
    },

    loss: function(y) {
      var l;
      if ( y[0] === 0) {
        l = -1 * Math.log(1- this.out.$(0));
      } else {
        l = -1 * Math.log(this.out.$(0));
      }
      return l;
    },

    backward: function(y) {
      this.dz.$_(0, this.out.$(0) - y[0]);
      return this.dz;
    },
  };


  exports.L1RegressionLayer = L1RegressionLayer;
  exports.L2RegressionLayer = L2RegressionLayer;
  exports.LogisticRegressionLayer = LogisticRegressionLayer;
})((typeof module != 'undefined' && module.exports) || ecj )
