(function(exports) {
  "use strict";

  var Volume;
  if ( exports.Volume !== undefined ) {
    Volume = exports.Volume;
  } else {
    Volume = require("./ecj_volume.js").Volume;
  }

  var SimpleSGDTrainer = function(opts) {
    if ( opts.learningRate !== undefined) {
      this.learningRate = opts.learningRate;
    } else {
      throw new Error().stack;
    }

    if ( opts.gradNumber !== undefined) {
      this._grad = new Volume(1, 1, opts.gradNumber);
    } else {
      throw new Error().stack;
    }

    if ( opts.momentum !== undefined) {
      this.momentum = opts.momentum;
    } else {
      this.momentum = 0;
    }
  };

  SimpleSGDTrainer.prototype = {
    init: function() {
      this._grad.reset();
    },

    update: function(network) {
      var i,j,k,n,v;
      n = 0;
      for(i = 0; i < network.hiddenLayers.length; i++) {
        for(j = 0; j < network.hiddenLayers[i].w.length; j++) {
          for(k = 0; k < network.hiddenLayers[i].w[j].size; k++) {
            v = network.grad.$(n) + this._grad.$(n) * this.momentum;
            this._grad.$_(n, v);
            v = -1 * v * this.learningRate + network.hiddenLayers[i].w[j].$(k);
            network.hiddenLayers[i].w[j].$_(k, v);
            n++;
          }
        }
      }
    },
  };


  var AdaDeltaTrainer = function(opts) {
    if ( opts.decayRate !== undefined) {
      this.decayRate = opts.decayRate;
    } else {
      this.decayRate = 0.95;
    }
    if ( opts.epsilon !== undefined) {
      this.epsilon = opts.epsilon;
    } else {
      this.epsilon = 0.0001;
    }

    if ( opts.gradNumber !== undefined) {
      this._squaredGrad = new Volume(1, 1, opts.gradNumber);
      this._squaredDelta = new Volume(1, 1, opts.gradNumber);
    } else {
      throw new Error().stack;
    }
  };

  AdaDeltaTrainer.prototype = {
    init: function() {
      this._squaredGrad.reset();
      this._squaredDelta.reset();
    },

    update: function(network) {
      var i,j,k,n,v,delta;

      // 首先更新 squared grad
      for(n = 0; n < this._squaredGrad.size; n++) {
        v = Math.pow(network.grad.$(n), 2) * (1 - this.decayRate);
        v = v + this._squaredGrad.$(n) * this.decayRate;
        this._squaredGrad.$_(n, v);
      }


      n = 0;
      for(i = 0; i < network.hiddenLayers.length; i++) {
        for(j = 0; j < network.hiddenLayers[i].w.length; j++) {
          for(k = 0; k < network.hiddenLayers[i].w[j].size; k++) {

            // 更新参数
            delta = -1 * network.grad.$(n) * Math.sqrt( this._squaredDelta.$(n) + this.epsilon )
                       / Math.sqrt( this._squaredGrad.$(n) + this.epsilon );

            v = network.hiddenLayers[i].w[j].$(k) + delta;
            network.hiddenLayers[i].w[j].$_(k, v);

            // 更新 squared delta
            v = Math.pow(delta, 2) * (1 - this.decayRate);
            v = v + this._squaredDelta.$(n) * this.decayRate;
            this._squaredDelta.$_(n, v);

            n++;
          }
        }
      }
    },

  };

  exports.SimpleSGDTrainer = SimpleSGDTrainer;
  exports.AdaDeltaTrainer = AdaDeltaTrainer;

})((typeof module != 'undefined' && module.exports) || ecj )
