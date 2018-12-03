(function(exports) {
  "use strict";

  var Volume, act;
  if ( exports.Volume !== undefined ) {
    Volume = exports.Volume;
    act = exports.act;
  } else {
    Volume = require("./ecj_volume.js").Volume;
    act = require("./ecj_active.js").act;
  }

  var FullConnectLayer = function(opt) {
    this.type = "full";
    this.actType = opt.actType;
    this.number = opt.number;
    this.inputSize = opt.inputSize;

    this.act = act.getActivation(this.actType);

    // 在计算中需要分配的内存，提前分配好
    this.w = [];
    this.dw = [];
    var i;
    for(i = 0; i < this.number; i++) {
        this.w.push( new Volume(1, 1, this.inputSize+1) );
        this.dw.push( new Volume(1, 1, this.inputSize+1) );
    }

    this.dz = new Volume(1, 1, this.number);
    this.out = new Volume(1, 1, this.number);

    this.in = new Volume(1, 1, this.inputSize);
    this.din = new Volume(1, 1, this.inputSize);
  };

  FullConnectLayer.prototype = {
    init: function() {
      var i;
      var std = Math.sqrt(1/this.inputSize);
      for(i = 0; i < this.number; i++) {
        this.w[i].randnorm(0, std);
        this.w[i].$_(this.inputSize, 0.0);
      }
    },

    forward: function(data, isTrain) {
      var i,j;
      var sum;

      for(i = 0; i < this.inputSize; i++) {
        this.in.$_(i, data.$(i));
      }

      for(i = 0; i < this.number; i++) {
        sum = this.w[i].$(this.inputSize);
        for(j = 0; j < this.inputSize; j++) {
          sum = sum + this.w[i].$(j) * data.$(j);
        }
        this.out.$_(i, this.act.forward(sum));
        if ( isTrain ) {
          this.dz.$_(i, this.act.backward(sum, this.out.$(i)));
        }

      }

      return this.out;
    },

    // 链式规则 ＋ 多分量偏微分
    backward: function(data) {
      var i,j;
      var sum;
      for(i = 0; i < this.number; i++) {
        this.dz.$_(i, this.dz.$(i) * data.$(i));

        this.dw[i].$_(this.inputSize, this.dz.$(i));
        for(j = 0; j < this.inputSize; j++) {
          this.dw[i].$_(j,  this.in.$(j) * this.dz.$(i));
        }
      }

      for(i = 0; i < this.inputSize; i++) {
        sum = 0.0;
        for(j = 0; j < this.number; j++) {
          sum = sum + this.dz.$(j) * this.w[j].$(i);
        }
        this.din.$_(i, sum);
      }

      return this.din;
    },

  };

  exports.FullConnectLayer = FullConnectLayer;

})((typeof module != 'undefined' && module.exports) || ecj )
