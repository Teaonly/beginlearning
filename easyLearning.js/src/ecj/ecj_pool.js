(function(exports) {
  "use strict";

  var Volume;
  if ( exports.Volume !== undefined ) {
    Volume = exports.Volume;
  } else {
    Volume = require("./ecj_volume.js").Volume;
  }

  var MaxPoolingLayer = function(opt) {
    this.type = "maxpool";

    this.inputSx = opt.inputSx;
    this.inputSy = opt.inputSy;
    this.inputDepth = opt.inputDepth;

    if ( ((this.inputSx % 2) !== 0) || ((this.inputSy % 2) !== 0) ) {
      throw new Error().stack;
    }

    this.in = new Volume(this.inputSx, this.inputSy, this.inputDepth);
    this.din = this.in;
    this.out = new Volume(this.inputSx/2, this.inputSy/2, this.inputDepth);

    this.w = [];
    this.dw = [];

    this._pool = new Array(4);
  };

  MaxPoolingLayer.prototype = {
    forward: function(data, isTrain) {
      var x,y,d,i;
      var maxv, maxi;
      var pool = new Array(4);

      this.in.reset();
      i = 0;
      for(y = 0; y < this.inputSy; y++) {
        for(x = 0; x < this.inputSx; x++) {
          for(d = 0; d < this.inputDepth; d++) {
            this.in.$$_(x,y,d,data.$(i));
            i++;
          }
        }
      }

      for(d = 0; d < this.inputDepth; d++) {
        for(y = 0; y < this.inputSy; y+=2) {
          for(x = 0; x < this.inputSx; x+=2) {
            this._pool[0] = this.in.$$(x,y,d);
            this._pool[1] = this.in.$$(x+1,y,d);
            this._pool[2] = this.in.$$(x,y+1,d);
            this._pool[3] = this.in.$$(x+1,y+1,d);

            maxi = 0;
            maxv = this._pool[0];
            this._pool[0] = 0;
            for(i = 1; i < 4; i++) {
              if (this._pool[i] > maxv ) {
                maxi = i;
                maxv = this._pool[i];
              }
              this._pool[i] = 0;
            }
            this._pool[maxi] = 1;
            this.out.$$_(x/2, y/2, d, maxv);

            if ( isTrain) {
              this.in.$$_(x,y,d, this._pool[0]);
              this.in.$$_(x+1,y,d, this._pool[1]);
              this.in.$$_(x,y+1,d, this._pool[2]);
              this.in.$$_(x+1,y+1,d, this._pool[3]);
            }
          }
        }
      }
      return this.out;
    },

    backward: function(data) {
      var x,y,d,n;

      n = 0;
      for(y = 0; y < this.inputSy; y+=2) {
        for(x = 0; x < this.inputSx; x+=2) {
          for(d = 0; d < this.inputDepth; d++) {
            this.din.$$_(x,y, d, this.din.$$(x,y,d) * data.$(n));
            this.din.$$_(x+1,y, d, this.din.$$(x+1,y,d) * data.$(n));
            this.din.$$_(x,y+1, d, this.din.$$(x,y+1,d) * data.$(n));
            this.din.$$_(x+1,y+1, d, this.din.$$(x+1,y+1,d) * data.$(n));
            n++;
          }
        }
      }

      return this.din;
    },

  };

  var NormPoolingLayer = function(opt) {
    this.type = "normpool";

    this.inputSx = opt.inputSx;
    this.inputSy = opt.inputSy;
    this.inputDepth = opt.inputDepth;

    if ( ((this.inputSx % 2) !== 0) || ((this.inputSy % 2) !== 0) ) {
      throw new Error().stack;
    }

    this.in = new Volume(this.inputSx, this.inputSy, this.inputDepth);
    this.din = this.in;
    this.out = new Volume(this.inputSx/2, this.inputSy/2, this.inputDepth);

    this.w = [];
    this.dw = [];
  };

  NormPoolingLayer.prototype = {
    forward: function(data, isTrain) {
      var x,y,d,i;
      var sum;
      var pool = new Array(4);

      this.in.reset();
      i = 0;
      for(y = 0; y < this.inputSy; y++) {
        for(x = 0; x < this.inputSx; x++) {
          for(d = 0; d < this.inputDepth; d++) {
            this.in.$$_(x,y,d,data.$(i));
            i++;
          }
        }
      }

      for(d = 0; d < this.inputDepth; d++) {
        for(y = 0; y < this.inputSy; y+=2) {
          for(x = 0; x < this.inputSx; x+=2) {
            sum = 0.0;
            sum += this.in.$$(x,y,d);
            sum += this.in.$$(x+1,y,d);
            sum += this.in.$$(x,y+1,d);
            sum += this.in.$$(x+1,y+1,d);

            this.out.$$_(x/2, y/2, d, sum/4);
            if ( isTrain) {
              this.in.$$_(x,y,d, 0.25);
              this.in.$$_(x+1,y,d, 0.25);
              this.in.$$_(x,y+1,d, 0.25);
              this.in.$$_(x+1,y+1,d, 0.25);
            }
          }
        }
      }

      return this.out;
    },

    backward: function(data) {
      var x,y,d,n;

      n = 0;
      for(y = 0; y < this.inputSy; y+=2) {
        for(x = 0; x < this.inputSx; x+=2) {
          for(d = 0; d < this.inputDepth; d++) {
            this.din.$$_(x,y, d, this.din.$$(x,y,d) * data.$(n));
            this.din.$$_(x+1,y, d, this.din.$$(x+1,y,d) * data.$(n));
            this.din.$$_(x,y+1, d, this.din.$$(x,y+1,d) * data.$(n));
            this.din.$$_(x+1,y+1, d, this.din.$$(x+1,y+1,d) * data.$(n));
            n++;
          }
        }
      }

      return this.din;
    },
  };


  exports.MaxPoolingLayer = MaxPoolingLayer;
  exports.NormPoolingLayer = NormPoolingLayer;

})((typeof module != 'undefined' && module.exports) || ecj )
