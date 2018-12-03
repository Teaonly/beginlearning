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

  var ConvolutionLayer = function(opt) {
    this.type = "conv";
    this.actType = opt.actType;
    this.number = opt.number;

    this.sx = opt.sx;
    this.sy = opt.sx;         // 滤波器尺寸是正方形的
    this.depth = opt.depth;
    this.pad = opt.pad;
    this.stride = opt.stride;

    this.act = act.getActivation(this.actType);

    // input information
    this.inputSx = opt.inputSx;
    this.inputSy = opt.inputSy;
    this.inputDepth = opt.depth;
    this.outputSx = (this.inputSx + 2*this.pad - this.sx) / this.stride + 1;
    this.outputSy = (this.inputSy + 2*this.pad - this.sx) / this.stride + 1;

    if ( (this.outputSx % 1) !== 0) {
      throw new Error().stack;
    }

    // 内存分配
    this.w = [];
    this.dw = [];
    var i,j;
    for(i = 0; i < this.number; i++) {
      this.w.push( new Volume(this.sx, this.sy, this.depth, 1) );
      this.dw.push( new Volume(this.sx, this.sy, this.depth, 1) );
    }

    this.dz = new Volume(this.outputSx, this.outputSy, this.number);
    this.out = new Volume(this.outputSx, this.outputSy, this.number);

    this.in = new Volume(this.inputSx+2*this.pad, this.inputSy+2*this.pad, this.inputDepth);
    this.din = new Volume(this.inputSx, this.inputSy, this.inputDepth);

    // 构造反向的映射，方便计算
    var x,y,m,n;
    this.backMap = new Array( (this.inputSx+2*this.pad) * (this.inputSy+2*this.pad) );

    for(m = 0; m < this.outputSy; m++) {
      for(n = 0; n < this.outputSx; n++) {
        for(y = m * this.stride;  y < m*this.stride + this.sy; y++) {
          for(x = n * this.stride; x < n*this.stride + this.sx; x++) {
            if ( this.backMap[x + y * (this.inputSx + 2 * this.pad)] === undefined) {
              this.backMap[x + y * (this.inputSx + 2 * this.pad)] = [];
            }
            var in2out = {};
            in2out.outx = n;
            in2out.outy = m;
            in2out.fx = (x - n*this.stride);
            in2out.fy = (y - m*this.stride);
            this.backMap[x + y * (this.inputSx + 2 * this.pad)].push(in2out);
          }
        }
      }
    }

  };

  ConvolutionLayer.prototype = {
    init: function() {
      var i,j;
      var std = Math.sqrt(1/(this.sx*this.sy*this.depth)) ;
      for(i = 0; i < this.number; i++) {
        this.w[i].randnorm(0, std);
        this.w[i].$_( this.w[i].size -1, 0);    //初始化bias = 0;
      }
    },

    forward: function(data, isTrain) {
      var i,j,m,n;
      var x,y,d;
      var z;

      // 扩边操作，另外保存内部Volume格式
      j = 0;
      this.in.reset();
      for(y = 0; y < this.inputSy; y++) {
        for(x = 0; x < this.inputSx; x++) {
          for(d = 0; d < this.inputDepth; d++) {
            this.in.$$_(x+this.pad, y+this.pad, d, data.$(j));
            j++;
          }
        }
      }

      // do convolution with input
      for(i = 0; i < this.number; i++) {
        // loop from output
        for(m = 0; m < this.outputSy; m++) {
          for(n = 0; n < this.outputSx; n++) {
            z = this.w[i].$(this.w[i].size-1);    // this is bias
            for(y = m * this.stride;  y < m*this.stride + this.sy; y++) {
              for(x = n * this.stride; x < n*this.stride + this.sx; x++) {
                for(d = 0; d < this.inputDepth; d++) {
                  z = z + this.w[i].$$(x-n*this.stride, y-m*this.stride, d) * this.in.$$(x, y, d);
                }
              }
            }
            var out = this.act.forward(z);
            this.out.$$_(n, m, i, out);
            if ( isTrain) {
              this.dz.$$_(n, m, i, this.act.backward(z,out));   // just z'
            }
          }
        }
      }

      return this.out;
    },

    backward: function(data) {
      var i,j,m,n;
      var sum;
      var x,y,d;

      // 计算dz = dy * z';
      // Volume格式转换，一维变三维
      j = 0;
      for(m = 0; m < this.outputSy; m++) {
        for(n = 0; n < this.outputSx; n++) {
          for(i = 0; i < this.number; i++) {
            this.dz.$$_(n,m,i, this.dz.$$(n,m,i) * data.$(j) );
            j++;
          }
        }
      }

      for(i = 0; i < this.number; i++) {
        // 计算偏置的导数
        sum = 0.0;
        for(m = 0; m < this.outputSy; m++) {
          for(n = 0; n < this.outputSx; n++) {
            sum += this.dz.$$(n, m, i);
          }
        }
        this.dw[i].$_(this.dw[i].size - 1, sum);

        // 计算weight的导数
        for(y = 0; y < this.sy; y++) {
          for(x = 0; x < this.sx; x++) {
            for(d = 0; d < this.depth; d++) {
              sum = 0.0;
              for(m = 0; m < this.outputSy; m++) {
                for(n = 0; n < this.outputSx; n++) {
                  sum += this.dz.$$(n,m,i) * this.in.$$(n*this.stride + x, m * this.stride + y, d);
                }
              }
              this.dw[i].$$_(x,y,d,sum);
            }
          }
        }
      }

      // 跨神经元计算din
      for(y = 0; y < this.inputSy; y++) {
        for(x = 0; x < this.inputSx; x++) {
          var di = x + this.pad + (y + this.pad) * (this.inputSx + 2*this.pad);
          for(d = 0; d < this.depth; d++) {
            sum = 0.0;
            for(i = 0; i < this.number; i++) {
              for(var p = 0; p < this.backMap[di].length; p++) {
                var dz = this.dz.$$( this.backMap[di][p].outx, this.backMap[di][p].outy, i);
                var w = this.w[i].$$( this.backMap[di][p].fx, this.backMap[di][p].fy, d);
                sum += dz * w;
              }
            }
            this.din.$$_(x,y,d,sum);
          }
        }
      }

      return this.din;
    },

  };

  exports.ConvolutionLayer = ConvolutionLayer;

})((typeof module != 'undefined' && module.exports) || ecj )
