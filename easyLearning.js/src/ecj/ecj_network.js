(function(exports) {
  "use strict";

  var vol, Volume, InputLayer, MaxPoolingLayer, NormPoolingLayer;
  var ConvolutionLayer, FullConnectLayer, DropoutLayer;
  var L1RegressionLayer, L2RegressionLayer, LogisticRegressionLayer;
  var SoftmaxLayer, MaxMarginLayer, HingelossLayer;
  var SimpleSGDTrainer, AdaDeltaTrainer;

  if ( exports.vol  === undefined ) {
    vol = require('./ecj_volume.js').vol;
    Volume = require('./ecj_volume.js').Volume;
    InputLayer = require('./ecj_input.js').InputLayer;
    FullConnectLayer = require('./ecj_fullconnect.js').FullConnectLayer;
    ConvolutionLayer = require('./ecj_convolution.js').ConvolutionLayer;
    L1RegressionLayer = require('./ecj_regression.js').L1RegressionLayer;
    L2RegressionLayer = require('./ecj_regression.js').L2RegressionLayer;
    LogisticRegressionLayer = require('./ecj_regression.js').LogisticRegressionLayer;
    MaxPoolingLayer = require('./ecj_pool.js').MaxPoolingLayer;
    NormPoolingLayer = require('./ecj_pool.js').NormPoolingLayer;
    SoftmaxLayer = require('./ecj_softmax.js').SoftmaxLayer;
    MaxMarginLayer = require('./ecj_maxmargin.js').MaxMarginLayer;
    HingelossLayer = require('./ecj_maxmargin.js').HingelossLayer;
    DropoutLayer = require('./ecj_dropout.js').DropoutLayer;
    SimpleSGDTrainer = require('./ecj_trainer.js').SimpleSGDTrainer;
    AdaDeltaTrainer = require('./ecj_trainer.js').AdaDeltaTrainer;
  } else {
    vol = exports.vol;
    Volume = exports.Volume;
    InputLayer = exports.InputLayer;
    FullConnectLayer = exports.FullConnectLayer;
    ConvolutionLayer = exports.ConvolutionLayer;
    L1RegressionLayer = exports.L1RegressionLayer;
    L2RegressionLayer = exports.L2RegressionLayer;
    LogisticRegressionLayer = exports.LogisticRegressionLayer;
    MaxPoolingLayer = exports.MaxPoolingLayer;
    NormPoolingLayer = exports.NormPoolingLayer;
    SoftmaxLayer = exports.SoftmaxLayer;
    MaxMarginLayer = exports.MaxMarginLayer;
    HingelossLayer = exports.HingelossLayer;
    DropoutLayer = exports.DropoutLayer;
    SimpleSGDTrainer = exports.SimpleSGDTrainer;
    AdaDeltaTrainer = exports.AdaDeltaTrainer;
  }

  var ConvolutionNetwork = function(opts) {
    var i,j;
    var opt;

    this.vol = vol;

    // 网络顶层参数
    if ( opts.l2Weight !== undefined ) {
      this.l2Weight = opts.l2Weight;
    } else {
      this.l2Weight = 0;
    }
    if ( opts.l1Weight !== undefined) {
      this.l1Weight = opts.l1Weight;
    } else {
      this.l1Weight = 0;
    }

    // 构造网络
    this.inputLayer = null;
    this.hiddenLayers = [];
    this.outputLayer = null;
    if (opts.layers.length < 3) {
      throw new Error().stack;
    }

    vol.begin();
    opt = opts.layers[0];
    this.inputLayer = new InputLayer(opt);
    for(i = 1; i < opts.layers.length - 1; i++) {
      opt = opts.layers[i];
      if ( opt.type === "full") {
        this.hiddenLayers.push(new FullConnectLayer(opt));
      } else if ( opt.type === "conv") {
        this.hiddenLayers.push(new ConvolutionLayer(opt));
      } else if ( opt.type === "normpool") {
        this.hiddenLayers.push(new NormPoolingLayer(opt));
      } else if ( opt.type === "maxpool") {
        this.hiddenLayers.push(new MaxPoolingLayer(opt));
      } else if ( opt.type === "dropout") {
        this.hiddenLayers.push(new DropoutLayer(opt));
      } else {
        throw new Error().stack;
      }
    }

    opt = opts.layers[opts.layers.length - 1];
    if ( opt.type === "l1regression") {
      this.outputLayer = new L1RegressionLayer(opt);
    } else if ( opt.type === "l2regression" || opt.type === "regression" ) {
      this.outputLayer = new L2RegressionLayer(opt);
    } else if ( opt.type === "logistic") {
      this.outputLayer = new LogisticRegressionLayer(opt);
    } else if ( opt.type === "softmax" ) {
      this.outputLayer = new SoftmaxLayer(opt);
    } else if ( opt.type === "maxmargin" ) {
      this.outputLayer = new MaxMarginLayer(opt);
    } else if ( opt.type === "hinge") {
      this.outputLayer = new HingelossLayer(opt);
    } else {
      throw new Error().stack;
    }

    var i,j,totalGradSize;
    totalGradSize = 0;
    for(i = 0; i < this.hiddenLayers.length; i++) {
      for(j = 0; j < this.hiddenLayers[i].dw.length; j++) {
        totalGradSize += this.hiddenLayers[i].dw[j].size;
      }
    }
    this.grad = new Volume(1, 1, totalGradSize);

    opts.gradNumber = totalGradSize;
    if (opts.trainer === undefined || opts.trainer === "simplesgd") {
      this.trainer = new SimpleSGDTrainer(opts);
    } else if ( opts.trainer === "adadelta") {
      this.trainer = new AdaDeltaTrainer(opts);
    } else {
      throw new Error().stack;
    }
    vol.end();

    // init every layer
    if ( this.inputLayer.init !== undefined) {
      this.inputLayer.init();
    }
    for(i = 0; i < this.hiddenLayers.length; i++) {
      if ( this.hiddenLayers[i].init !== undefined) {
        this.hiddenLayers[i].init();
      }
    }
    if ( this.outputLayer.init !== undefined) {
      this.outputLayer.init();
    }
    this.trainer.init();

  };

  ConvolutionNetwork.prototype = {
    forward: function(newSample, y) {
      var data, i, v;

      var isTrain = false;
      if ( y !== undefined ) {
        isTrain = true;
      }

      data = this.inputLayer.forward(newSample);
      for(i = 0; i < this.hiddenLayers.length; i++) {
        data = this.hiddenLayers[i].forward(data, isTrain);
      }

      v = this.outputLayer.forward(data);
      if ( isTrain ) {
        v = this.outputLayer.loss(y);
      }

      return v;
    },

    backward: function(y) {
      var data, i, j, k, n, v;

      data = this.outputLayer.backward(y);
      for(i = this.hiddenLayers.length - 1; i >= 0; i--) {
        data = this.hiddenLayers[i].backward(data);
      }

      // 累加梯度数据
      n = 0;
      for(i = 0; i < this.hiddenLayers.length; i++) {
        for(j = 0; j < this.hiddenLayers[i].dw.length; j++) {
          for(k = 0; k < this.hiddenLayers[i].dw[j].size; k++) {
            v = this.grad.$(n) + this.hiddenLayers[i].dw[j].$(k);
            this.grad.$_(n, v);
            n++;
          }
        }
      }
      return;
    },

    train: function(samples, labels, batch) {
      var i,j,k,n,v;
      var lossSum, l2decay, l1decay;

      this.grad.reset();
      lossSum = 0;
      for(i = 0; i < batch.length; i++) {
        n = batch[i];
        lossSum += this.forward(samples[n], labels[n]);
        this.backward(labels[n]);
      }
      lossSum /= batch.length;

      // 加上 L2/L1 项
      n = 0;
      l2decay = 0;
      l1decay = 0;
      for(i = 0; i < this.hiddenLayers.length; i++) {
        for(j = 0; j < this.hiddenLayers[i].w.length; j++) {
          for(k = 0; k < this.hiddenLayers[i].w[j].size; k++) {
            v = this.grad.$(n)/batch.length;
            if (k !== (this.hiddenLayers[i].w[j].size - 1)) {
              l2decay += 0.5 * this.hiddenLayers[i].w[j].$(k) * this.hiddenLayers[i].w[j].$(k);
              l1decay += Math.abs( this.hiddenLayers[i].w[j].$(k)  );
              // L2
              v = this.grad.$(n) + this.l2Weight * this.hiddenLayers[i].w[j].$(k);
              // L1
              if ( this.l1Weight > 0) {
                if ( this.hiddenLayers[i].w[j].$(k) > 0 ) {
                  v += this.l1Weight;
                } else {
                  v += -1 * this.l1Weight;
                }
              }
            }
            this.grad.$_(n, v);
            n++;
          }
        }
      }

      this.trainer.update(this);

      return {
        //'total' : lossSum + l2decay*this.l2Weight,
        'nnLoss': lossSum,
        'l1': l1decay,
        'l2': l2decay,
      };

    },


  };

  exports.ConvolutionNetwork = ConvolutionNetwork;

})((typeof module != 'undefined' && module.exports) || ecj )
