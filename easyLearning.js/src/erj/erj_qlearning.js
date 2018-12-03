(function(exports) {
  "use strict";

  var ConvolutionNetwork;
  
  if ( typeof window === 'undefined' ) {
    ConvolutionNetwork = require("../ecj/ecj_network.js").ConvolutionNetwork;
  } else {
    ConvolutionNetwork =  ecj.ConvolutionNetwork;
  }

  var QLearnAgent = function(opt) {

    // gamma 价值函数贴现率
    if ( opt.gamma === undefined ) {
      this.gamma = 0.75;
    } else {
      this.gamma = opt.gamma;
    }
    // epsilon 随机探索率
    if ( opt.epsilon === undefined ) {
      this.epsilon = 0.75;
    } else {
      this.epsilon = opt.epsilon;
    }
    // replay memory size 
    if ( opt.replaySize === undefined) {
      this.replaySize = 4096;
    } else {
      this.replaySize = opt.replaySize;
    }
    // update batch size
    if ( opt.batchSize === undefined) {
      this.batchSize = 32;
    } else {
      this.batchSize = opt.batchSize;
    }

    // 基于神经网络的模型参数
    if ( opt.inputSize === undefined 
        || opt.hiddenSize === undefined
        || opt.actionNumber === undefined) {
      throw new Error().stack;
    }
    this.actionNumber = opt.actionNumber;

    var netOpts = {};
    netOpts.epsilon = 0.0001;
    netOpts.trainer = "adadelta";
    netOpts.layers = [];
    netOpts.layers.push({sx:1,sy:1,depth:opts.inputSize});
    netOpts.layers.push({type:'full', actType:'relu', number:opt.hiddenSize, inputSize:opt.inputSize});
    netOpts.layers.push({type:'full', actType:'relu', number:opt.actionNumber, inputSize:opt.hiddenSize});
    netOpts.layers.push({type:'l2regression', inputSize:opt.actionNumber});
    this.net = new ConvolutionNetwork(netOpts);

    this.r0 = null;
    this.a0 = -1;
    this.s0 = null;
    this.a1 = -1;
    this.s1 = null;

    this.replayMemory = new Array(this.replaySize);
    this.replayIndex = 0;
  };

  QLearnAgent.prototype = {
   act: function(newState, isReset) {
      var act = -1;
      if(Math.random() < this.epsilon) { 
        act = Math.floor( Math.random() * this.actionNumber );
        act = act % this.actionNumber;
      } else {
        var actScores = this.forward(newState, false);
        act = actScores.maxIndex();
      }

      this.s0 = this.s1;
      this.a0 = this.a1;
      this.s1 = newState;
      this.a1 = act;

      if ( isReset ) {
        this.r0 = null;
      }

      return act;
    },

    // 目标模型：Q(s) = r + gamma * max_a' Q(s',a')
    // 更新方法：
    //    y = r + gamma * max_a' Q(s', a');
    //    min (y - Q(s))^2
    learn: function(r1) {
      if ( this.r0 === null ) {
        this.r0 = r1;
        return;
      }

      var replay = {};
      replay.r0 = this.r0;
      replay.a0 = this.a0;
      replay.s0 = this.s0;
      replay.s1 = this.s1;
      this.replayMemory[this.replayIndex] = replay;

      var batch = [];
      batch.push(this.replayIndex);
      for(var i = 1; i < this.batchSize; i++) {
        var ri = Math.floor( Math.random() * this.batchSize);
        ri = ri % this.batchSize;
        if ( this.replayMemory[ri] !== undefined ) {
          batch.push(ri);
        }
      }
      var loss = this._learnFromReplay(batch);

      // update for next learning
      this.replayIndex = (this.replayIndex+1) % this.replaySize;
      this.r0 = r1;

      return loss;
    },

    _learnFromReplay: function(batch) {
      var batchInput = [];
      var batchLabel = [];

      for(var i = 0; i < batch.length; i++) {

        var actScores = this._forwardQ(this.replayMemory[ batch[i] ].s1, false);
        var y = this.replayMemory[ batch[i] ].r0 + this.gamma * actScores.maxValue();

        actScores = this._forwardQ(this.replayMemory[ batch[i] ].s0, false);
        var label = [];
        for(var j = 0; j < this.actionNumber; j++) {
          label.push(actScores.$(j));
        }
        label[ this.replayMemory[ batch[i] ].a0 ] = y;
        batchInput.push( this.replayMemory[ batch[i] ].s0 );
        batchLabel.push( label );

        batch[i] = i;
      }

      var loss = this.net.train(batchInput, batchLabel, batch);
      return loss;
    },

    _forwardQ: function(newState, isTrain) {
      var data, i, v;

      if (isTrain === undefined) {
        isTrain = false;
      }

      data = this.net.inputLayer.forward(newState);
      for(i = 0; i < this.net.hiddenLayers.length; i++) {
        data = this.net.hiddenLayers[i].forward(data, isTrain);
      }

      v = this.net.outputLayer.forward(data);

      return v;
    },

 
  };

  exports.QLearnAgent = QLearnAgent;

})((typeof module != 'undefined' && module.exports) || erj )
