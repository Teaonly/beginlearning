"use strict";

var ConvolutionNetwork = require("../src/ecj/ecj_network.js").ConvolutionNetwork;

// 网络参数
var opts = {};
opts.learningRate = 0.4;
opts.l2Weight = 0.00001;
opts.layers = [];

// 输入层
var opt = {};
opt.sx = 1;
opt.sy = 1;
opt.depth = 2;
opts.layers.push(opt);

// 隐藏层 1
opt = {};
opt.type = "full";
opt.actType = "sigmoid";
opt.inputSize = 2;
opt.number = 4;
opts.layers.push(opt);

// 隐藏层 2
opt = {};
opt.type = "full";
opt.actType = "sigmoid";
opt.inputSize = 4;
opt.number = 1;
opts.layers.push(opt);

// 输出层
opt = {};
opt.type = "regression";
opt.inputSize = 1;
opts.layers.push(opt);

var convNet = new ConvolutionNetwork(opts);

var samples = [];
var labels = [];

var i;
for(i=0; i < 1024; i++) {
  var sample = new Float64Array(2);
  var y = new Float64Array(1);

  sample[0] = -1 + 2 * Math.random();
  sample[1] = -1 + 2 * Math.random();

  if ( (Math.abs(sample[0]) < 0.001) || (Math.abs(sample[1]) < 0.001))
    continue;

  y[0] = 0;
  if ( sample[0] * sample[1] > 0) {
    y[0] = 1;
  }

  samples.push(sample);
  labels.push(y);
}

var t = 0;
var batch = [];
for(i = 0; i < 128; i++) {
  batch.push(0);
}

for(t = 0; t < 1000; t++) {
  for(i = 0; i < batch.length; i++) {
    var n = (t*batch.length + i) % samples.length;
    batch[i] = n;
  }

  var loss = convNet.train(samples, labels, batch);
  console.log(loss.nnLoss);
}

var score = 0;
for (i = 0; i < 500; i++) {

  var sample = new Float64Array(2);
  var y = new Float64Array(1);

  sample[0] = -1 + 2 * Math.random();
  sample[1] = -1 + 2 * Math.random();

  y[0] = 0;
  if ( sample[0] * sample[1] > 0) {
    y[0] = 1;
  }

 var v = convNet.forward(sample).$(0);

  if ( v > 0.5 && (y[0] > 0.5) ) {
    score ++;
  }  else if ( v < 0.5 && (y[0] < 0.5) ) {
    score ++;
  }
}
console.log(">>>>>>>>> score = " + score);
