"use strict";

var ConvolutionNetwork = require("../src/ecj/ecj_network.js").ConvolutionNetwork;

// 网络参数
var opts = {};
opts.learningRate = 0.001;
opts.l2Weight = 0.0001;
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
opt.actType = "bypass";
opt.inputSize = 4;
opt.number = 3;
opts.layers.push(opt);

// 输出层
opt = {};
opt.type = "softmax";
opt.inputSize = 3;
opts.layers.push(opt);

var convNet = new ConvolutionNetwork(opts);

var samples = [];
var labels = [];

var i;
for(i=0; i < 1024; i++) {
  var sample = new Float64Array(2);
  var y = new Array(1);

  sample[0] = -1 + 2 * Math.random();
  sample[1] = -1 + 2 * Math.random();

  if ( (Math.abs(sample[0]) < 0.001) || (Math.abs(sample[1]) < 0.001))
    continue;

  y[0] = 0;
  if ( sample[0] * sample[1] > 0) {
    y[0] = 1;
  } else if ( sample[0] < 0) {
    y[0] = 2;
  }

  samples.push(sample);
  labels.push(y);
}

// 前向进行Loss函数计算
var lossSum = 0;
var i, loss, grad;
convNet.grad.reset();
for (i = 0; i < 4; i++) {
  loss = convNet.forward(samples[i], labels[i]);
  lossSum += loss;
  convNet.backward(labels[i]);
}
console.log("Total loss is :" + lossSum + "  Grad = " + convNet.grad.$(0));

// 计算数值版本的梯度
convNet.hiddenLayers[0].w[0].$_(0, convNet.hiddenLayers[0].w[0].$(0) + 0.00001);
var lossSumR = 0.0;
for (i = 0; i < 4; i++) {
  loss = convNet.forward(samples[i], labels[i]);
  lossSumR = lossSumR + loss;
}
convNet.hiddenLayers[0].w[0].$_(0, convNet.hiddenLayers[0].w[0].$(0) - 0.00002);
var lossSumL = 0.0;
for (i = 0; i < 4; i++) {
  loss = convNet.forward(samples[i], labels[i]);
  lossSumL = lossSumL + loss;
}
var grad0 = (lossSumR - lossSumL) / 0.00002;

console.log(">>>>>>lossSumR = " + lossSumR + "  LossSumL = " + lossSumL  + "    Grad0 = " + grad0);

var t = 0;
var batch = [];
for(i = 0; i < 128; i++) {
  batch.push(0);
}

for(t = 0; t < 10000; t++) {
  for(i = 0; i < batch.length; i++) {
    var n = (t*batch.length + i) % samples.length;
    batch[i] = n;
  }
  loss = convNet.train(samples, labels, batch);
}


console.log("After 10000 iterators loss = " + loss.nnLoss);

