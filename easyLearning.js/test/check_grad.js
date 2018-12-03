"use strict";

// 网络参数
var opts = {};
opts.learningRate = 0.01;
opts.l2Weight = 0.00001;
opts.layers = [];

// 输入层
var opt = {};
opt.sx = 32;
opt.sy = 32;
opt.depth = 3;
opts.layers.push(opt);

// 卷积层
opt = {};
opt.type = "conv";
opt.actType = "sigmoid";
opt.number = 5;
opt.sx = 3;
opt.sy = 3;
opt.depth = 3;
opt.inputSx = 32;
opt.inputSy = 32;
opt.pad = 1;
opt.stride = 1;
opts.layers.push(opt);

// polling层
opt = {};
opt.type = "normpool";
opt.inputSx = 32;
opt.inputSy = 32;
opt.inputDepth = 5;
opts.layers.push(opt);

// fc 层
opt = {};
opt.type = "full";
opt.actType = "bypass";
opt.inputSize = 16*16*5;
opt.number = 1;
opts.layers.push(opt);

// 输出层
opt = {};
opt.type = "logistic";
opt.inputSize = 1;
opts.layers.push(opt);

var ConvolutionNetwork = require("../src/ecj/ecj_network.js").ConvolutionNetwork;
var convNet = new ConvolutionNetwork(opts);

// 装载图像数据
var imageBatch = require("./32x32x3_4.js").imageBatch;
var i, j;
for(i = 0; i < imageBatch.length; i++) {
  for(j = 0; j < imageBatch[i].data.length; j++) {
    imageBatch[i].data[j] = (imageBatch[i].data[j] - 128) / 128;
  }
}

// 前向进行Loss函数计算
var lossSum = 0;
var i, loss, grad;
convNet.grad.reset();
for (i = 0; i < imageBatch.length; i++) {
  loss = convNet.forward(imageBatch[i].data, imageBatch[i].label);
  lossSum += loss;
  convNet.backward(imageBatch[i].label);
}
console.log("Total loss is :" + lossSum + "  Grad = " + convNet.grad.$(113));


// 计算数值版本的梯度
convNet.hiddenLayers[0].w[4].$_(1, convNet.hiddenLayers[0].w[4].$(1) + 0.00001);
var lossSumR = 0.0;
for (i = 0; i < imageBatch.length; i++) {
  loss = convNet.forward(imageBatch[i].data, imageBatch[i].label);
  lossSumR = lossSumR + loss;
}
convNet.hiddenLayers[0].w[4].$_(1, convNet.hiddenLayers[0].w[4].$(1) - 0.00002);
var lossSumL = 0.0;
for (i = 0; i < imageBatch.length; i++) {
  loss = convNet.forward(imageBatch[i].data, imageBatch[i].label);
  lossSumL = lossSumL + loss;
}
var grad0 = (lossSumR - lossSumL) / 0.00002;

console.log(">>>>>>lossSumR = " + lossSumR + "  LossSumL = " + lossSumL  + "    Grad0 = " + grad0);
