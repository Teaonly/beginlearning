"use strict";

// 网络参数
var opts = {};
opts.l2Weight = 0.00001;

opts.learningRate = 0.02;
opts.epsilon = 0.0001;
opts.trainer = "adadelta";

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
opt.actType = "relu";
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

/*
// dropout 层
opt = {};
opt.type = "dropout";
opt.sx = 16;
opt.sy = 16;
opt.depth = 5;
opt.dropProb = 0.5;
opts.layers.push(opt);
*/


// fc 层
opt = {};
opt.type = "full";
opt.actType = "sigmoid";
opt.inputSize = 16*16*5;
opt.number = 1;
opts.layers.push(opt);

// 输出层
opt = {};
opt.type = "regression";
opt.inputSize = 1;
opts.layers.push(opt);

var ConvolutionNetwork = require("../src/ecj/ecj_network.js").ConvolutionNetwork;
var convNet = new ConvolutionNetwork(opts);


// 装载图像数据
var imageBatch = require("./32x32x3_4.js").imageBatch;
var i, j;
var samples = [];
var labels = [];
var batch = [];
for(i = 0; i < imageBatch.length; i++) {
  for(j = 0; j < imageBatch[i].data.length; j++) {
    imageBatch[i].data[j] = (imageBatch[i].data[j] - 128) / 128;
  }
  samples.push ( imageBatch[i].data );
  labels.push(imageBatch[i].label);
  batch.push(i);
}

for(var t = 0; t < 1000; t++) {
  var loss = convNet.train(samples, labels, batch);
  console.log(">>>>>>>>>>> " + loss.nnLoss);
}


