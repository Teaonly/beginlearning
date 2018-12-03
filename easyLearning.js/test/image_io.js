"use strict";

var util = require("../src/ecj/ecj_util.js").util;

var buffer = undefined;
util.readPNG('./trees.png', buffer, function(buf) {
  buffer = buf;
  console.log(buffer.length);
});
