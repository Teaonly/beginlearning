var fs = require("fs");
var vm = require("vm");
var csv = require('csv');

function execute(path) {
  var code = fs.readFileSync(path, 'utf-8');
  vm.runInThisContext(code, path);
};

var exec = require('child_process').exec;
var run = function (command, callback){
  exec(command, function(error, stdout, stderr){
    if ( callback !== undefined) {
      callback(stdout, stderr);
    }
  });
};

var trainParser = csv.parse({}, function(err, data){
  // 去掉第一行数据
  var trainSamples = [];
  for(var i = 1; i < data.length; i++) {
    sample = {};
    sample.survived = data[i][1];
    sample.pclass = data[i][2];
    sample.name = data[i][3];
    sample.sex = data[i][4];
    sample.age = data[i][5];
    sample.sibsp = data[i][6];
    sample.parch = data[i][7];
    sample.ticket = data[i][8];
    sample.fare = data[i][9];
    sample.cabin = data[i][10];
    sample.embarked = data[i][11];
    trainSamples.push(sample);
  }

  var trainData = 'var trainData_titanic = ' + JSON.stringify( trainSamples ) + ';\n';
  fs.writeFileSync('./trainData.js', trainData);
});

var testParser = csv.parse({}, function(err, data){
  var testSamples = [];
  
  for(var i = 1; i < data.length; i++) {
    sample = {};
    //sample.survived = data[i][1];
    sample.pclass = data[i][1];
    sample.name = data[i][2];
    sample.sex = data[i][3];
    sample.age = data[i][4];
    sample.sibsp = data[i][5];
    sample.parch = data[i][6];
    sample.ticket = data[i][7];
    sample.fare = data[i][8];
    sample.cabin = data[i][9];
    sample.embarked = data[i][10];
    testSamples.push(sample);
  }
  
  /*
  var resultParser = csv.parse({}, function(err, data){
    for(var i = 1; i < data.length; i++) {
      testSamples[i-1].survived = data[i][1];
    }

    var testData = 'var testData_titanic = ' + JSON.stringify(testSamples) + ';\n';
    fs.writeFileSync('./testData.js', testData);
  });
 
  fs.createReadStream('./result.csv').pipe(resultParser);
  */
  var testData = 'var testData_titanic = ' + JSON.stringify(testSamples) + ';\n';
  fs.writeFileSync('./testData.js', testData);
});


fs.createReadStream('./train.csv').pipe(trainParser);
fs.createReadStream('./test.csv').pipe(testParser);



