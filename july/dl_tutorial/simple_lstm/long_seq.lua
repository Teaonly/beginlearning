require 'torch'
require 'nn'
require 'nngraph'
require "gnuplot"

local BOX_SIZE = 4
local RNN_SIZE = 196
local LAYER_NUMBER = 2

local box = {}
for i = 1, BOX_SIZE do
  box[i] = {};
  for j = 1, BOX_SIZE do
    box[i][j] = (i-1)*BOX_SIZE + j
  end
end
box[BOX_SIZE][BOX_SIZE] = 0

local model = torch.load('./model.bin');
model:evaluate();

local rnn_state = {}
for i = 1, LAYER_NUMBER*2 do
  rnn_state[i] = torch.ones(1, RNN_SIZE) * 0.05
end


local step = -1
local generateMove = function() 
  step = step + 1
 
  if ( step == 0) then
    return 'U'
  end

  if ( step == 1) then
    return 'L'
  end

  if ( step % 2 == 0) then
    return 'L';
  end

  return 'R';
end


local plotData = {}
plotData.value = {};
plotData.w2 = {};
plotData.x = {};

local x = BOX_SIZE
local y = BOX_SIZE
for i = 1, 500 do

  local nx = x
  local ny = y
  local move = -1
  local m = generateMove();

  if ( m == 'L' or m == 'l') then
    move = 2
  elseif ( m == 'R' or m == 'r') then
    move = 0
  elseif ( m == 'U' or m == 'u') then
    move = 3
  elseif ( m == 'D' or m == 'd') then
    move = 1
  end

  if ( move == 0) then
    nx = nx + 1
  elseif ( move == 1) then
    ny = ny + 1
  elseif ( move == 2) then
    nx = nx - 1
  elseif ( move == 3) then
    ny = ny - 1
  end

  if ( move >= 0 and nx >= 1 and nx <=BOX_SIZE and ny >=1 and ny <= BOX_SIZE) then
    local xx = torch.zeros(1, 4)
    xx[1][ move + 1] = 1

    local lst = model:forward({xx, unpack(rnn_state)})
    for j = 1, #rnn_state do
      rnn_state[j] = lst[j]
    end

    local value, prediction = lst[#lst]:max(2)
    local total_w2 = 0
    for j = 1, #rnn_state do
      total_w2 = total_w2 + torch.dot(rnn_state[j], rnn_state[j]);
    end

    print("step " .. step  .. " value = " .. torch.exp(value[1][1])  .. " W2 = " .. total_w2   .."  ==> " .. prediction[1][1])
    plotData.x[ #plotData.x + 1 ] = step
    plotData.w2[ #plotData.w2 + 1 ] = total_w2
    plotData.value[ #plotData.value + 1] = torch.exp(value[1][1])

    box[y][x] = box[ny][nx]
    box[ny][nx] = 0
    x = nx
    y = ny
  end
end

plotData.x = torch.Tensor(plotData.x);
plotData.value = torch.Tensor(plotData.value);
plotData.w2 = torch.Tensor(plotData.w2) / 1000000;

gnuplot.figure(1)
gnuplot.title('有限步骤的LSTM 运行')

gnuplot.plot(
  {'输出置信度' , plotData.x, plotData.value, '~'},
  {'内部状态L2值 (10^6)' , plotData.x, plotData.w2, '~'}
)


