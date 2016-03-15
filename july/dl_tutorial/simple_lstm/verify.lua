require 'torch'
require 'nn'
require 'nngraph'

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

local show_box = function()
  print("=================================")
  for y = 1,BOX_SIZE do
    local line = "";
    for x = 1,BOX_SIZE do
      if ( box[y][x] == 0) then
        line = line .. '  ' .. ' '
      elseif ( box[y][x] > 9) then
        line = line .. box[y][x] .. ' '
      else
        line = line .. ' ' .. box[y][x] .. ' '
      end
    end
    print(line)
  end
  print("----")
end

local model = torch.load('./model.bin');

local rnn_state = {}
for i = 1, LAYER_NUMBER*2 do
  rnn_state[i] = torch.ones(1, RNN_SIZE) * 0.05
end

local x = BOX_SIZE
local y = BOX_SIZE
while true do
  show_box();
  print("Please input L/R/U/D ");
  
  local nx = x
  local ny = y
  local move = -1
  local m = io.read()
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
    for i = 1, #rnn_state do
      rnn_state[i] = lst[i]
    end

    local _, prediction = lst[#lst]:max(2)
    print("model output ==> " .. prediction[1][1])

    box[y][x] = box[ny][nx]
    box[ny][nx] = 0
    x = nx
    y = ny
  end

end
