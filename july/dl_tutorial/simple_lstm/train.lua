require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

local toy_data = require 'toy_data'
local model_utils = require 'model_utils'
local LSTM = require 'LSTM'

local INPUT_SIZE = 4
local OUTPUT_SIZE = 15
local RNN_SIZE = 196
local LAYER_NUMBER = 2
local BATCH_SIZE = 16
local MAX_TIMING_STEP = 128
local GRAD_CLIP = 5.0

toyRNN = {};
toyRNN.model = LSTM.lstm(INPUT_SIZE, OUTPUT_SIZE, RNN_SIZE, LAYER_NUMBER)
toyRNN.criterion = nn.ClassNLLCriterion()

-- global var for training
params, grad_params = model_utils.combine_all_parameters(toyRNN.model)

-- init weights with simple uniform random 
params:uniform(-0.08, 0.08)

-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
for layer_idx = 1, LAYER_NUMBER do
  for _,node in ipairs(toyRNN.model.forwardnodes) do
    if node.data.annotations.name == "i2h_" .. layer_idx then
      print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
      -- the gates are, in order, i,f,o,g, so f is the 2nd block of weights
      node.data.module.bias[{{RNN_SIZE+1, 2*RNN_SIZE}}]:fill(1.0)
    end
  end
end


toyRNN.clone_models = model_utils.clone_many_times(toyRNN.model, MAX_TIMING_STEP)
toyRNN.clone_criterions = model_utils.clone_many_times(toyRNN.criterion, MAX_TIMING_STEP)

-- init internal state
cell_data = {}
for i = 1, LAYER_NUMBER*2 do
  cell_data[i] = torch.ones(BATCH_SIZE,  RNN_SIZE) * 0.05 
end

-- training function
local feval = function(x) 
  if ( x ~= params ) then
    params:copy(x)
  end
  grad_params:zero()

  ------------------ get minibatch -------------------
  
  local step_number = (torch.random() % (MAX_TIMING_STEP - 64)) + 64 
  --local step_number = MAX_TIMING_STEP
  
  local x, y = toy_data.get_batch(BATCH_SIZE, step_number)

  ------------------- forward pass -------------------
  local predictions = {}           -- softmax outputs
  local loss = 0

  local rnn_state = {};
  rnn_state[0] = {unpack(cell_data)}
  for t=1, step_number do
    toyRNN.clone_models[t]:training()

    local lst = toyRNN.clone_models[t]:forward{x[t], unpack(rnn_state[t-1])}
    rnn_state[t] = {}
    for i=1, #lst - 1 do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
    predictions[t] = lst[#lst] -- last element is the prediction
    loss = loss + toyRNN.clone_criterions[t]:forward(predictions[t], y[t])
  end
  loss = loss / step_number

  ------------------ backward pass -------------------
  local drnn_state = {[step_number] = {}};
  -- there is no loss on state at last step 
  for i = 1, LAYER_NUMBER*2 do
    drnn_state[step_number][i] = torch.zeros(BATCH_SIZE,  RNN_SIZE)
  end

  for t=step_number, 1, -1 do
    -- backprop through loss, and softmax/linear
    local doutput_t = toyRNN.clone_criterions[t]:backward(predictions[t], y[t])
    table.insert(drnn_state[t], doutput_t)
    local dlst = toyRNN.clone_models[t]:backward({x[t], unpack(rnn_state[t-1])}, drnn_state[t])
    drnn_state[t-1] = {}
    for k,v in pairs(dlst) do
      if k > 1 then -- k == 1 is gradient on x, which we dont need
        -- note we do k-1 because first item is dembeddings, and then follow the·
        -- derivatives of the state, starting at index 2. I know...
        drnn_state[t-1][k-1] = v
      end
    end
  end

  grad_params:clamp(-GRAD_CLIP, GRAD_CLIP)
  return loss, grad_params
end

local doTest = function() 
  toyRNN.model:evaluate();

  local step_number = MAX_TIMING_STEP
  local xx, yy = toy_data.singleSequence(step_number)
  local x = torch.Tensor(1, 4)

  local rnn_state = {}
  for i = 1, LAYER_NUMBER*2 do
    rnn_state[i] = torch.ones(1, RNN_SIZE) * 0.05
  end

  local score = 0
  for t=1, step_number do
    -- one hot input
    x[1]:zero()
    x[1][xx[t]+1] = 1

    local lst = toyRNN.model:forward({x, unpack(rnn_state)})
    for i = 1, #rnn_state do
      rnn_state[i] = lst[i]
    end

    local _, prediction = lst[#lst]:max(2)

    if ( prediction[1][1] == yy[t] ) then
      score = score + 1
    end

  end
  
  return score

end


local doTrain = function(num) 
  train_loss = {}
  local optim_state = {learningRate = 0.02, alpha = 0.95}

  local maxScore = 0

  for i = 1, num do
    local _, loss = optim.rmsprop(feval, params, optim_state)
    print('>>>Iterating ' .. i .. ' with loss = ' .. loss[1])

    if ( i % 500 == 0) then
       optim_state.learningRate = optim_state.learningRate * 0.99
    end

    if ( i % 100 == 0) then
      local totalScore = 0
      for  j = 1, 32 do
        totalScore = totalScore + doTest()
      end
      if ( totalScore > maxScore ) then
        torch.save("./model.bin", toyRNN.model);
        maxScore = totalScore
      end
      print(">>>>>>>>>>>>>>" .. maxScore/(32*MAX_TIMING_STEP) .. "  " .. totalScore / (32*MAX_TIMING_STEP));
    end
  end
end

doTrain(50000)
