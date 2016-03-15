local BOX_SIZE = 4

local toy_data = {}
toy_data.singleSequence = function (step_number) 
  local box = {};
  for i = 1, BOX_SIZE do
    box[i] = {};
    for j = 1, BOX_SIZE do
      box[i][j] = (i-1)*BOX_SIZE + j
    end
  end
  local x = BOX_SIZE
  local y = BOX_SIZE
  box[BOX_SIZE][BOX_SIZE] = 0;
  
  local xx = {}
  local yy = {}

  local n = 1
  while(true) do
    if ( n > step_number) then
      break
    end

    local nx = x
    local ny = y
    local move = torch.random() % 4
    if ( move == 0) then
      nx = nx + 1
    elseif ( move == 1) then
      ny = ny + 1
    elseif ( move == 2) then
      nx = nx - 1
    elseif ( move == 3) then
      ny = ny - 1
    end

    if ( nx >= 1 and nx <=BOX_SIZE and ny >=1 and ny <= BOX_SIZE) then
      table.insert(xx, move)
      table.insert(yy, box[ny][nx])
      box[y][x] = box[ny][nx]
      box[ny][nx] = 0
      x = nx
      y = ny

      n = n + 1
    end
  end

  return xx, yy;
end

toy_data.get_batch = function(batch, step_number)
  local x = {}
  local y = {}

  for i = 1, step_number do
    x[i] = torch.zeros(batch, 4);
    y[i] = torch.zeros(batch);
  end

  for i = 1, batch do
    local xx, yy = toy_data.singleSequence(step_number)
    for j = 1, step_number do
      x[j][i][ xx[j] + 1] = 1
      y[j][i] = yy[j]
    end
  end
  return x, y;
end

return toy_data;
