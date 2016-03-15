allSamples = {}

local guestureType = {'A', 'B', 'C', 'F', 'P', 'V'}

for i=1,400 do
  for j =1,#guestureType do
    local sample = {}
    sample.fileName = './guesture/' .. guestureType[j] .. '_' .. i .. '.png'
    sample.y = j

    allSamples[#allSamples+1] = sample
  end
end

for i=1, 10000 do
  local selected = math.floor( math.random() * #allSamples) + 1
  local temp = allSamples[selected]
  allSamples[selected] = allSamples[1]
  allSamples[1] = temp
end


