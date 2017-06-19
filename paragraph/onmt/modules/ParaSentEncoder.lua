require('nngraph')


local ParaSentEncoder, parent = torch.class('onmt.ParaSentEncoder', 'nn.Container')


function ParaSentEncoder:__init(sentInput, paraInput, sentRnn, paraRnn, merge)
  parent.__init(self)

  self.args = {}

  self.paraEncoder = onmt.BiEncoder.new(paraInput, paraRnn, merge)
  self.sentEncoder = onmt.BiEncoder.new(sentInput, sentRnn, merge)
  self.joinTable = nn.JoinTable(2)

  self.args.rnnSize = self.paraEncoder.args.hiddenSize + self.sentEncoder.args.hiddenSize
  self.args.hiddenSize = self.args.rnnSize
  self.args.sentSize = self.sentEncoder.args.hiddenSize
  self.args.paraSize = self.paraEncoder.args.hiddenSize

  self:add(self.paraEncoder)
  self:add(self.sentEncoder)
  self:add(self.joinTable)
end


function ParaSentEncoder.load(pretrained)
  local self = torch.factory('onmt.ParaSentEncoder')()

  parent.__init(self)

  if pretrained.modules[3] == nil then
    pretrained.modules[3] = nn.JoinTable(2)
  end

  self.paraEncoder = onmt.BiEncoder.load(pretrained.modules[1])
  self.sentEncoder = onmt.BiEncoder.load(pretrained.modules[2])
  self.joinTable = pretrained.modules[3]
  self.args = pretrained.args

  self:add(self.paraEncoder)
  self:add(self.sentEncoder)
  self:add(self.joinTable)

  return self
end


--[[ Return data to serialize. ]]
function ParaSentEncoder:serialize()
  local modulesData = {}
  for i = 1, 2 do
    table.insert(modulesData, self.modules[i]:serialize())
  end

  return {
    modules = modulesData,
    args = self.args
  }
end


function ParaSentEncoder:maskPadding()
  self.paraEncoder:maskPadding()
  self.sentEncoder:maskPadding()
end


function ParaSentEncoder:forward(batch)
  local sentStates, sentContext = self.sentEncoder:forward(batch) -- numEffectiveLayers * [batchSize, s_dim], [batchSize, n, s_dim]
  batch.sourceLength, batch.prgrphLength = batch.prgrphLength, batch.sourceLength
  batch.sourceSize, batch.prgrphSize = batch.prgrphSize, batch.sourceSize
  batch.sourceInput, batch.prgrphInput = batch.prgrphInput, batch.sourceInput
  batch.sourceInputFeatures, batch.prgrphInputFeatures = batch.prgrphInputFeatures, batch.sourceInputFeatures
  batch.sourceInputRev, batch.prgrphInputRev = batch.prgrphInputRev, batch.sourceInputRev
  batch.sourceInputRevFeatures, batch.prgrphInputRevFeatures = batch.prgrphInputRevFeatures, batch.sourceInputRevFeatures

  local paraStates, paraContext = self.paraEncoder:forward(batch) -- numEffectiveLayers * [batchSize, p_dim], [batchSize, n, p_dim]
  batch.sourceLength, batch.prgrphLength = batch.prgrphLength, batch.sourceLength
  batch.sourceSize, batch.prgrphSize = batch.prgrphSize, batch.sourceSize
  batch.sourceInput, batch.prgrphInput = batch.prgrphInput, batch.sourceInput
  batch.sourceInputFeatures, batch.prgrphInputFeatures = batch.prgrphInputFeatures, batch.sourceInputFeatures
  batch.sourceInputRev, batch.prgrphInputRev = batch.prgrphInputRev, batch.sourceInputRev
  batch.sourceInputRevFeatures, batch.prgrphInputRevFeatures = batch.prgrphInputRevFeatures, batch.sourceInputRevFeatures

  local states = {}
  for i = 1, #paraStates do
    states[i] = self.joinTable:forward({paraStates[i], sentStates[i]})
  end
  local context = sentContext -- [batchSize, n, dim]
  self.paraContext = paraContext
  return states, context
end


function ParaSentEncoder:backward(batch, gradStatesOutput, gradContextOutput)
  local gradParaStatesOutput = {}
  local gradSentStatesOutput = {}

  for i = 1, #gradStatesOutput do
    gradParaStatesOutput[i] = gradStatesOutput[i]:narrow(2, 1, self.args.paraSize)
    gradSentStatesOutput[i] = gradStatesOutput[i]:narrow(2, self.args.paraSize + 1, self.args.sentSize)
  end

  local gradSentInput = self.sentEncoder:backward(batch, gradSentStatesOutput, gradContextOutput)
  batch.sourceLength, batch.prgrphLength = batch.prgrphLength, batch.sourceLength
  batch.sourceSize, batch.prgrphSize = batch.prgrphSize, batch.sourceSize
  batch.sourceInput, batch.prgrphInput = batch.prgrphInput, batch.sourceInput
  batch.sourceInputFeatures, batch.prgrphInputFeatures = batch.prgrphInputFeatures, batch.sourceInputFeatures
  batch.sourceInputRev, batch.prgrphInputRev = batch.prgrphInputRev, batch.sourceInputRev
  batch.sourceInputRevFeatures, batch.prgrphInputRevFeatures = batch.prgrphInputRevFeatures, batch.sourceInputRevFeatures

  gradContextOutput = self.paraContext:zero()

  local gradParaInput = self.paraEncoder:backward(batch, gradParaStatesOutput, gradContextOutput)
  batch.sourceLength, batch.prgrphLength = batch.prgrphLength, batch.sourceLength
  batch.sourceSize, batch.prgrphSize = batch.prgrphSize, batch.sourceSize
  batch.sourceInput, batch.prgrphInput = batch.prgrphInput, batch.sourceInput
  batch.sourceInputFeatures, batch.prgrphInputFeatures = batch.prgrphInputFeatures, batch.sourceInputFeatures
  batch.sourceInputRev, batch.prgrphInputRev = batch.prgrphInputRev, batch.sourceInputRev
  batch.sourceInputRevFeatures, batch.prgrphInputRevFeatures = batch.prgrphInputRevFeatures, batch.sourceInputRevFeatures

  local gradInputs = {}
  for i = 1, #gradInputs do
    gradInputs[i] = gradParaInput[i] + gradSentInput[i]
  end
  return gradParaInput
end
