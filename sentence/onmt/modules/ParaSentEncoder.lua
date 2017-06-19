requrie('nngraph')


local ParaSentEncoder, parent = torch.class('onmt.ParaSentEncoder', 'nn.Container')


function ParaSentEncoder:__init(input, rnn, merge)
  parent.__init(self)

  self.args = {}
  self.args.merge = merge
  self.args.numEffectiveLayers = rnn.numEffectiveLayers
  self.args.rnnSize = rnn.outputSize * 2
  self.args.hiddenSize = self.args.rnnSize

  self.paraEncoder = onmt.BiEncoder.new(input, rnn, merge)
  self.sentEncoder = onmt.BiEncoder.new(input, rnn, merge)

  self:add(self.paraEncoder)
  self:add(self.sentEncoder)
end


function ParaSentEncoder.load(pretrained)
  local self = torch.factory('onmt.ParaSentEncoder')()

  parent.__init(self)

  self.paraEncoder = onmt.BiEncoder.load(pretrained.modules[1])
  self.sentEncoder = onmt.BiEncoder.load(pretrained.modules[2])
  self.args = pretrained.args

  self:add(self.paraEncoder)
  self:add(self.sentEncoder)

  return self
end

--[[ Return data to serialize. ]]
function ParaSentEncoder:serialize()
  local modulesData = {}
  for i = 1, #self.modules do
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
  local paraStates, paraContext = self.paraEncoder:forward(batch) -- [batchSize, dim], [batchSize, n, dim]
  local sentStates, sentContext = self.sentEncoder:forward(batch) -- [batchSize, dim], [batchSize, n, dim]
  local states = torch.cat({paraStates, sentStates}, 2)
  local context = torch.cat({paraContext, sentContext}, 3)
  return states, context
end


function ParaSentEncoder:backward(batch, gradStatesOutput, gradContextOutput)
  local gradStatesOutput = gradStatesOutput:chunk(2, 2)
  local gradContextOutput = gradContextOutput:chunk(2, 3)

  local gradParaStatesOutput = gradStatesOutput[1]
  local gradSentStatesOutput = gradStatesOutput[2]
  local gradParaContextOutput = gradContextOutput[1]
  local gradSentContextOutput = gradContextOutput[2]

  local gradParaInput = self.paraEncoder:backward(batch, gradParaStatesOutput, gradParaContextOutput)
  local gradSentInput = self.sentEncoder:backward(batch, gradSentStatesOutput, gradSentContextOutput)

  gradParaInput:add(gradSentInput)
  return gradParaInput
end
