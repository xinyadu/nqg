--[[ Return the maxLength, sizes, and non-zero count
  of a batch of `seq`s ignoring `ignore` words.
--]]
local function getLength(seq, ignore)
  local sizes = torch.IntTensor(#seq):zero()
  local max = 0
  local sum = 0

  for i = 1, #seq do
    local len = seq[i]:size(1)
    if ignore ~= nil then
      len = len - ignore
    end
    max = math.max(max, len)
    sum = sum + len
    sizes[i] = len
  end
  return max, sizes, sum
end

--[[ Data management and batch creation.

Batch interface reference [size]:

  * size: number of sentences in the batch [1]

  * sourceLength: max length in source batch [1]
  * sourceSize:  lengths of each source [batch x 1]
  * sourceInput:  left-padded idx's of source (PPPPPPABCDE) [batch x max]
  * sourceInputFeatures: table of source features sequences
  * sourceInputRev: right-padded  idx's of source rev (EDCBAPPPPPP) [batch x max]
  * sourceInputRevFeatures: table of reversed source features sequences

  * prgrphLength: max length in paragraph batch [1]
  * prgrphSize:  lengths of each paragraph [batch x 1]
  * prgrphInput:  left-padded idx's of paragraph (PPPPPPABCDE) [batch x max]
  * prgrphInputFeatures: table of paragraph features sequences
  * prgrphInputRev: right-padded  idx's of paragraph rev (EDCBAPPPPPP) [batch x max]
  * prgrphInputRevFeatures: table of reversed paragraph features sequences

  * targetLength: max length in target batch [1]
  * targetSize: lengths of each target [batch x 1]
  * targetNonZeros: number of non-ignored words in batch [1]
  * targetInput: input idx's of target (SABCDEPPPPPP) [batch x max]
  * targetInputFeatures: table of target input features sequences
  * targetOutput: expected output idx's of target (ABCDESPPPPPP) [batch x max]
  * targetOutputFeatures: table of target output features sequences

 TODO: change name of size => maxlen
--]]


--[[ A batch of sentences to translate and targets. Manages padding,
  features, and batch alignment (for efficiency).

  Used by the decoder and encoder objects.
--]]
local Batch = torch.class('Batch')

--[[ Create a batch object.

Parameters:

  * `src` - 2D table of source batch indices
  * `srcFeatures` - 2D table of source batch features (opt)
  * `tgt` - 2D table of target batch indices
  * `tgtFeatures` - 2D table of target batch features (opt)
--]]
function Batch:__init(src, srcFeatures, tgt, tgtFeatures, par, parFeatures)
  src = src or {}
  par = par or {}
  srcFeatures = srcFeatures or {}
  parFeatures = parFeatures or {}

  assert(#src == #par, "source and paragraph must have the same batch size")
  if tgt ~= nil then
    assert(#src == #tgt, "source and target must have the same batch size")
  end

  self.size = #src

  self.sourceLength, self.sourceSize = getLength(src)

  local sourceSeq = torch.IntTensor(self.sourceLength, self.size):fill(onmt.Constants.PAD)
  self.sourceInput = sourceSeq:clone()
  self.sourceInputRev = sourceSeq:clone()

  self.sourceInputFeatures = {}
  self.sourceInputRevFeatures = {}

  if #srcFeatures > 0 then
    for _ = 1, #srcFeatures[1] do
      table.insert(self.sourceInputFeatures, sourceSeq:clone())
      table.insert(self.sourceInputRevFeatures, sourceSeq:clone())
    end
  end

  self.prgrphLength, self.prgrphSize = getLength(par)

  local prgrphSeq = torch.IntTensor(self.prgrphLength, self.size):fill(onmt.Constants.PAD)
  self.prgrphInput = prgrphSeq:clone()
  self.prgrphInputRev = prgrphSeq:clone()

  self.prgrphInputFeatures = {}
  self.prgrphInputRevFeatures = {}

  if #parFeatures > 0 then
    for _ = 1, #parFeatures[1] do
      table.insert(self.prgrphInputFeatures, prgrphSeq:clone())
      table.insert(self.prgrphInputRevFeatures, prgrphSeq:clone())
    end
  end

  if tgt ~= nil then
    self.targetLength, self.targetSize, self.targetNonZeros = getLength(tgt, 1)

    local targetSeq = torch.IntTensor(self.targetLength, self.size):fill(onmt.Constants.PAD)
    self.targetInput = targetSeq:clone()
    self.targetOutput = targetSeq:clone()

    self.targetInputFeatures = {}
    self.targetOutputFeatures = {}

    if #tgtFeatures > 0 then
      for _ = 1, #tgtFeatures[1] do
        table.insert(self.targetInputFeatures, targetSeq:clone())
        table.insert(self.targetOutputFeatures, targetSeq:clone())
      end
    end
  end

  for b = 1, self.size do
    local sourceOffset = self.sourceLength - self.sourceSize[b] + 1
    local sourceInput = src[b]
    local sourceInputRev = src[b]:index(1, torch.linspace(self.sourceSize[b], 1, self.sourceSize[b]):long())

    -- Source input is left padded [PPPPPPABCDE] .
    self.sourceInput[{{sourceOffset, self.sourceLength}, b}]:copy(sourceInput)
    self.sourceInputPadLeft = true

    -- Rev source input is right padded [EDCBAPPPPPP] .
    self.sourceInputRev[{{1, self.sourceSize[b]}, b}]:copy(sourceInputRev)
    self.sourceInputRevPadLeft = false

    for i = 1, #self.sourceInputFeatures do
      local sourceInputFeatures = srcFeatures[b][i]
      local sourceInputRevFeatures = srcFeatures[b][i]:index(1, torch.linspace(self.sourceSize[b], 1, self.sourceSize[b]):long())

      self.sourceInputFeatures[i][{{sourceOffset, self.sourceLength}, b}]:copy(sourceInputFeatures)
      self.sourceInputRevFeatures[i][{{1, self.sourceSize[b]}, b}]:copy(sourceInputRevFeatures)
    end

    local prgrphOffset = self.prgrphLength - self.prgrphSize[b] + 1
    local prgrphInput = par[b]
    local prgrphInputRev = par[b]:index(1, torch.linspace(self.prgrphSize[b], 1, self.prgrphSize[b]):long())

    -- Paragraph input is left padded [PPPPPPABCDE] .
    self.prgrphInput[{{prgrphOffset, self.prgrphLength}, b}]:copy(prgrphInput)
    self.prgrphInputPadLeft = true

    -- Rev paragraph input is right padded [EDCBAPPPPPP] .
    self.prgrphInputRev[{{1, self.prgrphSize[b]}, b}]:copy(prgrphInputRev)
    self.prgrphInputRevPadLeft = false

    for i = 1, #self.prgrphInputFeatures do
      local prgrphInputFeatures = parFeatures[b][i]
      local prgrphInputRevFeatures = parFeatures[b][i]:index(1, torch.linspace(self.prgrphSize[b], 1, self.prgrphSize[b]):long())

      self.prgrphInputFeatures[i][{{prgrphOffset, self.prgrphLength}, b}]:copy(prgrphInputFeatures)
      self.prgrphInputRevFeatures[i][{{1, self.prgrphSize[b]}, b}]:copy(prgrphInputRevFeatures)
    end

    if tgt ~= nil then
      -- Input: [<s>ABCDE]
      -- Ouput: [ABCDE</s>]
      local targetLength = tgt[b]:size(1) - 1
      local targetInput = tgt[b]:narrow(1, 1, targetLength)
      local targetOutput = tgt[b]:narrow(1, 2, targetLength)

      -- Target is right padded [<S>ABCDEPPPPPP] .
      self.targetInput[{{1, targetLength}, b}]:copy(targetInput)
      self.targetOutput[{{1, targetLength}, b}]:copy(targetOutput)

      for i = 1, #self.targetInputFeatures do
        local targetInputFeatures = tgtFeatures[b][i]:narrow(1, 1, targetLength)
        local targetOutputFeatures = tgtFeatures[b][i]:narrow(1, 2, targetLength)

        self.targetInputFeatures[i][{{1, targetLength}, b}]:copy(targetInputFeatures)
        self.targetOutputFeatures[i][{{1, targetLength}, b}]:copy(targetOutputFeatures)
      end
    end
  end
end

local function addInputFeatures(inputs, featuresSeq, t)
  local features = {}
  for j = 1, #featuresSeq do
    table.insert(features, featuresSeq[j][t])
  end
  if #features > 1 then
    table.insert(inputs, features)
  else
    onmt.utils.Table.append(inputs, features)
  end
end

--[[ Get source input batch at timestep `t`. --]]
function Batch:getSourceInput(t)
  -- If a regular input, return word id, otherwise a table with features.
  if #self.sourceInputFeatures > 0 then
    local inputs = {self.sourceInput[t]}
    addInputFeatures(inputs, self.sourceInputFeatures, t)
    return inputs
  else
    return self.sourceInput[t]
  end
end

--[[ Get paragraph input batch at timestep `t`. --]]
function Batch:getPrgrphInput(t)
  -- If a regular input, return word id, otherwise a table with features.
  if #self.prgrphInputFeatures > 0 then
    local inputs = {self.prgrphInput[t]}
    addInputFeatures(inputs, self.prgrphInputFeatures, t)
    return inputs
  else
    return self.prgrphInput[t]
  end
end

--[[ Get target input batch at timestep `t`. --]]
function Batch:getTargetInput(t)
  -- If a regular input, return word id, otherwise a table with features.
  if #self.targetInputFeatures > 0 then
    local inputs = {self.targetInput[t]}
    addInputFeatures(inputs, self.targetInputFeatures, t)
    return inputs
  else
    return self.targetInput[t]
  end
end

--[[ Get target output batch at timestep `t` (values t+1). --]]
function Batch:getTargetOutput(t)
  -- If a regular input, return word id, otherwise a table with features.
  local outputs = { self.targetOutput[t] }
  for j = 1, #self.targetOutputFeatures do
    table.insert(outputs, self.targetOutputFeatures[j][t])
  end
  return outputs
end

return Batch
