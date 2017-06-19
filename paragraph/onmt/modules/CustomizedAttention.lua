require('nngraph')

--[[ Global attention takes a matrix and a query vector. It
then computes a parameterized convex combination of the matrix
based on the input query.


    H_1 H_2 H_3 ... H_n
     q   q   q       q
      |  |   |       |
       \ |   |      /
           .....
         \   |  /
             a

Constructs a unit mapping:
  $$(H_1 .. H_n, q) => (a)$$
  Where H is of `batch x n x h_dim` and q is of `batch x q_dim`.

  The full function is  $$\tanh(W_2 [(softmax((W_1 q + b_1) H) H), q] + b_2)$$.

--]]
local CustomizedAttention, parent = torch.class('onmt.CustomizedAttention', 'nn.Container')

--[[A nn-style module computing attention.

  Parameters:

  * `h_dim` - dimension of the context vectors.
  * `q_dim` - dimension of the query vectors.
--]]
function CustomizedAttention:__init(h_dim, q_dim)
  parent.__init(self)
  self.net = self:_buildModel(h_dim, q_dim)
  self:add(self.net)
end

function CustomizedAttention:_buildModel(h_dim, q_dim)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- batchL x q_dim
  table.insert(inputs, nn.Identity()()) -- batchL x sourceTimesteps x h_dim

  local targetT = nn.Linear(q_dim, h_dim, false)(inputs[1]) -- batchL x h_dim
  local context = inputs[2] -- batchL x sourceTimesteps x h_dim

  -- Get attention.
  local attn = nn.MM()({context, nn.Replicate(1,3)(targetT)}) -- batchL x sourceL x 1
  attn = nn.Sum(3)(attn)
  local softmaxAttn = nn.SoftMax()
  softmaxAttn.name = 'softmaxAttn'
  attn = softmaxAttn(attn)
  attn = nn.Replicate(1,2)(attn) -- batchL x 1 x sourceL

  -- Apply attention to context.
  local contextCombined = nn.MM()({attn, context}) -- batchL x 1 x h_dim
  contextCombined = nn.Sum(2)(contextCombined) -- batchL x h_dim
  contextCombined = nn.JoinTable(2)({contextCombined, inputs[1]}) -- batchL x (h_dim + q_dim)
  local contextOutput = nn.Tanh()(nn.Linear(h_dim + q_dim, q_dim, false)(contextCombined))

  return nn.gModule(inputs, {contextOutput})
end

function CustomizedAttention:updateOutput(input)
  self.output = self.net:updateOutput(input)
  return self.output
end

function CustomizedAttention:updateGradInput(input, gradOutput)
  self.gradInput = self.net:updateGradInput(input, gradOutput)
  return self.gradInput
end

function CustomizedAttention:accGradParameters(input, gradOutput, scale)
  return self.net:accGradParameters(input, gradOutput, scale)
end
