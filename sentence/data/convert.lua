npy4th = require 'npy4th'

-- 6B.300d
--array = npy4th.loadnpy("qg.src.6B.300d.npy")
--print(array:size())
--torch.save("qg.src.6B.300d.t7", array)
--
--array = npy4th.loadnpy("qg.tgt.6B.300d.npy")
--print(array:size())
--torch.save("qg.tgt.6B.300d.t7", array)
--
---- 6B.100d
--array = npy4th.loadnpy("qg.src.6B.100d.npy")
--print(array:size())
--torch.save("qg.src.6B.100d.t7", array)
--
--array = npy4th.loadnpy("qg.tgt.6B.100d.npy")
--print(array:size())
--torch.save("qg.tgt.6B.100d.t7", array)

-- 840B.300d
array = npy4th.loadnpy("qg.src.840B.300d.npy")
print(array:size())
torch.save("./embs/qg.src.840B.300d.t7", array)

array = npy4th.loadnpy("qg.tgt.840B.300d.npy")
print(array:size())
torch.save("./embs/qg.tgt.840B.300d.t7", array)

