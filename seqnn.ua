MM ← ⊞(/+×)⊙⍉
Zeros ← ×0°△
Eye ← ⊞=.⇡
HSum ← ⍜⍉(¤/+)
OnesLike ← +1×0
# arr -> arr[even idxs] arr[odd idxs]
Unzip ← ⊃(▽|▽¬)⊢⍉⋯⇡⧻.

HeInit ← ÷√⇌⊢:-1×2∵⋅⚂⊸Zeros
HeInitb ← ÷√:HeInit⊂:1.
InitL ← ⊟:∩□⊃(HeInitb⇌⊢)HeInit
DReLU ← ≥0
# DReLU ← ×-:1. ÷:1+1ⁿ:e¯

PUb ← °□◇⊃⊢(↘1)
LinL ← ⊂□⊙:⍜∩⍉+°□⊢°□:MM:∩°□⊙PUb⊙PUb°□⊙:⊃⊢∘
ReL ← ⊂□◇×>0.⊢.
# ReL ← ⊂□÷:1+1ⁿ:e¯°□⊢.

# topo -> net
Net ← ≡InitL≡⇌◫2
# net xs -> ys net
Run ← ⊙◌LinL⍢(ReL LinL|>1⧻°□:⊙.)⊙.¤□:
# net xs ys lr -> ys net ys lr xs
Forward ← ⊙⊙⊙:⊙⊙:⊸Run
# net ys lr xs -> net xs ys lr
ResX ← ⊙:⊙⊙:
Eval ← ⊙◌PUb Run

# y ys net exp_ys ->  Lgrad ys net exp_ys
DMSE ← ÷⊢⇌△.-⊙:⊙⊙⊸:

# grad ys net -> bbgrads layerins
BBGrads ← (
  ⊙≡⊢⊙:                                        # pick out all W matricies
  ⊙⊙(DReLU Unzip)                              # unzip and calculate relu derivatives
  ⊙(⍉⊟)¤□⊙(↘¯1)                                # trash first layer since we don't need to calc gradient of input
  ⇌⊙◌⍢(⊂□ ×MM⍉:∩∩°□°□ ⊙°⊟⊙:⊓⊃⊢∘⊃(⊢|↘1)|>0⧻:⊙.) # scan with init val of the ouput gradient, each step: (grad @ W) * ReluGrad
)
# bbgrads layerins -> dw db
LeafGrads ← ≡(∩□⊃(MM:⍉|⋅≡HSum)∩°□°⊟)⍉⊟:
# grad ys net -> dw db ys net
Backward ← LeafGrads⊸BBGrads
# dw db ys net lr -> net ys
ApplyGrads ← ⇌+×⊙:⊙⊙:⍉⊟

# net xs ys lr -> net xs ys lr
BackProp ← ResX⊸ApplyGrads Backward DMSE PUb ⊙⇌Forward

SE ← /+ⁿ2-∩⊢⊙:
