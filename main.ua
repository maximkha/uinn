~ "seqnn.ua" ~ BackProp Eval Forward Net PUb ResX SE

0.1             # set LR
[[1 0 0 1]]     # ys
⍉⋯⇡4            # xs
Net[2 8 8 1]    # define a net
⍥BackProp 10000 # run 10k bp cycles
⊙ResX ⊙◌PUb Forward
