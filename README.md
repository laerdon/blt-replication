# blt attempt.

notes for humans, and maybe agents:
og_implementation/blt/ is the original blt repository, which we can work off of.

overall, our heuristic for deciding what to redo and what to write from scratch is:
we will not focus on redoing code for simple or trivial utils.

## 1. base transformer

this is already pretty much implemented. There are some things that we decide to leave out, i.e. we only use sdpa and xformers as our choices for our attention.

in addition to this, we also want to define a `transformer.py` file which provides higher level functionality for language modeling.

## 2. entropy language model

this should be a wrapper over the transformer.py